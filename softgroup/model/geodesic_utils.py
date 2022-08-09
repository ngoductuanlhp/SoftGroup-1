from numpy import dtype
import torch
from softgroup.pointnet2.pointnet2_utils import ball_query_dist
from softgroup.ops.functions import knnquery

def unique_with_inds(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


@torch.no_grad()
def find_knn(gpu_index, locs, neighbor=32):
    n_points = locs.shape[0]
    # Search with torch GPU using pre-allocated arrays
    # new_d_torch_gpu = torch.zeros(n_points, neighbor, device=locs.device, dtype=torch.float32)
    # new_i_torch_gpu = torch.zeros(n_points, neighbor, device=locs.device, dtype=torch.int64)

    gpu_index.add(locs)

    new_d_torch_gpu, new_i_torch_gpu = gpu_index.search(locs, neighbor)
    gpu_index.reset()
    new_d_torch_gpu = torch.sqrt(new_d_torch_gpu)

    return new_d_torch_gpu, new_i_torch_gpu


@torch.no_grad()
def cal_geodesic_single(
    gpu_index, pre_enc_inds, locs_float_, batch_offset_, max_step=32, neighbor=32, radius=0.1, n_queries=128
):

    batch_size = pre_enc_inds.shape[0]
    geo_dists = []
    for b in range(batch_size):
        start = batch_offset_[b]
        end = batch_offset_[b + 1]

        query_inds = pre_enc_inds[b][:n_queries]
        locs_float_b = locs_float_[start:end]

        n_points = end - start

        new_d_torch_gpu, new_i_torch_gpu = find_knn(gpu_index, locs_float_b, neighbor=neighbor)

        geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_.device) - 1
        visited = torch.zeros((n_queries, n_points), dtype=torch.bool, device=locs_float_.device)

        for q in range(n_queries):
            D_geo, I_geo = new_d_torch_gpu[query_inds[q]], new_i_torch_gpu[query_inds[q]]

            indices, distances = I_geo[1:].reshape(-1), D_geo[1:].reshape(-1)

            cond = ((distances <= radius) & (indices >= 0)).bool()

            distances = distances[cond]
            indices = indices[cond]

            for it in range(max_step):

                indices_unique, corres_inds = unique_with_inds(indices)
                distances_uniques = distances[corres_inds]

                inds = torch.nonzero((visited[q, indices_unique] is False)).view(-1)

                if len(inds) < neighbor // 2:
                    break
                indices_unique = indices_unique[inds]
                distances_uniques = distances_uniques[inds]

                geo_dist[q, indices_unique] = distances_uniques
                visited[q, indices_unique] = True

                D_geo, I_geo = new_d_torch_gpu[indices_unique][:, 1:], new_i_torch_gpu[indices_unique][:, 1:]

                D_geo_cumsum = D_geo + distances_uniques.unsqueeze(-1)

                indices, distances_local, distances_global = (
                    I_geo.reshape(-1),
                    D_geo.reshape(-1),
                    D_geo_cumsum.reshape(-1),
                )
                cond = (distances_local <= radius) & (indices >= 0)
                distances = distances_global[cond]
                indices = indices[cond]
        geo_dists.append(geo_dist)
        del new_d_torch_gpu, new_i_torch_gpu
    return geo_dists


# NOTE fastest way to cal geodesic distance
@torch.no_grad()
def cal_geodesic_vectorize(
    gpu_index, pre_enc_inds, locs_float_, batch_offset_, max_step=128, neighbor=64, radius=0.05, n_queries=128
):

    batch_size = pre_enc_inds.shape[0]
    geo_dists = []
    for b in range(batch_size):
        start = batch_offset_[b]
        end = batch_offset_[b + 1]

        query_inds = pre_enc_inds[b][:n_queries].long()
        locs_float_b = locs_float_[start:end]

        n_points = end - start

        distances_arr, indices_arr = find_knn(gpu_index, locs_float_b, neighbor=neighbor)

        # NOTE nearest neigbor is themself -> remove first element
        distances_arr = distances_arr[:, 1:]
        indices_arr = indices_arr[:, 1:]


        geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_.device) - 1
        visited = torch.zeros((n_queries, n_points), dtype=torch.bool, device=locs_float_.device)

        arange_tensor = torch.arange(0, n_queries, dtype=torch.long, device=locs_float_.device)

        geo_dist[arange_tensor, query_inds] = 0.0
        visited[arange_tensor, query_inds] = True

        distances, indices = distances_arr[query_inds], indices_arr[query_inds]  # N_queries x n_neighbors

        cond = (distances <= radius) & (indices >= 0)  # N_queries x n_neighbors

        queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp
        points_inds = indices[queries_inds, neighbors_inds]  # n_temp
        points_distances = distances[queries_inds, neighbors_inds]  # n_temp

        for step in range(max_step):
            # NOTE find unique indices for each query
            stack_pointquery_inds = torch.stack([points_inds, queries_inds], dim=0)
            _, unique_inds = unique_with_inds(stack_pointquery_inds)

            points_inds = points_inds[unique_inds]
            queries_inds = queries_inds[unique_inds]
            points_distances = points_distances[unique_inds]

            # NOTE update geodesic and visited look-up table
            geo_dist[queries_inds, points_inds] = points_distances
            visited[queries_inds, points_inds] = True

            # NOTE get new neighbors
            distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds]  # n_temp x n_neighbors
            distances_new_cumsum = distances_new + points_distances[:, None]  # n_temp x n_neighbors

            # NOTE trick to repeat queries indices for new neighbor
            queries_inds = queries_inds[:, None].repeat(1, neighbor - 1)  # n_temp x n_neighbors

            # NOTE condition: no visited and radius and indices
            visited_cond = visited[queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
            cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond is False)  # n_temp x n_neighbors

            # NOTE filter
            temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp2

            if len(temp_inds) == 0:  # no new points:
                break

            points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
            points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
            queries_inds = queries_inds[temp_inds, neighbors_inds]  # n_temp2

        geo_dists.append(geo_dist)
    return geo_dists


# NOTE fastest way to cal geodesic distance
@torch.no_grad()
def cal_geodesic_vectorize_batch(gpu_index, context_locs, max_step=128, neighbor=64, radius=0.05, n_queries=128):
    # context_locs: b, n_context, 3
    batch_size, n_points = context_locs.shape[:2]
    geo_dists = torch.zeros((batch_size, n_queries, n_points), dtype=torch.float32, device=context_locs.device) - 1
    visiteds = torch.zeros((batch_size, n_queries, n_points), dtype=torch.bool, device=context_locs.device)

    # arange_tensor = torch.arange(0, n_queries, dtype=torch.long, device=context_locs.device)
    query_inds = torch.arange(0, n_queries, dtype=torch.long, device=context_locs.device)

    geo_dists[:, query_inds, query_inds] = 0.0
    visiteds[:, query_inds, query_inds] = True

    for b in range(batch_size):

        distances_arr, indices_arr = find_knn(gpu_index, context_locs[b], neighbor=neighbor)

        # NOTE nearest neigbor is themself -> remove first element
        distances_arr = distances_arr[:, 1:]
        indices_arr = indices_arr[:, 1:]

        # geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=context_locs.device) - 1
        # visited = torch.zeros((n_queries, n_points), dtype=torch.bool, device=context_locs.device)

        # geo_dist[arange_tensor, query_inds] = 0.0
        # visited[arange_tensor, query_inds] = True

        distances, indices = distances_arr[query_inds], indices_arr[query_inds]  # N_queries x n_neighbors

        cond = (distances <= radius) & (indices >= 0)  # N_queries x n_neighbors

        queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp
        points_inds = indices[queries_inds, neighbors_inds]  # n_temp
        points_distances = distances[queries_inds, neighbors_inds]  # n_temp

        for step in range(max_step):
            # NOTE find unique indices for each query
            stack_pointquery_inds = torch.stack([points_inds, queries_inds], dim=0)
            _, unique_inds = unique_with_inds(stack_pointquery_inds)

            points_inds = points_inds[unique_inds]
            queries_inds = queries_inds[unique_inds]
            points_distances = points_distances[unique_inds]

            # NOTE update geodesic and visited look-up table
            geo_dists[b, queries_inds, points_inds] = points_distances
            visiteds[b, queries_inds, points_inds] = True

            # NOTE get new neighbors
            distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds]  # n_temp x n_neighbors
            distances_new_cumsum = distances_new + points_distances[:, None]  # n_temp x n_neighbors

            # NOTE trick to repeat queries indices for new neighbor
            queries_inds = queries_inds[:, None].repeat(1, neighbor - 1)  # n_temp x n_neighbors

            # NOTE condition: no visited and radius and indices
            visited_cond = visiteds[b, queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
            cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond is False)  # n_temp x n_neighbors

            # NOTE filter
            temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp2

            if len(temp_inds) == 0:  # no new points:
                break

            points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
            points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
            queries_inds = queries_inds[temp_inds, neighbors_inds]  # n_temp2

    return geo_dists

# NOTE fastest way to cal geodesic distance
@torch.no_grad()
def cal_geodesic_vectorize2(
    gpu_index, query_inds, locs_float_b, max_step=16, neighbor=64, radius=0.05, n_queries=192, max_dist=10000, n_sample=256
):

    # batch_size = pre_enc_inds.shape[0]
    # geo_dists = []
    # for b in range(batch_size):
    #     start = batch_offset_[b]
    #     end = batch_offset_[b + 1]

    # query_inds = pre_enc_inds[b][:n_queries].long()
    # locs_float_b = locs_float_[start:end]

    query_inds = query_inds.long()
    n_points = locs_float_b.shape[0]

    distances_arr, indices_arr = find_knn(gpu_index, locs_float_b, neighbor=neighbor)

    # NOTE nearest neigbor is themself -> remove first element
    distances_arr = distances_arr[:, 1:]
    indices_arr = indices_arr[:, 1:]


    # indices_arr_debug = torch.sum((distances_arr <= radius), dim=-1).float()
    # print('debug knn', torch.mean(indices_arr_debug))
    # indices_arr_debug = torch.sum((indices_arr >= 0), dim=-1).float()
    # print('debug knn', torch.mean(indices_arr_debug))

    geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_b.device) + max_dist
    visited = torch.zeros((n_queries, n_points), dtype=torch.int, device=locs_float_b.device)

    arange_tensor = torch.arange(0, n_queries, dtype=torch.long, device=locs_float_b.device)

    geo_dist[arange_tensor, query_inds] = 0.0
    visited[arange_tensor, query_inds] = 1

    distances, indices = distances_arr[query_inds], indices_arr[query_inds]  # N_queries x n_neighbors

    cond = (distances <= radius) & (indices >= 0)  # N_queries x n_neighbors

    queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp
    points_inds = indices[queries_inds, neighbors_inds]  # n_temp
    points_distances = distances[queries_inds, neighbors_inds]  # n_temp

    for step in range(max_step):
        # NOTE find unique indices for each query
        stack_pointquery_inds = torch.stack([points_inds, queries_inds], dim=0)
        _, unique_inds = unique_with_inds(stack_pointquery_inds)

        points_inds = points_inds[unique_inds]
        queries_inds = queries_inds[unique_inds]
        points_distances = points_distances[unique_inds]

        # NOTE update geodesic and visited look-up table
        geo_dist[queries_inds, points_inds] = points_distances
        visited[queries_inds, points_inds] = 1

        # NOTE get new neighbors
        distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds]  # n_temp x n_neighbors
        distances_new_cumsum = distances_new + points_distances[:, None]  # n_temp x n_neighbors

        # NOTE trick to repeat queries indices for new neighbor
        queries_inds = queries_inds[:, None].repeat(1, neighbor - 1)  # n_temp x n_neighbors

        # NOTE condition: no visited and radius and indices
        visited_cond = visited[queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
        cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond == 0)  # n_temp x n_neighbors

        # NOTE filter
        temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp2
        # print(f'step {step}, len {len(temp_inds)}')
        if len(temp_inds) == 0:  # no new points:
            break

        points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
        points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
        queries_inds = queries_inds[temp_inds, neighbors_inds]  # n_temp2

    del distances_arr, indices_arr

    geo_dist[visited==0] = max_dist # N_queries, n_points

    # debug_visited = (torch.sum(visited, dim=-1).float())
    # print('mean visited', torch.mean(debug_visited), torch.min(debug_visited), torch.max(debug_visited))

    geo_neighbors_dist, geo_neighbors_inds = torch.topk(geo_dist, k=n_sample, dim=-1, largest=False) # n_queries, k
    geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds = torch.nonzero(geo_neighbors_dist == max_dist, as_tuple=True)
    # print('debug', len(geo_neighbors_dist_queries_inds)/n_queries, 256)
    query_inds_repeat = query_inds[geo_neighbors_dist_queries_inds]
    geo_neighbors_inds[geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds] = query_inds_repeat
    geo_neighbors_dist[geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds] = 0

    return geo_neighbors_dist, geo_neighbors_inds # n_queries, k
    # return geo_dist
    # geo_dists.append(geo_dist)
    # return geo_dists


# NOTE fastest way to cal geodesic distance
@torch.no_grad()
def cal_geodesic_vectorize3(
    gpu_index, query_inds, locs_float_b, max_step=16, neighbor=64, radius=0.05, n_queries=192, max_dist=10000, n_sample=256
):

    # batch_size = pre_enc_inds.shape[0]
    # geo_dists = []
    # for b in range(batch_size):
    #     start = batch_offset_[b]
    #     end = batch_offset_[b + 1]

    # query_inds = pre_enc_inds[b][:n_queries].long()
    # locs_float_b = locs_float_[start:end]

    query_inds = query_inds.long()
    n_points = locs_float_b.shape[0]

    # distances_arr, indices_arr = find_knn(gpu_index, locs_float_b, neighbor=neighbor)



    # # NOTE nearest neigbor is themself -> remove first element
    # distances_arr = distances_arr[:, 1:]
    # indices_arr = indices_arr[:, 1:]

    # query_locs = locs_float_b[query_inds, :]

    locs_contiguous = locs_float_b.unsqueeze(0).contiguous()
    indices_arr, distances_arr = ball_query_dist(radius, neighbor-1, locs_contiguous, locs_contiguous) # 1, npoint, nsample
    indices_arr = indices_arr[0].long()
    distances_arr = torch.sqrt(distances_arr[0].float())

    # indices_arr_debug = torch.sum((distances_arr <= radius), dim=-1).float()
    # print('debug knn', torch.mean(indices_arr_debug))
    # indices_arr_debug = torch.sum((indices_arr >= 0), dim=-1).float()
    # print('debug knn', torch.mean(indices_arr_debug))

    geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_b.device) + max_dist
    visited = torch.zeros((n_queries, n_points), dtype=torch.int, device=locs_float_b.device)

    arange_tensor = torch.arange(0, n_queries, dtype=torch.long, device=locs_float_b.device)

    geo_dist[arange_tensor, query_inds] = 0.0
    visited[arange_tensor, query_inds] = 1

    distances, indices = distances_arr[query_inds], indices_arr[query_inds]  # N_queries x n_neighbors

    cond = (distances <= radius) & (indices >= 0)  # N_queries x n_neighbors

    queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp
    points_inds = indices[queries_inds, neighbors_inds]  # n_temp
    points_distances = distances[queries_inds, neighbors_inds]  # n_temp

    for step in range(max_step):
        # NOTE find unique indices for each query
        stack_pointquery_inds = torch.stack([points_inds, queries_inds], dim=0)
        _, unique_inds = unique_with_inds(stack_pointquery_inds)

        points_inds = points_inds[unique_inds]
        queries_inds = queries_inds[unique_inds]
        points_distances = points_distances[unique_inds]

        # NOTE update geodesic and visited look-up table
        geo_dist[queries_inds, points_inds] = points_distances
        visited[queries_inds, points_inds] = 1

        # NOTE get new neighbors
        distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds]  # n_temp x n_neighbors
        distances_new_cumsum = distances_new + points_distances[:, None]  # n_temp x n_neighbors

        # NOTE trick to repeat queries indices for new neighbor
        queries_inds = queries_inds[:, None].repeat(1, neighbor - 1)  # n_temp x n_neighbors

        # NOTE condition: no visited and radius and indices
        visited_cond = visited[queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
        cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond == 0)  # n_temp x n_neighbors

        # NOTE filter
        temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp2
        # print(f'step {step}, len {len(temp_inds)}')
        if len(temp_inds) == 0:  # no new points:
            break

        points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
        points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
        queries_inds = queries_inds[temp_inds, neighbors_inds]  # n_temp2


    geo_dist[visited==0] = max_dist # N_queries, n_points

    # debug_visited = (torch.sum(visited, dim=-1).float())
    # print('mean visited', torch.mean(debug_visited), torch.min(debug_visited), torch.max(debug_visited))

    geo_neighbors_dist, geo_neighbors_inds = torch.topk(geo_dist, k=n_sample, dim=-1, largest=False) # n_queries, k
    geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds = torch.nonzero(geo_neighbors_dist == max_dist, as_tuple=True)
    # print('debug', len(geo_neighbors_dist_queries_inds)/n_queries, 256)
    query_inds_repeat = query_inds[geo_neighbors_dist_queries_inds]
    geo_neighbors_inds[geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds] = query_inds_repeat
    geo_neighbors_dist[geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds] = 0

    return geo_neighbors_dist, geo_neighbors_inds # n_queries, k

# NOTE fastest way to cal geodesic distance
@torch.no_grad()
def cal_geodesic_vectorize4(
    query_inds, locs_float_b, max_step=16, neighbor=64, radius=0.05, n_queries=192, max_dist=10000, n_sample=256
):

    # batch_size = pre_enc_inds.shape[0]
    # geo_dists = []
    # for b in range(batch_size):
    #     start = batch_offset_[b]
    #     end = batch_offset_[b + 1]

    # query_inds = pre_enc_inds[b][:n_queries].long()
    # locs_float_b = locs_float_[start:end]

    query_inds = query_inds.long()
    n_points = locs_float_b.shape[0]

    # distances_arr, indices_arr = find_knn(gpu_index, locs_float_b, neighbor=neighbor)
    offset = torch.Tensor([0, n_points]).int().to(locs_float_b.device)
    indices_arr, distances_arr = knnquery(neighbor, locs_float_b, locs_float_b, offset, offset)

    # NOTE nearest neigbor is themself -> remove first element
    distances_arr = distances_arr[:, 1:]
    indices_arr = indices_arr[:, 1:].long()


    # indices_arr_debug = torch.sum((distances_arr <= radius), dim=-1).float()
    # print('debug knn', torch.mean(indices_arr_debug))
    # indices_arr_debug = torch.sum((indices_arr >= 0), dim=-1).float()
    # print('debug knn', torch.mean(indices_arr_debug))

    geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_b.device) + max_dist
    visited = torch.zeros((n_queries, n_points), dtype=torch.int, device=locs_float_b.device)

    arange_tensor = torch.arange(0, n_queries, dtype=torch.long, device=locs_float_b.device)

    geo_dist[arange_tensor, query_inds] = 0.0
    visited[arange_tensor, query_inds] = 1

    distances, indices = distances_arr[query_inds], indices_arr[query_inds]  # N_queries x n_neighbors

    cond = (distances <= radius) & (indices >= 0)  # N_queries x n_neighbors

    queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp
    points_inds = indices[queries_inds, neighbors_inds]  # n_temp
    points_distances = distances[queries_inds, neighbors_inds]  # n_temp

    for step in range(max_step):
        # NOTE find unique indices for each query
        stack_pointquery_inds = torch.stack([points_inds, queries_inds], dim=0)
        _, unique_inds = unique_with_inds(stack_pointquery_inds)

        points_inds = points_inds[unique_inds]
        queries_inds = queries_inds[unique_inds]
        points_distances = points_distances[unique_inds]

        # NOTE update geodesic and visited look-up table
        geo_dist[queries_inds, points_inds] = points_distances
        visited[queries_inds, points_inds] = 1

        # NOTE get new neighbors
        distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds]  # n_temp x n_neighbors
        distances_new_cumsum = distances_new + points_distances[:, None]  # n_temp x n_neighbors

        # NOTE trick to repeat queries indices for new neighbor
        queries_inds = queries_inds[:, None].repeat(1, neighbor - 1)  # n_temp x n_neighbors

        # NOTE condition: no visited and radius and indices
        visited_cond = visited[queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
        cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond == 0)  # n_temp x n_neighbors

        # NOTE filter
        temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp2
        # print(f'step {step}, len {len(temp_inds)}')
        if len(temp_inds) == 0:  # no new points:
            break

        points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
        points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
        queries_inds = queries_inds[temp_inds, neighbors_inds]  # n_temp2

    del distances_arr, indices_arr

    geo_dist[visited==0] = max_dist # N_queries, n_points

    # debug_visited = (torch.sum(visited, dim=-1).float())
    # print('mean visited', torch.mean(debug_visited), torch.min(debug_visited), torch.max(debug_visited))

    geo_neighbors_dist, geo_neighbors_inds = torch.topk(geo_dist, k=n_sample, dim=-1, largest=False) # n_queries, k
    geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds = torch.nonzero(geo_neighbors_dist == max_dist, as_tuple=True)
    # print('debug', len(geo_neighbors_dist_queries_inds)/n_queries, 256)
    query_inds_repeat = query_inds[geo_neighbors_dist_queries_inds]
    geo_neighbors_inds[geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds] = query_inds_repeat
    geo_neighbors_dist[geo_neighbors_dist_queries_inds, geo_neighbors_dist_points_inds] = 0

    return geo_neighbors_dist, geo_neighbors_inds # n_queries, k
    # return geo_dist
    # geo_dists.append(geo_dist)
    # return geo_dists
