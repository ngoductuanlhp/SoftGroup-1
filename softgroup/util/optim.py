import torch.optim


def build_optimizer(model, optim_cfg):
    assert "type" in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop("type")
    optim = getattr(torch.optim, optim_type)

    # backbone_modules = ['input_conv', 'unet', 'output_layer', 'semantic_linear', 'offset_linear', 'offset_vertices_linear', 'box_conf_linear']
    # backbone_params = []
    # head_modules = ['set_aggregator', 'pos_embedding', 'query_projection', 'encoder_to_decoder_projection', 'detr_sem_head', 'detr_conf_head', 'decoder',\
    #                 'mask_tower', 'before_embedding_tower', 'controller']
    # head_params = []
    # # for mod_name in backbone_modules:
    # for mod_name in backbone_modules:
    #     backbone_params += getattr(model, mod_name).parameters()
    # for mod_name in head_modules:
    #     head_params += getattr(model, mod_name).parameters()

    # params_list = [
    #             {'params': backbone_params, 'lr': _optim_cfg.lr / 10.0},
    #             {'params': head_params, 'lr': _optim_cfg.lr}
    #         ]

    # return optim(params_list)
    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)
