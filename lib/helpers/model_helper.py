from lib.models.MonoMH import MonoMH

def build_model(cfg, mean_size):
    if cfg['type'] == 'MonoMH':
        # Pass enhancement config to model if available
        enhancements_cfg = cfg.get('enhancements', {})
        return MonoMH(backbone=cfg['backbone'], neck=cfg['neck'],
                      mean_size=mean_size, enhancements_cfg=enhancements_cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
