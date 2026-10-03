import os
import torch


def get_checkpoint_state(model=None, optimizer=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state}


def get_full_checkpoint_state(model=None, optimizer=None, epoch=None,
                              lr_scheduler=None, warmup_lr_scheduler=None,
                              best_results=None, prototype_bank=None):
    """Extended checkpoint that captures full training state for resume."""
    state = get_checkpoint_state(model, optimizer, epoch)
    if lr_scheduler is not None:
        state['lr_scheduler_state'] = lr_scheduler.state_dict()
    if warmup_lr_scheduler is not None:
        state['warmup_lr_scheduler_state'] = warmup_lr_scheduler.state_dict()
    if best_results is not None:
        state['best_results'] = best_results
    if prototype_bank is not None:
        state['prototype_bank_state'] = prototype_bank.state_dict()
    return state


def save_checkpoint(state, filename, logger):
    logger.info("==> Saving to checkpoint '{}'".format(filename))
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, logger, map_location):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
        epoch = checkpoint.get('epoch', -1)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(map_location)

        logger.info("==> Done")
    else:
        raise FileNotFoundError
    return epoch


def load_full_checkpoint(checkpoint, model, optimizer, lr_scheduler=None,
                         warmup_lr_scheduler=None, logger=None, map_location='cuda:0'):
    """Restore full training state from an extended checkpoint."""
    epoch = checkpoint.get('epoch', -1)

    if model is not None and checkpoint.get('model_state') is not None:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        if logger:
            logger.info("  Restored model weights")

    if optimizer is not None and checkpoint.get('optimizer_state') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(map_location)
        if logger:
            logger.info("  Restored optimizer state")

    if lr_scheduler is not None and 'lr_scheduler_state' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        if logger:
            logger.info("  Restored LR scheduler state")

    if warmup_lr_scheduler is not None and 'warmup_lr_scheduler_state' in checkpoint:
        warmup_lr_scheduler.load_state_dict(checkpoint['warmup_lr_scheduler_state'])
        if logger:
            logger.info("  Restored warmup LR scheduler state")

    return epoch, checkpoint.get('best_results'), checkpoint.get('prototype_bank_state')