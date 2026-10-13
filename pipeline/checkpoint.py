"""Validate complete training checkpoints before any live state is modified."""
import torch


def validate_checkpoint(checkpoint, network, n_in_row=None):
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 2:
        raise ValueError('Unsupported checkpoint version. Legacy parameter-only checkpoints lack BatchNorm statistics; retain them and start a new training run.')
    if checkpoint.get('model_spec') != network.checkpoint_spec():
        raise ValueError('Checkpoint architecture/board dimensions do not match this network.')
    if n_in_row is not None and checkpoint.get('n_in_row') != n_in_row:
        raise ValueError('Checkpoint game rules do not match n_in_row.')
    network.validate_state(checkpoint.get('model_state'))
    if n_in_row is None:
        return  # Inference does not restore optimizer or replay state.
    optimizer = checkpoint.get('optimizer_state')
    params = network.get_all_params()[0]
    if not isinstance(optimizer, dict) or set(optimizer.get('state', {})) != set(params):
        raise ValueError('Incomplete optimizer state.')
    for name, param in params.items():
        moments = optimizer['state'][name]
        if not isinstance(moments, dict) or set(moments) != {'m', 'v'}:
            raise ValueError('Invalid Adam state: '+name)
        for tensor in moments.values():
            if not torch.is_tensor(tensor) or tensor.shape != param.shape or not torch.isfinite(tensor).all():
                raise ValueError('Incompatible optimizer tensor: '+name)
    size = network.board_size
    for sample in checkpoint.get('data_buffer') or []:
        if len(sample) != 3 or tuple(sample[0].shape) != (4, size, size) or tuple(sample[1].shape) not in {(size*size,), (size*size+1,)}:
            raise ValueError('Replay buffer dimensions do not match the checkpoint.')
