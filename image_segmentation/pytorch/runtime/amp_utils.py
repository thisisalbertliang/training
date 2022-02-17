import torch.cuda.amp
try:
    import torch_xla.amp
    import torch_xla.amp.grad_scaler
except ImportError:
    print(
        "Missing packages: torch_xla.amp, torch_xla.amp.grad_scaler; "
        "these packages are available in torch-xla>=1.11"
    )

def get_grad_scaler(flags):
    if flags.torch_xla:
        scaler = torch_xla.amp.grad_scaler.GradScaler()
    else:
        scaler = torch.cuda.amp.GradScaler()
    return scaler


def get_autocast(flags):
    if flags.torch_xla:
        autocast = torch_xla.amp.autocast
    else:
        autocast = torch.cuda.amp.autocast
    return autocast
