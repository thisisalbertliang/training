import torch_xla.debug.profiler as xp


def main():
    xp.trace(
        service_addr=f"localhost:{9111}",
        logdir="gs://albert-datasets-test/tensorboard/unet3d",
        duration_ms=30000,
    )


if __name__ == '__main__':
    main()
