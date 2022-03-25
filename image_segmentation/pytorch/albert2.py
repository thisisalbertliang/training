import torch_xla.debug.profiler as xp

if __name__ == "__main__":

    port = 9001

    xp.trace(
        service_addr=f"localhost:{port}",
        logdir="dummy_trace_logdir",
        duration_ms=3000,
    )
