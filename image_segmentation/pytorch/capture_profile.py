import argparse

import torch_xla.debug.profiler as xp


def main():
    parser = argparse.ArgumentParser(
        description="Performs an on-demand profiling session on provided profiler servers."
    )

    parser.add_argument(
        "--service_addr",
        dest="service_addr",
        type=str,
        required=True,
        help='comma delimited string of addresses of the profiling servers to profile. ex. "10.0.0.2:8466" or "localhost:9012".',
    )
    parser.add_argument(
        "--logdir",
        dest="logdir",
        type=str,
        required=True,
        help='the path to write profiling output to. Both the profiler client and server must have access. ex. "gs://bucket/file/path".',
    )
    parser.add_argument(
        "--duration_ms",
        dest="duration_ms",
        type=int,
        default=30000,
        help="duration in milliseconds for tracing the server.",
    )

    flags = parser.parse_args()

    xp.trace(
        service_addr=flags.service_addr,
        logdir=flags.logdir,
        duration_ms=flags.duration_ms,
    )


if __name__ == "__main__":
    main()
