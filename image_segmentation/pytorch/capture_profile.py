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
    parser.add_argument(
        "--interactive",
        dest="interactive",
        type=str,
        choices=[None, "once", "loop"],
        default=None,
        help=(
            "run in interactive mode.\n"
            'If set to "once", the profiler client asks for user confirmation before starting profiling.\n'
            'If set to "loop", the profiler client repeatedly runs profiling, asking for user confirmation on each run.\n'
            "Defaults to None, which disables interactive mode."
        ),
    )

    flags = parser.parse_args()

    def trace():
        xp.trace(
            service_addr=flags.service_addr,
            logdir=flags.logdir,
            duration_ms=flags.duration_ms,
        )
        print(f"Saved profiling output to {flags.logdir}")

    def request_user_confirmation():
        input("Press enter to start profiling:")

    # Run performance profiling
    if flags.interactive == "once":
        request_user_confirmation()
        trace()
    elif flags.interactive == "loop":
        while True:
            request_user_confirmation()
            trace()
    else:
        trace()


if __name__ == "__main__":
    main()
