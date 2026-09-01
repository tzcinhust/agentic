"""Select a deterministic train-only panel without inspecting task contents."""

from __future__ import annotations

import argparse
import hashlib

from state_bench.protocol import load_default_protocol, load_split_task_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--salt", default="latticeguard-mechanism-pilot-v1")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    protocol = load_default_protocol()
    task_ids = load_split_task_ids(args.domain, "train", protocol.split_version)
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{args.salt}:{args.domain}:{task_id}".encode()
        ).hexdigest(),
    )
    print(",".join(ranked[: args.count]))


if __name__ == "__main__":
    main()
