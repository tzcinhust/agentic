"""Create a deterministic train-only build/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _rank(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", default="shopping_assistant")
    args = parser.parse_args()

    source = args.source / args.domain
    build_dir = args.output / "build" / args.domain
    validation_dir = args.output / "validation" / args.domain
    build_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(source.glob("*.json"))
    if len(files) != 100:
        raise ValueError(f"Expected 100 train trajectories, found {len(files)}")
    validation_names = {
        path.stem for path in sorted(files, key=lambda item: _rank(item.stem))[:20]
    }
    for path in files:
        destination = validation_dir if path.stem in validation_names else build_dir
        shutil.copy2(path, destination / path.name)

    manifest = {
        "domain": args.domain,
        "source": str(source),
        "build_count": len(list(build_dir.glob("*.json"))),
        "validation_count": len(list(validation_dir.glob("*.json"))),
        "validation_task_ids": sorted(path.stem for path in validation_dir.glob("*.json")),
    }
    if manifest["validation_count"] != 20 or manifest["build_count"] != 80:
        raise AssertionError(manifest)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
