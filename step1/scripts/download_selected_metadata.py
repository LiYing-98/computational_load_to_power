#!/usr/bin/env python3
"""Fetch only the pre-timeline metadata prefix of approved gated results files."""

from __future__ import annotations

import getpass
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step1.scripts.select_representative_runs import allowed_metadata_path


REPO_ID = "ml-energy/benchmark-v3"
TIMELINE_MARKERS = (b',\n  "timeline":', b',\r\n  "timeline":')
MAX_PREFIX_BYTES = 512 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_prefix(response: object) -> tuple[dict[str, object], int]:
    """Read only enough response bytes to stop immediately before embedded timeline."""
    payload = bytearray()
    for chunk in response.iter_content(chunk_size=8192):  # type: ignore[attr-defined]
        if not chunk:
            continue
        payload.extend(chunk)
        positions = [payload.find(marker) for marker in TIMELINE_MARKERS]
        positions = [position for position in positions if position >= 0]
        if positions:
            document = bytes(payload[: min(positions)]) + b"\n}"
            return json.loads(document.decode("utf-8")), len(payload)
        if len(payload) > MAX_PREFIX_BYTES:
            raise ValueError("timeline marker not found within bounded metadata prefix")
    raise ValueError("response ended before an embedded timeline marker was found")


def main() -> None:
    import requests
    from huggingface_hub import HfApi

    manifest_path = Path("step1/analysis/selected_runs.json")
    destination_root = Path("step1/data/selected_runs_raw")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest["runs"]
    if not 6 <= len(runs) <= 12:
        raise SystemExit(f"Refusing non-bounded selection: {len(runs)} files")
    for run in runs:
        if not allowed_metadata_path(run["results_path"]):
            raise SystemExit(f"Refusing unsafe target: {run['results_path']}")

    token = getpass.getpass("Hugging Face read token (input hidden): ")
    try:
        api = HfApi(token=token)
        info = api.dataset_info(REPO_ID)
        revision = info.sha
        destination_root.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Range": f"bytes=0-{MAX_PREFIX_BYTES - 1}",
            }
        )
        for index, run in enumerate(runs, start=1):
            encoded_path = "/".join(
                quote(part, safe="+") for part in run["results_path"].split("/")
            )
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{encoded_path}"
            with session.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                metadata, bytes_read = metadata_prefix(response)

            destination = destination_root / f"run_{index:02d}" / "metadata.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            run["local_metadata_path"] = destination.as_posix()
            run["metadata_bytes"] = destination.stat().st_size
            run["source_prefix_bytes_read"] = bytes_read
            run["metadata_sha256"] = sha256_file(destination)
            run["timeline_excluded"] = True
        manifest["dataset_revision"] = revision
        manifest["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["acquisition_note"] = (
            "Only the JSON prefix before the embedded timeline key was read and retained."
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Extracted {len(runs)} bounded metadata prefixes at revision {revision}; "
            "embedded timelines were not retained"
        )
    finally:
        token = ""


if __name__ == "__main__":
    main()
