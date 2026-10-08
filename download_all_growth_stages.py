"""
download_all_growth_stages.py

Parallel resumable S3 downloader for all 8,486 growth-stage RGB images (_D.JPG)
from IEEE DataPort S3 bucket: s3://ieee-dataport/data/1426491/100367/

Features:
- Extracts all growth-stage _D.JPG URIs directly from full_manifest.txt.
- Checks local destination and existing directories to skip already downloaded images.
- Enforces disk space safety (stops/warns if free space < 3.0 GB).
- Multi-threaded download with configurable worker count.
- Updates download_progress.txt in real-time.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

S3_PREFIX = "s3://ieee-dataport/data/1426491/100367/"
STAGES = ["nursery", "vegetative", "booting", "flowering", "mature"]
MIN_FREE_GB = 3.0  # Stop if drive free space drops below 3 GB

def extract_uris_from_manifest(manifest_path: Path) -> list[str]:
    uris = []
    with open(manifest_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                p = parts[3]
                if p.endswith("_D.JPG"):
                    low = p.lower()
                    if any(s in low for s in STAGES):
                        uris.append(f"s3://ieee-dataport/{p}")
    return uris

def check_disk_space(path: Path) -> float:
    """Returns free space in GB on the drive containing path."""
    try:
        total, used, free = shutil.disk_usage(path.anchor if path.anchor else path)
        return free / (1024 ** 3)
    except Exception:
        return 999.0

def download_single(uri: str, dest_root: Path) -> Tuple[str, str, str, int]:
    uri = uri.strip()
    if not uri:
        return (uri, "SKIP", "Empty URI", 0)

    match = re.search(r"Field\d+.*", uri)
    if match:
        rel_path = match.group(0)
    else:
        rel_path = os.path.basename(uri)

    local_path = dest_root / rel_path

    # Check if already present and valid
    if local_path.exists() and local_path.stat().st_size > 1000:
        return (rel_path, "SKIP", f"{local_path.stat().st_size} bytes", local_path.stat().st_size)

    # Check disk safety
    free_gb = check_disk_space(dest_root)
    if free_gb < MIN_FREE_GB:
        return (rel_path, "DISK_FULL", f"Free space {free_gb:.2f}GB < {MIN_FREE_GB}GB limit", 0)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", uri, str(local_path), "--only-show-errors"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        if res.returncode == 0 and local_path.exists() and local_path.stat().st_size > 1000:
            sz = local_path.stat().st_size
            return (rel_path, "OK", f"{sz} bytes", sz)
        else:
            if local_path.exists():
                local_path.unlink(missing_ok=True)
            err = res.stderr.strip() or f"exit code {res.returncode}"
            return (rel_path, "FAIL", err, 0)
    except Exception as e:
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        return (rel_path, "FAIL", str(e), 0)

def update_progress(progress_file: Path, completed: int, total: int, ok: int, skip: int, fail: int, disk_full: int, total_bytes: int, elapsed: float):
    pct = (completed / total * 100) if total else 0
    mb = total_bytes / (1024 * 1024)
    speed = mb / elapsed if elapsed > 0 else 0
    bar_len = 25
    filled = int(bar_len * completed // total) if total else 0
    bar = "=" * filled + "-" * (bar_len - filled)

    text = (
        f"==========================================================\n"
        f" DOWNLOADING ALL GROWTH-STAGE RGB IMAGES (8,486 Total)\n"
        f"==========================================================\n"
        f" Progress   : [{bar}] {pct:.1f}%\n"
        f" Completed  : {completed:,} / {total:,} files\n"
        f" Downloaded : {ok:,} new ({mb:,.1f} MB, {speed:.2f} MB/s)\n"
        f" Skipped    : {skip:,} (already existed on disk)\n"
        f" Failed     : {fail:,}\n"
        f" Disk Safety: {'TRIGGERED (Drive nearly full)' if disk_full > 0 else 'NORMAL'}\n"
        f" Elapsed    : {elapsed:.1f}s\n"
        f" Status     : {'COMPLETE' if completed == total else 'IN_PROGRESS'}\n"
        f"==========================================================\n"
    )
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Download all 8,486 growth-stage RGB images")
    parser.add_argument("--manifest", type=Path, default=Path("full_manifest.txt"), help="full_manifest.txt path")
    parser.add_argument("--dest", type=Path, default=Path("test_500_rgb_images"), help="Target download directory")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent worker threads")
    parser.add_argument("--progress-file", type=Path, default=Path("download_progress.txt"), help="Live progress file")
    args = parser.parse_args()

    dest_root = args.dest.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Initializing Download for All Growth-Stage RGB Images ===", flush=True)
    print(f"Destination: {dest_root}", flush=True)
    print(f"Workers    : {args.workers}", flush=True)
    print(f"Free space : {check_disk_space(dest_root):.2f} GB", flush=True)

    uris = extract_uris_from_manifest(args.manifest)
    total = len(uris)
    print(f"Total URIs extracted from manifest: {total}", flush=True)

    completed = 0
    ok_count = 0
    skip_count = 0
    fail_count = 0
    disk_full_count = 0
    total_bytes = 0

    start_time = time.time()
    update_progress(args.progress_file, 0, total, 0, 0, 0, 0, 0, 0.1)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single, u, dest_root): u for u in uris}
        for fut in as_completed(futures):
            rel_path, status, msg, sz = fut.result()
            completed += 1
            if status == "OK":
                ok_count += 1
                total_bytes += sz
            elif status == "SKIP":
                skip_count += 1
            elif status == "DISK_FULL":
                disk_full_count += 1
                # Cancel remaining if disk full
                if disk_full_count >= 10:
                    print("\n[WARNING] Free disk space low (<3GB). Pausing download to protect disk.", flush=True)
                    for f in futures:
                        f.cancel()
                    break
            else:
                fail_count += 1

            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - start_time
                update_progress(args.progress_file, completed, total, ok_count, skip_count, fail_count, disk_full_count, total_bytes, elapsed)
                print(f"[{completed}/{total}] OK: {ok_count} | Skip: {skip_count} | Fail: {fail_count} | DiskFull: {disk_full_count}", flush=True)

    elapsed = time.time() - start_time
    update_progress(args.progress_file, completed, total, ok_count, skip_count, fail_count, disk_full_count, total_bytes, elapsed)
    print(f"\nDownload run finished in {elapsed:.1f}s. OK: {ok_count}, Skipped: {skip_count}, Failed: {fail_count}", flush=True)

if __name__ == "__main__":
    main()
