#!/usr/bin/env python3
"""Download the dataset files required by MESS.

Complete files are skipped, partial files are resumed with HTTP range
requests, and every completed download is checked against its published size.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
from pathlib import Path
import socket
import sys
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


COMPASS_ROOT = "https://storage.googleapis.com/compass_osdi/"
SIFT100M_ROOT = (
    "https://huggingface.co/datasets/"
    "Nanvivi/SIFT100M-DiskANN/resolve/main/"
)


@dataclass(frozen=True)
class DatasetFile:
    dataset: str
    relative_path: str
    size: int
    url: str
    digest_name: str
    digest: str


def compass_file(
    dataset: str, relative_path: str, size: int, md5: str
) -> DatasetFile:
    return DatasetFile(
        dataset=dataset,
        relative_path=relative_path,
        size=size,
        url=COMPASS_ROOT + relative_path,
        digest_name="md5",
        digest=md5,
    )


def sift100m_file(
    relative_path: str, size: int, sha256: str
) -> DatasetFile:
    filename = Path(relative_path).name
    return DatasetFile(
        dataset="sift100m",
        relative_path=relative_path,
        size=size,
        url=SIFT100M_ROOT + filename + "?download=true",
        digest_name="sha256",
        digest=sha256,
    )


FILES: Sequence[DatasetFile] = (
    compass_file(
        "sift",
        "dataset/sift/base.fvecs",
        516000000,
        "469e2c5345fa5c655db16305970e0b42",
    ),
    compass_file(
        "sift",
        "dataset/sift/query.fvecs",
        5160000,
        "f06fa8bb4e856ed05b1df3e25e76a2a0",
    ),
    compass_file(
        "sift",
        "dataset/sift/gt.ivecs",
        4040000,
        "eea2871c56a0d075306ab058f58759e3",
    ),
    compass_file(
        "laion",
        "dataset/laion1m/100k/laion_base.fvecs",
        205200000,
        "bea8fa0292d7aa3b1bcc21783231020c",
    ),
    compass_file(
        "laion",
        "dataset/laion1m/laion_query.fvecs",
        2052000,
        "70bba5231d5649add6452b3fea7e8d10",
    ),
    compass_file(
        "laion",
        "dataset/laion1m/100k/gt.ivecs",
        44000,
        "a97e43e3c1b1223986f06a6280aebccb",
    ),
    compass_file(
        "trip",
        "dataset/trip_distilbert/passages.fvecs",
        4687427196,
        "a3fa233fd71fe319a722a786b8192d6f",
    ),
    compass_file(
        "trip",
        "dataset/trip_distilbert/queries.fvecs",
        3614300,
        "bdee677bc2edc6638285505298793cf3",
    ),
    compass_file(
        "trip",
        "dataset/trip_distilbert/gt_10.ivecs",
        51700,
        "24603d7be41d39fc52cbba76e7951c1e",
    ),
    compass_file(
        "trip",
        "dataset/trip_distilbert/benchmark_tsv/documents/docs.tsv",
        2726970433,
        "65aeabe4df5e50770b42b91f2d09493a",
    ),
    compass_file(
        "trip",
        "dataset/trip_distilbert/benchmark_tsv/topics/"
        "topics.head.val.tsv",
        37561,
        "b866eb0d307b412d7bba739d1b8eced8",
    ),
    compass_file(
        "trip",
        "dataset/trip_distilbert/benchmark_tsv/qrels/"
        "qrels.dctr.head.val.tsv",
        1215928,
        "92581bb214cfe2a77864ad1aa238e90d",
    ),
    compass_file(
        "msmarco",
        "dataset/msmarco_bert/passages.fvecs",
        27197447548,
        "d3b78ae9dd0d0f775d01e74d9a75d6d6",
    ),
    compass_file(
        "msmarco",
        "dataset/msmarco_bert/queries.fvecs",
        21470480,
        "856e8e5b9dbd02dff590939a8cfae2ae",
    ),
    compass_file(
        "msmarco",
        "dataset/msmarco_bert/gt_10.ivecs",
        307120,
        "25438a12932f66a3238efe3594a9a2a6",
    ),
    compass_file(
        "msmarco",
        "dataset/msmarco_bert/passages/collection.tsv",
        3061567852,
        "31e42eeac3a3ea1b75c689c483be4484",
    ),
    compass_file(
        "msmarco",
        "dataset/msmarco_bert/passages/queries.dev.small.tsv",
        290193,
        "4621c583f1089d223db228a4f95a05d1",
    ),
    compass_file(
        "msmarco",
        "dataset/msmarco_bert/passages/qrels.dev.small.tsv",
        143300,
        "38a80559a561707ac2ec0f150ecd1e8a",
    ),
    sift100m_file(
        "dataset/sift100m/base.bin",
        12800000008,
        "1773542ad77adec09ca3ce4890cb6a69d83c82c4dcf9bf1aee789631e7dc2347",
    ),
    sift100m_file(
        "dataset/sift100m/query.bin",
        1280008,
        "eca755831fc9a8004e14886df48b81109a8ed3bfc6b632509c93c0460a30d552",
    ),
    sift100m_file(
        "dataset/sift100m/gt.bin",
        40000008,
        "ffddca9922dfa4f9f960464234bc2b4552a23c1093e98a62b870402f897720d1",
    ),
)

DATASET_ORDER = ("sift", "laion", "trip", "msmarco", "sift100m")


def expand_datasets(names: Iterable[str]) -> Sequence[str]:
    requested = list(names)
    expanded: List[str] = []
    for name in requested:
        if name == "compass":
            expanded.extend(DATASET_ORDER[:-1])
        elif name == "all":
            expanded.extend(DATASET_ORDER)
        else:
            expanded.append(name)
    return tuple(dict.fromkeys(expanded))


def human_size(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def calculate_digest(
    path: Path, digest_name: str, chunk_size: int = 8 * 1024 * 1024
) -> str:
    checksum = hashlib.new(digest_name)
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()


def inspect_file(
    spec: DatasetFile, target: Path, verify: bool
) -> str:
    if not target.is_file():
        return "missing"
    actual_size = target.stat().st_size
    if actual_size < spec.size:
        return "partial"
    if actual_size > spec.size:
        return "oversized"
    if verify:
        print(
            f"[verify] {target} ({spec.digest_name})",
            flush=True,
        )
        if calculate_digest(target, spec.digest_name) != spec.digest:
            return "checksum-mismatch"
    return "complete"


def print_progress(
    target: Path,
    current: int,
    total: int,
    started_at: float,
    starting_size: int,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    rate = (current - starting_size) / elapsed
    percent = 100.0 * current / total
    print(
        f"\r[download] {target}: {percent:6.2f}% "
        f"({human_size(current)}/{human_size(total)}, "
        f"{human_size(int(rate))}/s)",
        end="",
        flush=True,
    )


@contextmanager
def ipv4_only_resolution(enabled: bool) -> Iterator[None]:
    """Temporarily restrict urllib DNS resolution to IPv4."""
    if not enabled:
        yield
        return

    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4(
        host: str,
        port: int,
        family: int = 0,
        socktype: int = 0,
        proto: int = 0,
        flags: int = 0,
    ):
        return original_getaddrinfo(
            host, port, socket.AF_INET, socktype, proto, flags
        )

    socket.getaddrinfo = getaddrinfo_ipv4
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def is_network_unreachable(error: BaseException) -> bool:
    """Return whether an exception chain reports no route to the host."""
    current: object = error
    seen = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "errno", None) in (
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
        ):
            return True
        reason = getattr(current, "reason", None)
        if not isinstance(reason, BaseException):
            break
        current = reason
    return False


def transfer_once(
    spec: DatasetFile,
    target: Path,
    timeout: int,
    ipv4_only: bool,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    offset = target.stat().st_size if target.is_file() else 0
    headers = {"User-Agent": "MESS-dataset-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = Request(spec.url, headers=headers)
    with ipv4_only_resolution(ipv4_only):
        response = urlopen(request, timeout=timeout)
    with response:
        status = getattr(response, "status", response.getcode())
        if offset and status == 206:
            mode = "ab"
        elif offset and status == 200:
            print(
                f"[restart] server did not accept resume for {target}"
            )
            offset = 0
            mode = "wb"
        elif not offset and status in (200, 206):
            mode = "wb"
        else:
            raise RuntimeError(
                f"Unexpected HTTP status {status} for {spec.url}"
            )

        started_at = time.monotonic()
        last_update = started_at
        with target.open(mode) as output:
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                output.write(block)
                now = time.monotonic()
                if now - last_update >= 0.5:
                    print_progress(
                        target,
                        output.tell(),
                        spec.size,
                        started_at,
                        offset,
                    )
                    last_update = now
            output.flush()
            print_progress(
                target,
                output.tell(),
                spec.size,
                started_at,
                offset,
            )
    print()


def download_file(
    spec: DatasetFile,
    target: Path,
    verify: bool,
    force: bool,
    retries: int,
    timeout: int,
    ipv4_only: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    status = inspect_file(spec, target, verify and not force)

    if status == "complete" and not force:
        print(f"[skip] {target} ({human_size(spec.size)})")
        return
    if status in ("oversized", "checksum-mismatch") and not force:
        raise RuntimeError(
            f"{target} is {status}; rerun with --force to replace it"
        )
    if force and target.exists():
        target.unlink()
        status = "missing"

    if status == "partial":
        print(
            f"[resume] {target}: {human_size(target.stat().st_size)} "
            f"of {human_size(spec.size)}"
        )
    else:
        print(f"[start] {target} ({human_size(spec.size)})")

    last_error: Optional[BaseException] = None
    use_ipv4 = ipv4_only
    network_unreachable = False
    for attempt in range(1, retries + 1):
        try:
            transfer_once(spec, target, timeout, use_ipv4)
            actual_size = target.stat().st_size
            if actual_size == spec.size:
                if verify:
                    print(
                        f"[verify] {target} ({spec.digest_name})",
                        flush=True,
                    )
                    actual_digest = calculate_digest(
                        target, spec.digest_name
                    )
                    if actual_digest != spec.digest:
                        raise RuntimeError(
                            f"Checksum mismatch for {target}: "
                            f"{actual_digest} != {spec.digest}"
                        )
                print(f"[done] {target}")
                return
            if actual_size > spec.size:
                raise RuntimeError(
                    f"Downloaded file is too large: {target} "
                    f"({actual_size} > {spec.size})"
                )
            last_error = RuntimeError(
                f"Connection ended at {actual_size} of {spec.size} bytes"
            )
        except (HTTPError, URLError, OSError, RuntimeError) as error:
            last_error = error
            if is_network_unreachable(error):
                network_unreachable = True
                if not use_ipv4:
                    use_ipv4 = True
                    print(
                        "[network] no route on the default path; "
                        "retrying over IPv4 only",
                        file=sys.stderr,
                    )

        if attempt < retries:
            print(
                f"[retry {attempt}/{retries}] {target}: {last_error}",
                file=sys.stderr,
            )
            time.sleep(min(2 ** (attempt - 1), 10))

    message = (
        f"Failed to download {target} after {retries} attempts: "
        f"{last_error}"
    )
    if network_unreachable and spec.dataset == "sift100m":
        message += (
            "\nNo route to Hugging Face was available, including the IPv4 "
            "fallback. Configure HTTPS_PROXY, allow outbound HTTPS to "
            "huggingface.co and its download CDN, or download the file on "
            "another host and copy it to the path above."
        )
    raise RuntimeError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the datasets required by MESS. "
            "Complete files are skipped and partial files are resumed."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER + ("compass", "all"),
        default=["all"],
        help=(
            "datasets to download (default: all). "
            "'compass' excludes SIFT100M"
        ),
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="data directory (default: <repository>/data)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report missing or incomplete files without downloading",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="also verify published checksums (reads every byte)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing complete, oversized, or corrupt files",
    )
    parser.add_argument(
        "--ipv4",
        action="store_true",
        help="use IPv4 only (also enabled automatically after no-route errors)",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.retries < 1 or args.timeout < 1:
        raise ValueError("retries and timeout must be positive")

    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else Path(__file__).resolve().parent
    )
    selected = set(expand_datasets(args.datasets))
    selected_files = [item for item in FILES if item.dataset in selected]

    if args.check_only:
        all_complete = True
        for spec in selected_files:
            target = data_root / spec.relative_path
            status = inspect_file(spec, target, args.verify)
            if status == "complete":
                print(f"[OK]         {target}")
            else:
                current = target.stat().st_size if target.is_file() else 0
                print(
                    f"[{status.upper():<12}] {target}: "
                    f"{current:,}/{spec.size:,} bytes"
                )
                all_complete = False
        return 0 if all_complete else 1

    for spec in selected_files:
        download_file(
            spec=spec,
            target=data_root / spec.relative_path,
            verify=args.verify,
            force=args.force,
            retries=args.retries,
            timeout=args.timeout,
            ipv4_only=args.ipv4,
        )

    print("Dataset download completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
