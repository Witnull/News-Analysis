#!/usr/bin/env python3
"""Download a Hugging Face model repository into ./models/hf/hub/<model>.

HF_TOKEN is auto-loaded from a .env file (via python-dotenv) if present;
falls back to the HF_TOKEN environment variable. A warning is printed if
no token is found anywhere (public repos still work without one).

Examples:
    python download_hf_model.py Qwen/Qwen2.5-7B-Instruct --raw-model
    python download_hf_model.py Qwen/Qwen2.5-7B-Instruct --for-gguf
    python download_hf_model.py bartowski/Qwen2.5-7B-Instruct-GGUF --gguf
    python download_hf_model.py bartowski/Qwen2.5-7B-Instruct-GGUF --gguf -q Q5_K_M
    python download_hf_model.py org/model --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

try:
    from dotenv import find_dotenv, load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = PROJECT_ROOT / "models" / "hf" / "hub"

CONFIG_PATTERNS = [
    "*.json",
    "*.model",
    "*.py",
    "*.jinja",
    "*.tiktoken",
    "LICENSE*",
    "README*",
]

QUANT_TAG_RE = re.compile(r"(?:^|[-_.])((?:iq|q)\d[a-z0-9_]*|f16|f32|bf16)(?:\.gguf)$", re.IGNORECASE)


def load_hf_token(cli_token: str | None) -> str | None:
    """Resolve an HF token: --token flag > .env (HF_TOKEN) > environment. Warns if none found."""
    if cli_token:
        return cli_token

    if _DOTENV_AVAILABLE:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "[WARN] No HF_TOKEN found (checked --token, .env, and environment variables). "
            "Gated or private repositories will fail. Add HF_TOKEN=hf_... to a .env file "
            "or pass --token to fix this.",
            file=sys.stderr,
        )
    return token


def safe_destination_name(repo_id: str) -> str:
    """Make one cross-platform directory name from `owner/model`."""
    normalized = repo_id.strip().strip("/")
    if not normalized or normalized.startswith(".") or ".." in normalized.split("/"):
        raise ValueError("repo_id must look like 'organization/model'.")
    return re.sub(r"[^A-Za-z0-9._-]+", "--", normalized.replace("/", "--"))


def build_patterns(mode: str, quant: str) -> list[str] | None:
    """Return allow_patterns for snapshot_download based on the selected mode."""
    if mode == "raw-model":
        return None  # download everything

    if mode == "for-gguf":
        return CONFIG_PATTERNS + ["*.safetensors", "*.safetensors.index.json"]

    if mode == "gguf":
        # config/tokenizer files (small) + gguf files matching the requested quant
        return CONFIG_PATTERNS + [f"*{quant}*.gguf", f"*{quant.lower()}*.gguf", f"*{quant.upper()}*.gguf"]

    raise ValueError(f"Unknown mode: {mode}")


def categorize_files(files: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in files:
        lower = name.lower()
        if lower.endswith(".gguf"):
            groups["gguf"].append(name)
        elif lower.endswith(".safetensors") or lower.endswith(".safetensors.index.json"):
            groups["safetensors"].append(name)
        elif lower.endswith((".bin", ".pt", ".pth")):
            groups["pytorch"].append(name)
        elif lower.endswith((".json", ".model", ".tiktoken", ".jinja")) or "tokenizer" in lower:
            groups["config/tokenizer"].append(name)
        else:
            groups["other"].append(name)
    return groups


def detect_quant_tags(files: list[str]) -> list[str]:
    tags = set()
    for name in files:
        if not name.lower().endswith(".gguf"):
            continue
        match = QUANT_TAG_RE.search(name)
        if match:
            tags.add(match.group(1).upper())
    return sorted(tags)


def print_dry_run_report(repo_id: str, revision: str | None, mode: str, quant: str, destination: Path, all_files: list[str]) -> None:
    patterns = build_patterns(mode, quant)
    matched = (
        all_files
        if patterns is None
        else [name for name in all_files if any(fnmatch.fnmatch(name, p) for p in patterns)]
    )

    print(f"Repository:  {repo_id}")
    print(f"Revision:    {revision or 'default branch'}")
    print(f"Mode:        {mode}" + (f" (quant={quant})" if mode == "gguf" else ""))
    print(f"Destination: {destination}")

    print(f"\nAll files in repo ({len(all_files)}), grouped by type:")
    groups = categorize_files(all_files)
    for group_name in ("config/tokenizer", "safetensors", "pytorch", "gguf", "other"):
        group_files = groups.get(group_name)
        if not group_files:
            continue
        print(f"\n  [{group_name}] ({len(group_files)}):")
        for name in sorted(group_files):
            marker = "x" if name in matched else " "
            print(f"    [{marker}] {name}")

    quant_tags = detect_quant_tags(all_files)
    if quant_tags:
        print(f"\nDetected GGUF quant tags available in this repo: {', '.join(quant_tags)}")
        print(f"Use -q/--quant with --gguf to select one, e.g.: -q {quant_tags[0]}")

    print(f"\n{len(matched)}/{len(all_files)} files would be downloaded with the current mode.")
    print("'[x]' = would be downloaded now, '[ ]' = excluded by the current mode/quant filter.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot into ./models/hf/hub/<model>."
    )
    parser.add_argument("repo_id", help="Hugging Face repository ID, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Parent directory for models (default: ./models/hf/hub relative to this script)",
    )
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit hash; defaults to the repo default branch")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--raw-model",
        action="store_const",
        dest="mode",
        const="raw-model",
        help="Download the full repository as-is (all weight formats and files). Default mode.",
    )
    mode_group.add_argument(
        "--for-gguf",
        action="store_const",
        dest="mode",
        const="for-gguf",
        help="Download only config/tokenizer files + .safetensors weights (for later GGUF conversion)",
    )
    mode_group.add_argument(
        "--gguf",
        action="store_const",
        dest="mode",
        const="gguf",
        help="Download only pre-built .gguf files matching --quant, plus config/tokenizer files",
    )
    parser.set_defaults(mode="raw-model")

    parser.add_argument(
        "-q", "--quant",
        default="Q4_K_M",
        help="Quantization tag to match when using --gguf, e.g. Q4_K_M, Q5_K_M, Q8_0 (default: Q4_K_M)",
    )
    parser.add_argument("--token", default=None, help="HF access token; overrides .env / environment variable")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all repository files (grouped by type, with detected GGUF quant tags) without downloading",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download of existing files")
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent downloads (default: 8)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_id = args.repo_id.strip()
    token = load_hf_token(args.token)

    try:
        destination = args.base_dir.expanduser().resolve() / safe_destination_name(repo_id)
    except ValueError as exc:
        print(f"Invalid repo_id: {exc}", file=sys.stderr)
        return 2

    try:
        if args.dry_run:
            api = HfApi(token=token)
            all_files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=args.revision)
            print_dry_run_report(repo_id, args.revision, args.mode, args.quant, destination, all_files)
            return 0

        patterns = build_patterns(args.mode, args.quant)
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Repository:  {repo_id}")
        print(f"Revision:    {args.revision or 'default branch'}")
        print(f"Mode:        {args.mode}" + (f" (quant={args.quant})" if args.mode == "gguf" else ""))
        print(f"Destination: {destination}")

        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=args.revision,
            local_dir=destination,
            token=token,
            allow_patterns=patterns,
            force_download=args.force,
            max_workers=args.max_workers,
        )

        if args.mode == "gguf":
            downloaded_gguf = list(Path(snapshot_path).rglob("*.gguf"))
            if not downloaded_gguf:
                print(
                    f"\n[WARN] No .gguf files matched quant '{args.quant}'. "
                    "Run with --dry-run to inspect available quant tags in this repo.",
                    file=sys.stderr,
                )

        print(f"\nDone. Model snapshot is available at:\n{snapshot_path}")
        return 0

    except GatedRepoError:
        print(
            "This is a gated repository. Accept its license/access terms on Hugging Face, "
            "then provide a token via --token, .env (HF_TOKEN=...), or the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
    except RepositoryNotFoundError:
        print("Repository not found, or your token does not have permission to access it.", file=sys.stderr)
    except LocalEntryNotFoundError:
        print("The model is not available locally and could not be fetched from Hugging Face.", file=sys.stderr)
    except HfHubHTTPError as exc:
        print(f"Hugging Face request failed: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nDownload cancelled.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
