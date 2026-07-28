#!/usr/bin/env python3
"""Freeze and enforce the one-shot Fresh-Web-200 evaluation workflow."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = "rwkv-agent-fresh-web-workflow.v1"
CATEGORIES = (
    "dynamic_fact",
    "entity_organization",
    "cross_source_comparison",
    "number_date",
    "ambiguity_resolution",
)
LANGUAGES = ("zh", "en")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def checkpoint_artifacts(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    files = [resolved] if resolved.is_file() else sorted(
        value for value in resolved.rglob("*") if value.is_file()
    )
    if not files:
        raise ValueError(f"checkpoint has no files: {resolved}")
    base = resolved.parent if resolved.is_file() else resolved
    return [
        {
            "path": str(value.relative_to(base)),
            "bytes": value.stat().st_size,
            "sha256": sha256(value),
        }
        for value in files
    ]


def freeze_checkpoint(checkpoint: Path, output: Path) -> dict[str, Any]:
    artifacts = checkpoint_artifacts(checkpoint)
    record = {
        "schema_version": SCHEMA_VERSION,
        "event": "checkpoint_frozen_before_fresh_collection",
        "created_at": utc_now(),
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "artifacts": artifacts,
    }
    record["checkpoint_manifest_sha256"] = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _json_dump(output, record, exclusive=True)
    return record


def _require_text(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{label}.{key} must be non-empty text")
    return item.strip()


def validate_collection(
    rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_root: Path,
    training_domains: set[str],
    require_complete: bool = True,
) -> dict[str, Any]:
    ids: set[str] = set()
    languages: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    outside_training = 0
    source_domains: set[str] = set()
    for index, row in enumerate(rows):
        label = f"case[{index}]"
        case_id = _require_text(row, "id", label)
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        language = _require_text(row, "language", label)
        category = _require_text(row, "category", label)
        if language not in LANGUAGES:
            raise ValueError(f"{label}.language must be zh or en")
        if category not in CATEGORIES:
            raise ValueError(f"{label}.category is unsupported")
        languages[language] += 1
        categories[category] += 1
        _require_text(row, "prompt", label)
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers or not all(
            isinstance(value, str) and value.strip() for value in answers
        ):
            raise ValueError(f"{label}.answers must be non-empty strings")
        sources = row.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{label}.sources must be non-empty")
        authoritative = 0
        case_domains: set[str] = set()
        for source_index, source in enumerate(sources):
            source_label = f"{label}.sources[{source_index}]"
            if not isinstance(source, Mapping):
                raise ValueError(f"{source_label} must be an object")
            uri = _require_text(source, "uri", source_label)
            host = (urlsplit(uri).hostname or "").casefold().strip(".")
            if not host:
                raise ValueError(f"{source_label}.uri must have a host")
            case_domains.add(host)
            source_domains.add(host)
            authoritative += bool(source.get("authoritative"))
            relative = Path(_require_text(source, "snapshot_path", source_label))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{source_label}.snapshot_path must be relative")
            snapshot = snapshot_root / relative
            if not snapshot.is_file():
                raise ValueError(f"missing snapshot: {snapshot}")
            if sha256(snapshot) != _require_text(source, "snapshot_sha256", source_label):
                raise ValueError(f"snapshot hash mismatch: {snapshot}")
            spans = source.get("evidence_spans")
            if not isinstance(spans, list) or not spans or not all(
                isinstance(value, str) and value.strip() for value in spans
            ):
                raise ValueError(f"{source_label}.evidence_spans must be non-empty")
        if len(sources) < 2 and authoritative < 1:
            raise ValueError(f"{label} needs two sources or one authoritative source")
        if case_domains and all(
            not any(
                host == trained or host.endswith("." + trained)
                for trained in training_domains
            )
            for host in case_domains
        ):
            outside_training += 1

    if require_complete:
        if len(rows) != 200:
            raise ValueError(f"Fresh-Web must contain exactly 200 cases, got {len(rows)}")
        if languages != Counter({"zh": 100, "en": 100}):
            raise ValueError(f"language quotas failed: {dict(languages)}")
        if categories != Counter({category: 40 for category in CATEGORIES}):
            raise ValueError(f"category quotas failed: {dict(categories)}")
        if outside_training < 100:
            raise ValueError(
                f"only {outside_training} cases are outside training domains; need 100"
            )
    return {
        "cases": len(rows),
        "languages": dict(languages),
        "categories": dict(categories),
        "outside_training_domain_cases": outside_training,
        "unique_source_domains": len(source_domains),
    }


def freeze_collection(
    draft: Path,
    snapshot_root: Path,
    checkpoint_record: Path,
    output_dir: Path,
    *,
    training_domains: set[str],
) -> dict[str, Any]:
    checkpoint = json.loads(checkpoint_record.read_text(encoding="utf-8"))
    if checkpoint.get("event") != "checkpoint_frozen_before_fresh_collection":
        raise ValueError("invalid checkpoint freeze record")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _jsonl_load(draft)
    summary = validate_collection(
        rows,
        snapshot_root=snapshot_root,
        training_domains=training_domains,
    )
    public_rows = []
    private_rows = []
    for row in rows:
        public_rows.append(
            {
                "schema_version": "rwkv-agent-benchmark-case.v1",
                "id": str(row["id"]),
                "dataset": "fresh_web",
                "split": "fresh_web_once",
                "track": "web_research",
                "language": str(row["language"]),
                "prompt": str(row["prompt"]),
                "gold": {
                    "answerable": True,
                    "requires_citations": True,
                    "should_call_tools": True,
                },
                "limits": {"max_rounds": 2, "max_requests": 8, "max_latency_ms": 20000},
                "metadata": {"category": str(row["category"])},
            }
        )
        private_rows.append(dict(row))
    public_path = output_dir / "cases.public.jsonl"
    private_path = output_dir / "gold.private.jsonl"
    _jsonl_dump(public_path, public_rows)
    _jsonl_dump(private_path, private_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "event": "fresh_collection_frozen",
        "created_at": utc_now(),
        "checkpoint_manifest_sha256": checkpoint["checkpoint_manifest_sha256"],
        "summary": summary,
        "artifacts": {
            public_path.name: {"bytes": public_path.stat().st_size, "sha256": sha256(public_path)},
            private_path.name: {"bytes": private_path.stat().st_size, "sha256": sha256(private_path)},
        },
        "blind_run_state": "unused",
    }
    _json_dump(output_dir / "manifest.json", manifest, exclusive=True)
    return manifest


def claim_blind_run(manifest_path: Path, run_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("blind_run_state") != "unused":
        raise ValueError("manifest is not unused")
    claim = {
        "schema_version": SCHEMA_VERSION,
        "event": "fresh_blind_run_claimed",
        "created_at": utc_now(),
        "run_id": str(run_id),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint_manifest_sha256": manifest["checkpoint_manifest_sha256"],
    }
    _json_dump(manifest_path.parent / "blind-run-claim.json", claim, exclusive=True)
    return claim


def materialize_blind_scoring_cases(
    manifest_path: Path,
    claim_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Reveal private Gold to an exclusive scoring directory after the claim.

    This transition deliberately cannot run before ``claim_blind_run``.  The
    output uses the normal WebWalkerQA scoring schema so the frozen runner can
    evaluate Fresh-Web without a second, subtly different metric path.
    """

    manifest_path = manifest_path.expanduser().resolve()
    claim_path = claim_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if manifest.get("event") != "fresh_collection_frozen":
        raise ValueError("invalid fresh collection manifest")
    if claim.get("event") != "fresh_blind_run_claimed":
        raise ValueError("invalid blind-run claim")
    if claim.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("blind-run claim does not bind this collection")
    if claim.get("checkpoint_manifest_sha256") != manifest.get(
        "checkpoint_manifest_sha256"
    ):
        raise ValueError("checkpoint binding differs between claim and collection")
    artifacts = dict(manifest.get("artifacts") or {})
    public_path = manifest_path.parent / "cases.public.jsonl"
    private_path = manifest_path.parent / "gold.private.jsonl"
    for path in (public_path, private_path):
        record = artifacts.get(path.name)
        if not isinstance(record, Mapping) or record.get("sha256") != sha256(path):
            raise ValueError(f"collection artifact hash mismatch: {path}")
    public = _jsonl_load(public_path)
    private = _jsonl_load(private_path)
    private_by_id = {str(row.get("id") or ""): row for row in private}
    if len(private_by_id) != len(private) or {str(row.get("id") or "") for row in public} != set(private_by_id):
        raise ValueError("public/private Fresh-Web case IDs differ")
    scoring_rows = []
    for public_row in public:
        case_id = str(public_row["id"])
        gold = private_by_id[case_id]
        sources = list(gold.get("sources") or [])
        scoring_rows.append(
            {
                **public_row,
                "dataset": "webwalkerqa",
                "split": "fresh_web_once",
                "gold": {
                    "answers": [str(value) for value in gold.get("answers") or []],
                    "answerable": True,
                    "requires_citations": True,
                    "should_call_tools": True,
                    "source_uris": [str(source.get("uri") or "") for source in sources],
                },
                "metadata": {
                    **dict(public_row.get("metadata") or {}),
                    "fresh_web": True,
                    "fresh_run_id": str(claim.get("run_id") or ""),
                },
            }
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    cases_path = output_dir / "webwalkerqa.jsonl"
    _jsonl_dump(cases_path, scoring_rows)
    cases_path.chmod(0o600)
    record = {
        "schema_version": SCHEMA_VERSION,
        "event": "fresh_blind_scoring_cases_materialized",
        "created_at": utc_now(),
        "run_id": str(claim.get("run_id") or ""),
        "checkpoint_manifest_sha256": manifest["checkpoint_manifest_sha256"],
        "collection_manifest_sha256": sha256(manifest_path),
        "blind_claim_sha256": sha256(claim_path),
        "artifacts": {
            cases_path.name: {
                "bytes": cases_path.stat().st_size,
                "cases": len(scoring_rows),
                "sha256": sha256(cases_path),
            }
        },
    }
    _json_dump(output_dir / "manifest.json", record, exclusive=True)
    return record


def _domains(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip().casefold().strip(".")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    checkpoint = subparsers.add_parser("freeze-checkpoint")
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    collection = subparsers.add_parser("freeze-collection")
    collection.add_argument("--draft", type=Path, required=True)
    collection.add_argument("--snapshot-root", type=Path, required=True)
    collection.add_argument("--checkpoint-record", type=Path, required=True)
    collection.add_argument("--training-domains", type=Path)
    collection.add_argument("--output-dir", type=Path, required=True)
    claim = subparsers.add_parser("claim-blind-run")
    claim.add_argument("--manifest", type=Path, required=True)
    claim.add_argument("--run-id", required=True)
    materialize = subparsers.add_parser("materialize-blind-cases")
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--claim", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-checkpoint":
        value = freeze_checkpoint(args.checkpoint, args.output)
    elif args.command == "freeze-collection":
        value = freeze_collection(
            args.draft,
            args.snapshot_root,
            args.checkpoint_record,
            args.output_dir,
            training_domains=_domains(args.training_domains),
        )
    elif args.command == "claim-blind-run":
        value = claim_blind_run(args.manifest, args.run_id)
    else:
        value = materialize_blind_scoring_cases(
            args.manifest,
            args.claim,
            args.output_dir,
        )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
