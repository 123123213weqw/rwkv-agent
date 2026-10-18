from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .api import serve
from .config import AppConfig, ModelConfig
from .crawler import FocusedCrawler
from .commoncrawl import CommonCrawlImporter, filter_records, iter_cdxj
from .db import SearchDatabase
from .service import SearchService


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", dest="model_path", help="Local HF RWKV directory or model ID")
    parser.add_argument("--model-label", help="Short model label shown in the frontend")
    parser.add_argument("--device", help="Torch device, for example cuda:1 or cpu")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--no-model", action="store_true", help="Disable RWKV and use extractive fallback")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--native-model", dest="native_model", action="store_true", default=None)
    backend.add_argument("--adapter-model", dest="native_model", action="store_false")


def apply_model_arguments(config: ModelConfig, args: argparse.Namespace) -> None:
    if getattr(args, "model_path", None):
        config.path = args.model_path
        config.enabled = True
    if getattr(args, "model_label", None):
        config.label = args.model_label
    if getattr(args, "device", None):
        config.device = args.device
    if getattr(args, "dtype", None):
        config.dtype = args.dtype
    if getattr(args, "max_input_tokens", None) is not None:
        config.max_input_tokens = max(256, args.max_input_tokens)
    if getattr(args, "max_new_tokens", None) is not None:
        config.max_new_tokens = max(16, args.max_new_tokens)
    if getattr(args, "native_model", None) is not None:
        config.native_model = args.native_model
    if getattr(args, "no_model", False):
        config.enabled = False


def load_answerer(config: ModelConfig) -> Tuple[Optional[Any], Dict[str, Any]]:
    model_name = os.path.basename(os.path.normpath(config.path)) if config.path else None
    status: Dict[str, Any] = {
        "enabled": config.enabled,
        "ready": False,
        "label": config.label,
        "model": model_name,
        "device": config.device,
        "dtype": config.dtype,
        "native_model": config.native_model,
        "error": None,
    }
    if not config.enabled:
        return None, status
    if not config.path:
        status["error"] = "model.enabled is true but model.path is empty"
        return None, status
    try:
        from .rwkv_answerer import HFLocalRWKVAnswerer

        answerer = HFLocalRWKVAnswerer(
            config.path,
            label=config.label,
            device=config.device,
            dtype=config.dtype,
            native_model=config.native_model,
            max_input_tokens=config.max_input_tokens,
            max_new_tokens=config.max_new_tokens,
            repair_once=config.repair_once,
            warmup=config.warmup,
            session_cache_enabled=config.session_cache_enabled,
            session_ttl_seconds=config.session_ttl_seconds,
            session_max_entries=config.session_max_entries,
            session_cpu_offload=config.session_cpu_offload,
        )
        return answerer, answerer.status()
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        return None, status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rwkv-search", description="Local crawler and search for RWKV")
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument("--database", help="Override database path")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="Initialize SQLite schema")

    crawl = commands.add_parser("crawl", help="Crawl seed URLs")
    crawl.add_argument("urls", nargs="+")
    crawl.add_argument("--max-pages", type=int)
    crawl.add_argument("--max-depth", type=int)
    crawl.add_argument("--allow-private", action="store_true", help="Allow loopback/private hosts for testing")

    search = commands.add_parser("search", help="Search local index")
    search.add_argument("query")
    search.add_argument("--freshness", choices=("stable", "latest", "realtime"), default="stable")
    search.add_argument("--limit", type=int, default=10)

    ask = commands.add_parser("ask", help="Route, search, and answer")
    ask.add_argument("query")
    ask.add_argument("--timezone", default="Asia/Shanghai")
    ask.add_argument("--mode", choices=("fast", "deep", "local"), default="fast")
    add_model_arguments(ask)

    server = commands.add_parser("serve", help="Run local frontend and SSE API")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    add_model_arguments(server)

    cc = commands.add_parser("cc-import", help="Selectively import pages from a local Common Crawl CDXJ export")
    cc.add_argument("cdxj", type=Path)
    cc.add_argument("--domain", action="append", default=[])
    cc.add_argument("--language", action="append", default=[])
    cc.add_argument("--max-pages", type=int, default=100)

    commands.add_parser("stats", help="Show index/frontier statistics")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.load(args.config)
    if args.database:
        config.database = args.database
    database = SearchDatabase(config.database)

    if args.command == "init":
        print(json.dumps(database.stats(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "crawl":
        if args.max_depth is not None:
            config.crawler.max_depth = max(0, args.max_depth)
        if args.allow_private:
            config.crawler.allow_private_networks = True
        crawler = FocusedCrawler(database, config.crawler)
        crawler.seed(args.urls)
        result = asyncio.run(crawler.run(args.max_pages))
        print(json.dumps({"crawl": result, "database": database.stats()}, ensure_ascii=False, indent=2))
        return 0 if not result["failed"] else 2
    answerer = None
    model_status: Dict[str, Any] = {
        "enabled": False,
        "ready": False,
        "label": config.model.label,
        "model": None,
        "error": None,
    }
    if args.command in {"ask", "serve"}:
        apply_model_arguments(config.model, args)
        answerer, model_status = load_answerer(config.model)
        if config.model.enabled and answerer is None:
            print(
                f"RWKV model load failed; using extractive fallback: {model_status['error']}",
                file=sys.stderr,
            )
    service = SearchService(
        database,
        search_config=config.search,
        answerer=answerer,
        realtime_config=config.realtime_search,
        shadow_config=config.shadow_search,
    )
    if args.command == "search":
        print(json.dumps(service.search(args.query, freshness=args.freshness, limit=args.limit), ensure_ascii=False, indent=2))
        return 0
    if args.command == "ask":
        for event in service.ask_events(args.query, user_timezone=args.timezone, mode=args.mode):
            print(json.dumps(event, ensure_ascii=False))
        return 0
    if args.command == "serve":
        serve(
            database,
            host=args.host,
            port=args.port,
            search_config=config.search,
            crawl_config=config.crawler,
            answerer=answerer,
            model_status=model_status,
            realtime_config=config.realtime_search,
            shadow_config=config.shadow_search,
        )
        return 0
    if args.command == "cc-import":
        records = filter_records(
            iter_cdxj(args.cdxj), domains=args.domain, languages=args.language
        )
        result = CommonCrawlImporter(database).import_records(records, max_pages=args.max_pages)
        print(json.dumps({"common_crawl": result, "database": database.stats()}, ensure_ascii=False, indent=2))
        return 0 if not result["failed"] else 2
    if args.command == "stats":
        print(json.dumps(database.stats(), ensure_ascii=False, indent=2))
        return 0
    return 2
