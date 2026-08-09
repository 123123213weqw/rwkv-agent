#!/usr/bin/env python3
"""Generate many local websites from RWKV-produced Website DSL records.

The model owns every semantic and visual decision in the DSL.  A deterministic
renderer performs only escaping, layout, validation, and atomic artifact writes.
This mirrors an Agent pipeline of natural language -> DSL -> validation -> action.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable, Sequence


SCHEMA = "rwkv-agent-website-swarm.v1"
PROTOCOL_MARKERS = ("<tool_call>", "<tool_result>", "<think>", "System:", "User:", "Assistant:")
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class WebsiteSpec:
    index: int
    site_id: str
    category: str
    audience: str
    mood: str
    brand: str
    brief: str


@dataclass
class Generation:
    spec: WebsiteSpec
    prompt: str
    raw: str
    dsl: dict[str, Any] | None
    errors: list[str]
    attempts: int
    output_tokens: int


CATEGORIES = (
    "independent coffee studio",
    "open-source robotics lab",
    "urban gardening collective",
    "indie game launch",
    "privacy-first note app",
    "mountain travel journal",
    "community science museum",
    "sustainable fashion atelier",
    "local music festival",
    "creative coding workshop",
    "astronomy club",
    "artisan bakery",
    "language learning studio",
    "wildlife conservation project",
    "minimal fitness coach",
    "neighborhood bookshop",
    "renewable energy startup",
    "digital art exhibition",
    "slow living newsletter",
    "maker-space membership",
)
AUDIENCES = (
    "curious university students",
    "busy young professionals",
    "families and local residents",
    "independent creators",
    "technical early adopters",
)
MOODS = (
    "bold editorial",
    "warm and organic",
    "futuristic neon",
    "calm minimal",
    "playful geometric",
    "premium monochrome",
    "retro optimistic",
    "clean scientific",
    "cinematic dark",
    "bright accessible",
)
BRAND_LEFT = (
    "Luma",
    "Orbit",
    "Moss",
    "Nova",
    "Cedar",
    "Pixel",
    "Amber",
    "Echo",
    "Atlas",
    "River",
)
BRAND_RIGHT = ("Works", "Studio", "Lab", "House", "Collective", "Foundry", "Field", "Club", "Grid", "Point")


def build_specs(count: int, seed: int) -> list[WebsiteSpec]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    brands = [(left, right) for left in BRAND_LEFT for right in BRAND_RIGHT]
    categories = list(CATEGORIES)
    audiences = list(AUDIENCES)
    moods = list(MOODS)
    rng.shuffle(brands)
    rng.shuffle(categories)
    rng.shuffle(audiences)
    rng.shuffle(moods)
    specs = []
    for offset in range(1, count + 1):
        left, right = brands[(offset - 1) % len(brands)]
        generation = (offset - 1) // len(brands) + 1
        category = categories[(offset - 1) % len(categories)]
        audience = audiences[((offset - 1) * 3) % len(audiences)]
        mood = moods[((offset - 1) * 7) % len(moods)]
        site_id = f"site-{offset:03d}"
        brand = f"{left} {right}" + (f" {generation}" if generation > 1 else "")
        brief = f"Create a landing page for a {category}, aimed at {audience}, with a {mood} visual direction."
        specs.append(WebsiteSpec(offset, site_id, category, audience, mood, brand, brief))
    return specs


def website_prompt(spec: WebsiteSpec, *, repair_raw: str = "", repair_errors: Sequence[str] = ()) -> str:
    contract = (
        "Return one compact JSON object. The opening { is already supplied, so begin with the first quoted key and end with }. "
        "Use exactly these keys: title, tagline, theme, hero, features, stats, footer. "
        "theme must contain background,surface,primary,text,accent as six-digit #RRGGBB colors. "
        "hero must contain eyebrow,headline,summary,cta. features must contain exactly 3 objects with title,description. "
        "stats must contain exactly 3 objects with value,label. All strings must be concise plain text. "
        "Do not emit markdown, HTML, comments, roles, reasoning, or additional keys."
    )
    if repair_raw:
        return (
            "System: You repair a Website DSL object while preserving its creative decisions. "
            + contract
            + "\n\nUser: Requested site:\n"
            + spec.brief
            + f"\nBrand: {spec.brand}\nValidation errors: {json.dumps(list(repair_errors), ensure_ascii=False)}"
            + "\nInvalid previous output:\n"
            + repair_raw[:5000]
            + "\n\nAssistant: {"
        )
    return (
        "System: You are one worker in a local RWKV website swarm. Design the requested website as Website DSL. "
        + contract
        + "\n\nUser: "
        + spec.brief
        + f"\nBrand: {spec.brand}\nEach page must have specific, original copy for this brand and audience."
        + "\n\nAssistant: {"
    )


def extract_json(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    value = raw.strip()
    if any(marker in value for marker in PROTOCOL_MARKERS):
        return None, ["protocol_leak"]
    if value.startswith("```"):
        return None, ["markdown_fence"]
    try:
        parsed, end = json.JSONDecoder().raw_decode(value)
    except json.JSONDecodeError as exc:
        return None, [f"json:{exc.msg}@{exc.pos}"]
    if value[end:].strip():
        return None, ["trailing_content"]
    if not isinstance(parsed, dict):
        return None, ["root_not_object"]
    return parsed, []


def _plain_string(value: Any, path: str, errors: list[str], *, minimum: int = 1, maximum: int = 180) -> str:
    if not isinstance(value, str):
        errors.append(f"{path}:not_string")
        return ""
    text = " ".join(value.split())
    if not minimum <= len(text) <= maximum:
        errors.append(f"{path}:length")
    if any(marker in text for marker in PROTOCOL_MARKERS) or "<" in text or ">" in text:
        errors.append(f"{path}:unsafe_text")
    return text


def validate_dsl(value: dict[str, Any], spec: WebsiteSpec) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    expected = {"title", "tagline", "theme", "hero", "features", "stats", "footer"}
    if set(value) != expected:
        errors.append("root_keys")
    title = _plain_string(value.get("title"), "title", errors, maximum=80)
    tagline = _plain_string(value.get("tagline"), "tagline", errors, maximum=120)
    footer = _plain_string(value.get("footer"), "footer", errors, maximum=100)
    if spec.brand.lower() not in title.lower():
        errors.append("title:missing_brand")

    theme_raw = value.get("theme")
    theme: dict[str, str] = {}
    theme_keys = {"background", "surface", "primary", "text", "accent"}
    if not isinstance(theme_raw, dict) or set(theme_raw) != theme_keys:
        errors.append("theme:keys")
    else:
        for key in sorted(theme_keys):
            color = str(theme_raw.get(key, ""))
            if not HEX_COLOR.fullmatch(color):
                errors.append(f"theme.{key}:color")
            theme[key] = color.upper()

    hero_raw = value.get("hero")
    hero: dict[str, str] = {}
    hero_keys = {"eyebrow", "headline", "summary", "cta"}
    if not isinstance(hero_raw, dict) or set(hero_raw) != hero_keys:
        errors.append("hero:keys")
    else:
        for key, maximum in (("eyebrow", 60), ("headline", 100), ("summary", 220), ("cta", 40)):
            hero[key] = _plain_string(hero_raw.get(key), f"hero.{key}", errors, maximum=maximum)

    features_raw = value.get("features")
    features: list[dict[str, str]] = []
    if not isinstance(features_raw, list) or len(features_raw) != 3:
        errors.append("features:count")
    else:
        for index, item in enumerate(features_raw):
            if not isinstance(item, dict) or set(item) != {"title", "description"}:
                errors.append(f"features.{index}:keys")
                continue
            features.append(
                {
                    "title": _plain_string(item.get("title"), f"features.{index}.title", errors, maximum=60),
                    "description": _plain_string(
                        item.get("description"), f"features.{index}.description", errors, maximum=160
                    ),
                }
            )

    stats_raw = value.get("stats")
    stats: list[dict[str, str]] = []
    if not isinstance(stats_raw, list) or len(stats_raw) != 3:
        errors.append("stats:count")
    else:
        for index, item in enumerate(stats_raw):
            if not isinstance(item, dict) or set(item) != {"value", "label"}:
                errors.append(f"stats.{index}:keys")
                continue
            stats.append(
                {
                    "value": _plain_string(item.get("value"), f"stats.{index}.value", errors, maximum=24),
                    "label": _plain_string(item.get("label"), f"stats.{index}.label", errors, maximum=60),
                }
            )
    if errors:
        return None, errors
    return {
        "title": title,
        "tagline": tagline,
        "theme": theme,
        "hero": hero,
        "features": features,
        "stats": stats,
        "footer": footer,
    }, []


def render_html(spec: WebsiteSpec, dsl: dict[str, Any]) -> str:
    theme = dsl["theme"]
    hero = dsl["hero"]
    feature_html = "".join(
        f'<article class="feature"><span>{index:02d}</span><h3>{escape(item["title"])}</h3>'
        f'<p>{escape(item["description"])}</p></article>'
        for index, item in enumerate(dsl["features"], 1)
    )
    stat_html = "".join(
        f'<div class="stat"><strong>{escape(item["value"])}</strong><small>{escape(item["label"])}</small></div>'
        for item in dsl["stats"]
    )
    return f"""<!doctype html>
<html lang="en" data-generated-by="rwkv-local-state-agent">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="{escape(dsl['tagline'], quote=True)}"><title>{escape(dsl['title'])}</title>
  <style>
    :root{{--bg:{theme['background']};--surface:{theme['surface']};--primary:{theme['primary']};--text:{theme['text']};--accent:{theme['accent']}}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 Inter,ui-sans-serif,system-ui,sans-serif}}
    body:before{{content:"";position:fixed;inset:0;background:radial-gradient(circle at 75% 10%,var(--accent) 0,transparent 32%);opacity:.16;pointer-events:none}}
    .shell{{width:min(1120px,calc(100% - 40px));margin:auto}} nav{{display:flex;justify-content:space-between;align-items:center;padding:28px 0}}
    .brand{{font-weight:850;letter-spacing:-.04em;font-size:1.3rem}} .chip{{border:1px solid color-mix(in srgb,var(--text) 22%,transparent);border-radius:999px;padding:7px 13px}}
    .hero{{min-height:62vh;display:grid;align-content:center;max-width:900px;padding:70px 0}} .eyebrow{{color:var(--primary);font-weight:800;text-transform:uppercase;letter-spacing:.16em}}
    h1{{font-size:clamp(3rem,8vw,7rem);line-height:.88;letter-spacing:-.07em;margin:.25em 0}} .lead{{font-size:clamp(1.1rem,2vw,1.45rem);max-width:670px;opacity:.78}}
    .cta{{display:inline-block;background:var(--primary);color:var(--bg);text-decoration:none;font-weight:850;padding:14px 22px;border-radius:14px;margin-top:25px}}
    .features{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding:34px 0}} .feature{{background:var(--surface);padding:26px;border-radius:22px;border:1px solid color-mix(in srgb,var(--text) 10%,transparent)}}
    .feature span{{color:var(--accent);font-weight:850}} .feature h3{{font-size:1.35rem;margin:.7em 0 .3em}} .feature p{{opacity:.72;margin:0}}
    .stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:color-mix(in srgb,var(--text) 12%,transparent);margin:50px 0;border-radius:22px;overflow:hidden}}
    .stat{{background:var(--surface);padding:30px;display:grid}} .stat strong{{font-size:2rem;color:var(--primary)}} .stat small{{opacity:.66}} footer{{display:flex;justify-content:space-between;padding:40px 0 60px;opacity:.66}}
    @media(max-width:760px){{.features,.stats{{grid-template-columns:1fr}}h1{{font-size:3.5rem}}}}
  </style>
</head>
<body><div class="shell">
  <nav><div class="brand">{escape(spec.brand)}</div><div class="chip">{escape(spec.category)}</div></nav>
  <main><section class="hero"><div class="eyebrow">{escape(hero['eyebrow'])}</div><h1>{escape(hero['headline'])}</h1>
    <p class="lead">{escape(hero['summary'])}</p><a class="cta" href="#features">{escape(hero['cta'])}</a></section>
    <section id="features" class="features">{feature_html}</section><section class="stats">{stat_html}</section></main>
  <footer><span>{escape(dsl['footer'])}</span><span>{escape(spec.site_id)}</span></footer>
</div></body></html>
"""


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.titles: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.titles.append(data)


def validate_html(text: str) -> list[str]:
    errors = []
    if not text.lower().startswith("<!doctype html>"):
        errors.append("doctype")
    parser = _StructureParser()
    try:
        parser.feed(text)
    except Exception as exc:  # pragma: no cover - defensive stdlib boundary
        return [f"parse:{type(exc).__name__}"]
    for tag in ("html", "head", "title", "style", "body", "main"):
        if tag not in parser.tags:
            errors.append(f"tag:{tag}")
    if not "".join(parser.titles).strip():
        errors.append("empty_title")
    if any(marker in text for marker in ("<tool_call>", "<tool_result>", "<think>")):
        errors.append("protocol_leak")
    return errors


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def render_gallery(generations: Sequence[Generation]) -> str:
    cards = "".join(
        f'<article><iframe loading="lazy" src="{escape(item.spec.site_id)}/index.html" title="{escape(item.dsl["title"])}"></iframe>'
        f'<div><strong>{escape(item.dsl["title"])}</strong><span>{escape(item.spec.mood)}</span></div></article>'
        for item in generations
        if item.dsl is not None
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RWKV Website Swarm</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#090b11;color:#f5f7ff;font-family:Inter,system-ui,sans-serif}}
header{{padding:42px max(24px,4vw)}}h1{{font-size:clamp(2.8rem,7vw,6rem);letter-spacing:-.06em;margin:0}}p{{color:#9ba4bc;font-size:1.2rem}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px;padding:0 max(24px,4vw) 60px}}article{{background:#141824;border:1px solid #242b3d;border-radius:18px;overflow:hidden}}
iframe{{width:100%;height:420px;border:0;background:white}}article div{{display:flex;justify-content:space-between;gap:14px;padding:16px}}span{{color:#8791a8}}</style></head>
<body><header><h1>100 states. 100 websites.</h1><p>Generated locally by RWKV on AMD Radeon, validated and rendered without a cloud model.</p></header><main>{cards}</main></body></html>"""


class HFWebsiteGenerator:
    def __init__(self, model_path: Path, *, device: str, dtype: str) -> None:
        os.environ.setdefault("RWKV7_NATIVE_MODEL_BACKEND", "eager")
        os.environ.setdefault("RWKV7_NATIVE_MODEL_JIT", "0")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_value = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype_value,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.model.config.use_cache = True
        self.device = device
        self.model_path = model_path

    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int) -> tuple[list[str], list[int], dict[str, Any]]:
        torch = self.torch
        encoded = [list(map(int, self.tokenizer.encode(prompt))) for prompt in prompts]
        width = max(map(len, encoded))
        input_ids = torch.zeros((len(encoded), width), device=self.device, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, token_ids in enumerate(encoded):
            input_ids[row, -len(token_ids) :] = torch.tensor(token_ids, device=self.device)
            attention_mask[row, -len(token_ids) :] = 1
        generated: list[list[int]] = [[] for _ in prompts]
        finished = torch.zeros(len(prompts), device=self.device, dtype=torch.bool)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            logits = output.logits[:, -1, :]
            for _ in range(max_new_tokens):
                tokens = torch.argmax(logits, dim=-1)
                for row, token in enumerate(tokens.tolist()):
                    if not bool(finished[row]):
                        if token == 0:
                            finished[row] = True
                        else:
                            generated[row].append(int(token))
                if bool(finished.all()):
                    break
                feed = torch.where(finished, torch.zeros_like(tokens), tokens).unsqueeze(1)
                output = self.model(input_ids=feed, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                logits = output.logits[:, -1, :]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        texts = ["{" + self.tokenizer.decode(tokens) for tokens in generated]
        counts = [len(tokens) for tokens in generated]
        return texts, counts, {
            "rows": len(prompts),
            "prompt_tokens": sum(map(len, encoded)),
            "output_tokens": sum(counts),
            "elapsed_seconds": round(elapsed, 6),
            "output_tokens_per_second": round(sum(counts) / elapsed if elapsed else 0.0, 3),
        }


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args.count, args.seed)
    generator = HFWebsiteGenerator(args.model.resolve(), device=args.device, dtype=args.dtype)
    torch = generator.torch
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generations: list[Generation] = []
    batch_metrics: list[dict[str, Any]] = []

    for batch_number, batch_specs in enumerate(_chunks(specs, args.batch_size), 1):
        prompts = [website_prompt(spec) for spec in batch_specs]
        texts, counts, metrics = generator.generate_batch(prompts, args.max_new_tokens)
        batch_metrics.append({"batch": batch_number, **metrics})
        for spec, prompt, raw, tokens in zip(batch_specs, prompts, texts, counts):
            parsed, errors = extract_json(raw)
            dsl = None
            if parsed is not None:
                dsl, errors = validate_dsl(parsed, spec)
            generations.append(Generation(spec, prompt, raw, dsl, errors, 1, tokens))
        print(f"batch={batch_number} rows={len(batch_specs)} tok/s={metrics['output_tokens_per_second']}", flush=True)

    for attempt in range(1, args.repair_attempts + 1):
        invalid = [item for item in generations if item.dsl is None]
        if not invalid:
            break
        for repair_batch in _chunks(invalid, args.batch_size):
            prompts = [website_prompt(item.spec, repair_raw=item.raw, repair_errors=item.errors) for item in repair_batch]
            texts, counts, metrics = generator.generate_batch(prompts, args.max_new_tokens)
            batch_metrics.append({"batch": len(batch_metrics) + 1, "repair_attempt": attempt, **metrics})
            for item, prompt, raw, tokens in zip(repair_batch, prompts, texts, counts):
                parsed, errors = extract_json(raw)
                dsl = None
                if parsed is not None:
                    dsl, errors = validate_dsl(parsed, item.spec)
                item.prompt = prompt
                item.raw = raw
                item.dsl = dsl
                item.errors = errors
                item.attempts += 1
                item.output_tokens += tokens

    html_errors: dict[str, list[str]] = {}
    for item in generations:
        if item.dsl is None:
            continue
        html = render_html(item.spec, item.dsl)
        errors = validate_html(html)
        if errors:
            html_errors[item.spec.site_id] = errors
            continue
        _atomic_write(output / item.spec.site_id / "index.html", html)
        _atomic_write(output / item.spec.site_id / "design.json", json.dumps(item.dsl, ensure_ascii=False, indent=2) + "\n")

    valid = [item for item in generations if item.dsl is not None and item.spec.site_id not in html_errors]
    _atomic_write(output / "index.html", render_gallery(valid))
    with (output / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for item in generations:
            handle.write(
                json.dumps(
                    {
                        "spec": asdict(item.spec),
                        "prompt": item.prompt,
                        "raw": item.raw,
                        "dsl": item.dsl,
                        "errors": item.errors,
                        "attempts": item.attempts,
                        "output_tokens": item.output_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    elapsed = time.perf_counter() - started
    titles = [str(item.dsl["title"]).casefold() for item in valid]
    metrics = {
        "schema": SCHEMA,
        "status": "pass" if len(valid) == args.count and len(set(titles)) == args.count else "fail",
        "count_requested": args.count,
        "count_valid": len(valid),
        "dsl_valid_rate": round(sum(item.dsl is not None for item in generations) / args.count, 6),
        "html_valid_rate": round(len(valid) / args.count, 6),
        "unique_title_rate": round(len(set(titles)) / args.count, 6),
        "protocol_leaks": sum("protocol_leak" in item.errors for item in generations),
        "timeouts": 0,
        "logical_workers": args.count,
        "max_physical_batch_size": args.batch_size,
        "wall_seconds": round(elapsed, 6),
        "output_tokens": sum(item.output_tokens for item in generations),
        "output_tokens_per_second": round(
            sum(item.output_tokens for item in generations) / elapsed if elapsed else 0.0, 3
        ),
        "repair_count": sum(item.attempts > 1 for item in generations),
        "batch_metrics": batch_metrics,
        "model": str(args.model.resolve()),
        "model_sha256_index": _optional_sha(args.model / "model.safetensors.index.json"),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "html_errors": html_errors,
        "failed_sites": {item.spec.site_id: item.errors for item in generations if item.dsl is None},
    }
    _atomic_write(output / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    metrics["metrics_sha256"] = sha256((output / "metrics.json").read_bytes()).hexdigest()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _optional_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > args.count:
        parser.error("--batch-size must be between 1 and --count")
    if args.max_new_tokens < 32:
        parser.error("--max-new-tokens must be at least 32")
    return args


def main() -> None:
    try:
        metrics = run(parse_args())
    except Exception as exc:
        print(f"website swarm failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    if metrics["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
