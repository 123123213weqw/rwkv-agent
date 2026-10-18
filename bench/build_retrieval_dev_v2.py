#!/usr/bin/env python3
"""Build the manually curated bilingual retrieval development set."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


FORBIDDEN = [
    "search_homepage",
    "dictionary",
    "error_page",
    "login_or_captcha",
    "empty_content",
]


def topic(
    zh: str,
    en: str,
    category: str,
    freshness: str,
    source_policy: str,
    domains: list[str],
    patterns: list[str],
    notes: str,
    query_style: str,
    task_family: str,
    gold_ttl_days: int,
) -> dict[str, Any]:
    return locals()


TOPICS = [
    topic("python 3.14 free threading到底怎么开，查官方文档", "How do I enable free threading in Python 3.14? Use the official guide.", "official_docs", "latest", "official_required", ["docs.python.org"], ["/3/howto/free-threading-python"], "Python free-threading official how-to", "conversational", "technical_howto", 90),
    topic("PyTorch装CUDA版本该选哪个命令", "Which official install command should I use for the CUDA build of PyTorch?", "official_docs", "latest", "official_required", ["pytorch.org"], ["/get-started/locally"], "PyTorch installation selector", "conversational", "technical_howto", 30),
    topic("node自带fetch现在稳定了吗 看API文档", "Is the built-in Node.js fetch API stable now? Check the API documentation.", "official_docs", "latest", "official_required", ["nodejs.org"], ["/api/globals"], "Node.js fetch API status", "conversational", "technical_reference", 90),
    topic("PostgreSQL jsonb索引怎么写，官网例子", "Find the official PostgreSQL JSONB indexing examples.", "official_docs", "stable", "official_required", ["postgresql.org"], ["/docs/current/datatype-json"], "PostgreSQL JSONB indexing documentation", "conversational", "technical_howto", 365),
    topic("nginx websocket 反代配置官方写法", "Official Nginx WebSocket reverse proxy configuration", "official_docs", "stable", "official_required", ["nginx.org"], ["/en/docs/http/websocket"], "Nginx WebSocket proxy documentation", "terse", "technical_howto", 365),
    topic("docker compose怎么给容器用GPU", "How can Docker Compose give a container GPU access?", "official_docs", "latest", "official_required", ["docs.docker.com"], ["/compose/how-tos/gpu-support"], "Docker Compose GPU support", "conversational", "technical_howto", 90),
    topic("k8s ingress官方现在推荐怎么配", "What does the current Kubernetes documentation recommend for Ingress?", "official_docs", "latest", "official_required", ["kubernetes.io"], ["/docs/concepts/services-networking/ingress"], "Kubernetes Ingress documentation", "conversational", "technical_howto", 90),
    topic("Rust异步编程官方book在哪", "Where is the official Rust Async Book?", "official_docs", "stable", "official_required", ["rust-lang.github.io"], ["/async-book"], "Official Rust Async Book", "terse", "technical_reference", 365),
    topic("Go memory model 原文", "Go memory model original specification", "standards_specification", "stable", "primary_required", ["go.dev"], ["/ref/mem"], "Go memory model specification", "terse", "language_specification", 730),
    topic("TypeScript最近一版有什么breaking change", "What breaking changes are in the newest TypeScript release?", "software_release", "latest", "official_required", ["devblogs.microsoft.com"], ["/typescript/announcing-typescript"], "TypeScript release announcement", "conversational", "software_release", 30),
    topic("安卓这个月安全补丁修了哪些洞", "Which vulnerabilities were fixed in this month's Android security bulletin?", "security_advisory", "realtime", "official_required", ["source.android.com"], ["/docs/security/bulletin"], "Android monthly security bulletin", "conversational", "security_advisory", 14),
    topic("ubuntu最新USN安全公告", "Latest Ubuntu USN security notices", "security_advisory", "realtime", "official_required", ["ubuntu.com"], ["/security/notices"], "Ubuntu security notices", "terse", "security_advisory", 14),
    topic("Debian security tracker 查最新高危漏洞", "Find recent high-severity issues in the Debian Security Tracker.", "security_advisory", "realtime", "official_required", ["security-tracker.debian.org"], ["/tracker"], "Debian Security Tracker", "canonical", "security_advisory", 14),
    topic("CISA已知被利用漏洞目录最近加了什么", "What was recently added to the CISA Known Exploited Vulnerabilities catalog?", "security_advisory", "realtime", "official_required", ["cisa.gov"], ["/known-exploited-vulnerabilities-catalog"], "CISA KEV catalog updates", "conversational", "security_advisory", 7),
    topic("NVD上最新的OpenSSH CVE详情", "Details of the newest OpenSSH CVE in NVD", "security_advisory", "latest", "primary_required", ["nvd.nist.gov"], ["/vuln/detail"], "NVD OpenSSH vulnerability detail", "canonical", "security_advisory", 14),
    topic("Firefox最近安全更新公告", "Recent Firefox security update advisories", "security_advisory", "latest", "official_required", ["mozilla.org"], ["/security/advisories"], "Mozilla security advisories", "terse", "security_advisory", 14),
    topic("GitHub advisory数据库里近期的PyPI高危漏洞", "Recent high-severity PyPI vulnerabilities in the GitHub Advisory Database", "security_advisory", "latest", "primary_required", ["github.com"], ["/advisories"], "GitHub Advisory Database search", "canonical", "security_advisory", 14),
    topic("openssl最近公布了啥漏洞", "What vulnerabilities has OpenSSL disclosed recently?", "security_advisory", "latest", "official_required", ["openssl-library.org"], ["/news/vulnerabilities"], "OpenSSL vulnerability announcements", "conversational", "security_advisory", 30),
    topic("Kubernetes官方CVE列表最新更新", "Latest updates to the official Kubernetes CVE feed", "security_advisory", "latest", "official_required", ["kubernetes.io"], ["/docs/reference/issues-security/official-cve-feed"], "Kubernetes official CVE feed", "canonical", "security_advisory", 30),
    topic("curl安全漏洞公告原文", "Official curl security advisories", "security_advisory", "latest", "official_required", ["curl.se"], ["/docs/security"], "curl security advisories", "terse", "security_advisory", 30),
    topic("现在最新的C++标准是哪一版 ISO页面", "What is the current C++ standard? Find its ISO record.", "standards_specification", "latest", "primary_required", ["iso.org"], ["/standard"], "ISO C++ standard record", "conversational", "standards_lookup", 180),
    topic("ECMAScript当前规范原文", "Current normative ECMAScript specification", "standards_specification", "latest", "primary_required", ["tc39.es"], ["/ecma262"], "ECMAScript language specification", "terse", "standards_lookup", 90),
    topic("HTML Living Standard 最新原文", "Current HTML Living Standard normative text", "standards_specification", "latest", "primary_required", ["html.spec.whatwg.org"], ["/"], "WHATWG HTML Living Standard", "terse", "standards_lookup", 90),
    topic("HTTP Semantics RFC 9110原文和勘误", "HTTP Semantics RFC 9110 original text and errata", "standards_specification", "stable", "primary_required", ["rfc-editor.org"], ["/rfc/rfc9110"], "HTTP Semantics RFC", "canonical", "standards_lookup", 730),
    topic("WebGPU规范现在到哪了", "What is the current WebGPU specification?", "standards_specification", "latest", "primary_required", ["gpuweb.github.io"], ["/gpuweb"], "WebGPU specification", "conversational", "standards_lookup", 90),
    topic("国家统计局最新人口数据公报", "Latest official China population statistics release", "statistics", "latest", "official_required", ["stats.gov.cn"], ["/sj/zxfb", "/sjjd"], "China population statistics release", "canonical", "public_statistics", 30),
    topic("央行最新货币政策执行报告", "Latest PBOC monetary policy implementation report", "government_policy", "latest", "official_required", ["pbc.gov.cn"], ["/zhengcehuobisi"], "PBOC monetary policy report", "canonical", "government_publication", 30),
    topic("工信部最新通信业统计数据", "Latest MIIT telecommunications industry statistics", "statistics", "realtime", "official_required", ["miit.gov.cn"], ["/gxsj/tjfx"], "MIIT telecom statistics", "canonical", "public_statistics", 30),
    topic("最近有哪些消费品召回 市监总局原文", "Recent consumer product recalls from China's market regulator", "realtime_public_info", "realtime", "official_required", ["samr.gov.cn"], ["/zw/zfxxgk", "/xw"], "SAMR product recall notices", "conversational", "public_recall", 14),
    topic("国家药监局最新药品召回公告", "Latest drug recall announcement from China's medical products regulator", "realtime_public_info", "latest", "official_required", ["nmpa.gov.cn"], ["/xxgk/ggtg"], "NMPA drug recall announcements", "canonical", "public_recall", 30),
    topic("全国现在有哪些气象预警", "What official weather warnings are active across China now?", "realtime_public_info", "realtime", "official_required", ["weather.cma.cn"], ["/web/weather/warning"], "China national weather warnings", "conversational", "public_warning", 1),
    topic("刚刚地震了吗 查中国地震台网", "Was there an earthquake just now? Check China Earthquake Networks.", "realtime_public_info", "realtime", "official_required", ["news.ceic.ac.cn"], ["/index.html", "/CC"], "China Earthquake Networks latest events", "conversational", "public_warning", 1),
    topic("美国最新生产者价格指数PPI", "Latest US Producer Price Index release", "statistics", "realtime", "official_required", ["bls.gov"], ["/news.release/ppi", "/ppi"], "US Producer Price Index release", "terse", "public_statistics", 30),
    topic("美国能源署每周石油状况报告", "Latest EIA Weekly Petroleum Status Report", "statistics", "realtime", "official_required", ["eia.gov"], ["/petroleum/supply/weekly"], "EIA weekly petroleum report", "canonical", "public_statistics", 7),
    topic("现在有没有太空天气预警", "Are there NOAA space weather alerts right now?", "realtime_public_info", "realtime", "official_required", ["swpc.noaa.gov"], ["/products/alerts-watches-and-warnings"], "NOAA space weather alerts", "conversational", "public_warning", 1),
    topic("CDC现在有哪些旅行健康提醒", "Current CDC travel health notices", "realtime_public_info", "realtime", "official_required", ["wwwnc.cdc.gov"], ["/travel/notices"], "CDC travel health notices", "canonical", "health_warning", 7),
    topic("伯克希尔最新13F去SEC哪里看", "Where can I find Berkshire Hathaway's newest 13F on SEC EDGAR?", "company_filing", "latest", "primary_required", ["sec.gov"], ["/Archives/edgar/data", "/edgar/browse"], "Berkshire Hathaway SEC filing", "conversational", "company_filing", 30),
    topic("小米最新港交所公告", "Latest Xiaomi announcement on HKEX", "company_filing", "realtime", "primary_required", ["hkexnews.hk"], ["/listedco/listconews"], "Xiaomi HKEX announcements", "terse", "company_filing", 14),
    topic("AMD最新季度财报原文", "AMD latest quarterly earnings release", "company_filing", "latest", "primary_required", ["ir.amd.com"], ["/news-events/press-releases", "/financial-information"], "AMD investor relations results", "terse", "company_filing", 30),
    topic("Cloudflare最近的宕机复盘", "Cloudflare's latest official outage postmortem", "newsroom", "latest", "official_required", ["blog.cloudflare.com"], ["/tag/outage", "/"], "Cloudflare outage postmortem", "conversational", "incident_report", 30),
    topic("iPhone有没有新的官方维修计划", "Does Apple have a new iPhone service program?", "product_support", "latest", "official_required", ["support.apple.com"], ["/service-programs"], "Apple service programs", "conversational", "product_support", 30),
    topic("WHO最新临床指南从哪里下载", "Where can I download the newest WHO clinical guidelines?", "official_docs", "latest", "official_required", ["who.int"], ["/publications/i"], "WHO publication and guideline index", "conversational", "health_guideline", 30),
    topic("NASA阿尔忒弥斯任务最近进展", "Latest official progress on NASA's Artemis missions", "newsroom", "latest", "official_required", ["nasa.gov"], ["/humans-in-space/artemis"], "NASA Artemis updates", "conversational", "official_news", 14),
    topic("Mamba模型最早那篇论文", "Original Mamba state-space model paper", "academic_paper", "stable", "original_required", ["arxiv.org"], ["/abs/2312.00752", "/pdf/2312.00752"], "Original Mamba paper", "terse", "academic_source", 730),
    topic("llama.cpp最近大家都在报什么问题", "What problems are llama.cpp users reporting recently?", "community_discussion", "latest", "community_required", ["github.com"], ["/ggml-org/llama.cpp/issues"], "llama.cpp current issue discussions", "conversational", "developer_community", 14),
    topic("vLLM最近有哪些CUDA报错issue", "Recent vLLM CUDA error reports from users", "community_discussion", "latest", "community_required", ["github.com"], ["/vllm-project/vllm/issues"], "vLLM current issue reports", "conversational", "developer_community", 14),
    topic("v2ex上大家升级macOS后的真实体验", "Real user experiences after macOS upgrades on V2EX", "community_discussion", "latest", "community_required", ["v2ex.com"], ["/go/apple", "/t/"], "V2EX Apple community experiences", "conversational", "user_experience", 14),
    topic("reddit LocalLLaMA里RWKV实际体验", "reddit LocalLLaMA real world RWKV experience", "community_discussion", "latest", "community_required", ["reddit.com"], ["/r/LocalLLaMA"], "Reddit LocalLLaMA RWKV experiences", "noisy", "user_experience", 14),
    topic("stackoverflow docker compose gpu 常见报错", "Stack Overflow questions about Docker Compose GPU errors", "community_discussion", "latest", "community_required", ["stackoverflow.com"], ["/questions/tagged/docker-compose", "/questions/"], "Stack Overflow Docker Compose GPU troubleshooting", "noisy", "developer_community", 30),
    topic("Hacker News上怎么评价最新PostgreSQL版本", "Hacker News discussion about the latest PostgreSQL release", "community_discussion", "latest", "community_required", ["news.ycombinator.com"], ["/item"], "Hacker News PostgreSQL discussion", "conversational", "developer_community", 14),
]


def build_rows() -> list[dict[str, Any]]:
    if len(TOPICS) != 50:
        raise RuntimeError(f"expected 50 bilingual topics, got {len(TOPICS)}")
    rows = []
    for language in ("zh", "en"):
        for number, value in enumerate(TOPICS, 101):
            row = {
                "schema_version": "realtime-retrieval-case.v1",
                "id": f"retrieval-{language}-{number:03d}",
                "query": value[language],
                "language": language,
                "category": value["category"],
                "freshness": value["freshness"],
                "source_policy": value["source_policy"],
                "expected_domains_any": value["domains"],
                "target_url_patterns_any": value["patterns"],
                "forbidden_result_types": FORBIDDEN,
                "notes": value["notes"],
                "query_style": value["query_style"],
                "task_family": value["task_family"],
                "gold_ttl_days": value["gold_ttl_days"],
                "annotation_status": "source_policy_reviewed",
                "origin": "manually_curated_realistic",
            }
            rows.append(row)
    return rows


def main() -> None:
    output = Path(__file__).with_name("realtime_web_retrieval_dev_v2.jsonl")
    rows = build_rows()
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "retrieval-development-set-manifest.v1",
        "name": "realtime-web-retrieval-dev-v2",
        "status": "development",
        "case_count": len(rows),
        "paired_topic_count": len(TOPICS),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "language_counts": dict(Counter(row["language"] for row in rows)),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "freshness_counts": dict(Counter(row["freshness"] for row in rows)),
        "source_policy_counts": dict(
            Counter(row["source_policy"] for row in rows)
        ),
        "query_style_counts": dict(Counter(row["query_style"] for row in rows)),
        "origin": "manually_curated_realistic",
        "is_user_log": False,
        "is_blind_test": False,
        "runtime_label_visibility": False,
    }
    manifest_path = output.with_name("realtime_web_retrieval_dev_v2_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} cases to {output}")


if __name__ == "__main__":
    main()
