"""Read-only LoRA Studio contextual help registry (stdlib only).

Discovers Markdown and optional JSON sidecars on every request, so editing or
adding a document on disk requires no application rebuild. No arbitrary paths
are accepted from HTTP clients. This module does not parse or execute Markdown.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent / "help_documentation"
SUPPORTED = ("en", "ru", "fr", "es", "it", "ja", "zh-CN", "pt-BR", "de", "ko")
TOPIC_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
MAX_MARKDOWN_BYTES = 65536
MAX_METADATA_BYTES = 16384
MAX_TOPICS = 150

class HelpError(ValueError):
    pass


def locale_code(value):
    raw = str(value or "en").replace("_", "-")
    if raw in SUPPORTED:
        return raw
    base = raw.split("-")[0].lower()
    if base == "pt":
        return "pt-BR"
    if base == "zh":
        return "zh-CN"
    return base if base in SUPPORTED else "en"


def valid_topic(topic):
    if not isinstance(topic, str) or not TOPIC_RE.fullmatch(topic):
        raise HelpError("Invalid help topic")
    return topic


def read_small_text(path, limit):
    # Even if a hostile local edit changes an article to a symlink, do not
    # expose files outside the help folder through the API.
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise HelpError("Help file is missing or exceeds the size limit")
    return path.read_text(encoding="utf-8-sig")


def metadata(topic, root=ROOT):
    valid_topic(topic)
    directory = root / "registry"
    if directory.is_symlink():
        raise HelpError("Help metadata directory cannot be a symlink")
    path = directory / (topic + ".json")
    if not path.exists():
        return {}
    try:
        data = json.loads(read_small_text(path, MAX_METADATA_BYTES))
    except (ValueError, UnicodeError, OSError) as error:
        raise HelpError("Invalid help metadata for " + topic) from error
    if not isinstance(data, dict):
        raise HelpError("Help metadata must be an object")
    result = {"related": [], "contexts": [], "tags": [], "order": 100}
    for key in ("related", "contexts", "tags"):
        values = data.get(key, [])
        if not isinstance(values, list) or len(values) > 40:
            raise HelpError("Invalid help metadata list")
        if key == "related":
            result[key] = [valid_topic(value) for value in values]
        else:
            result[key] = [value for value in values if isinstance(value, str) and len(value) <= 80]
    order = data.get("order", 100)
    result["order"] = order if isinstance(order, int) and 0 <= order <= 10000 else 100
    return result


def available_topics(root=ROOT):
    found = set()
    for lang in SUPPORTED:
        directory = root / lang
        if not directory.is_dir() or directory.is_symlink():
            continue
        for file in directory.glob("*.md"):
            if TOPIC_RE.fullmatch(file.stem) and not file.is_symlink():
                found.add(file.stem)
    return sorted(found)[:MAX_TOPICS]


def locate(topic, locale, root=ROOT):
    valid_topic(topic)
    requested = locale_code(locale)
    for language in (requested, "en"):
        directory = root / language
        if directory.is_symlink():
            continue
        candidate = directory / (topic + ".md")
        if candidate.is_file() and not candidate.is_symlink():
            return candidate, language
    raise HelpError("Help topic not found")


def summarize(content):
    lines = content.splitlines()
    title = next((line.lstrip("# ").strip() for line in lines if line.startswith("# ") and line.strip("# ")), "Help")
    description = next((line.strip() for line in lines if line.strip() and not line.lstrip().startswith(("#", "```", "- ", "* ", ">"))), "")
    return title[:180], description[:350]


def article(topic, locale="en", root=ROOT):
    valid_topic(topic)
    file, used = locate(topic, locale, root)
    content = read_small_text(file, MAX_MARKDOWN_BYTES)
    title, summary = summarize(content)
    info = metadata(topic, root)
    return {"id": topic, "title": title, "summary": summary,
            "content": content, "locale": used, "requested_locale": locale_code(locale),
            "fallback": used != locale_code(locale), "related": info.get("related", []),
            "tags": info.get("tags", []), "contexts": info.get("contexts", []),
            "order": info.get("order", 100)}


def index(locale="en", root=ROOT):
    requested = locale_code(locale)
    results = []
    for topic in available_topics(root):
        try:
            item = article(topic, requested, root)
        except (OSError, UnicodeError, HelpError):
            continue
        item.pop("content", None)
        results.append(item)
    results.sort(key=lambda item: (item["order"], item["title"].casefold()))
    ids = {item["id"] for item in results}
    for item in results:
        item["related"] = [target for target in item["related"] if target in ids and target != item["id"]]
    return {"schema_version": 1, "locale": requested, "topics": results}
