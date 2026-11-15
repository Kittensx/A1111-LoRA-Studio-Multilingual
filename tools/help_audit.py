#!/usr/bin/env python3
"""Audit LoRA Studio's runtime help documentation without installing or changing files."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'files/extensions-builtin/Lora/lora_help_registry.py'
module_spec = importlib.util.spec_from_file_location('lora_help_registry', SCRIPT)
registry = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(registry)


def main():
    topics = registry.available_topics()
    if not topics:
        print('ERROR: No user-facing help articles were found in help_documentation.', file=sys.stderr)
        return 1
    problems = []
    langs = registry.SUPPORTED
    for lang in langs:
        folder = registry.ROOT / lang
        count = sum((folder / (name + '.md')).is_file() for name in topics)
        fallback = ' (English fallback for missing articles)' if lang != 'en' and count < len(topics) else ''
        print(f'  {lang:6s} {count:3d}/{len(topics)} translated topics{fallback}')
    ids = set(topics)
    for topic in topics:
        english = registry.ROOT / 'en' / (topic + '.md')
        if not english.is_file():
            problems.append(f'{topic}: missing English fallback')
        try:
            item = registry.article(topic, 'en')
            if not item['title'] or not item['summary']:
                problems.append(f'{topic}: missing title or summary')
            for related in item['related']:
                if related not in ids or related == topic:
                    problems.append(f'{topic}: invalid related topic: {related}')
            for ctx in item['contexts']:
                if not ctx.startswith('studio.'):
                    problems.append(f'{topic}: invalid context ID: {ctx}')
        except (registry.HelpError, OSError, UnicodeError) as e:
            problems.append(f'{topic}: {e}')
    # Catch duplicate context IDs: two articles must not silently own one button.
    contexts = {}
    for topic in topics:
        try:
            for context in registry.metadata(topic).get('contexts',[]):
                if context in contexts:
                    problems.append(f'{context}: mapped by {contexts[context]} and {topic}')
                contexts[context] = topic
        except registry.HelpError as e:
            problems.append(f'{topic}: {e}')
    print(f'Help registry: {len(topics)} topics, {len(contexts)} mapped UI contexts.')
    if problems:
        for problem in problems:
            print('ERROR: '+problem, file=sys.stderr)
        return 1
    print('Related-topic links, context IDs and English fallbacks verified.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
