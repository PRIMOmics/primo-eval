#!/usr/bin/env python3
"""Post-tool hook to remove standalone comment lines added by Claude."""

import json
import re
import sys
from pathlib import Path

PRESERVED_PATTERNS = [
    r"^\s*#\s*TODO",
    r"^\s*#\s*FIXME",
    r"^\s*#\s*NOTE",
    r"^\s*#\s*XXX",
    r"^\s*#\s*HACK",
    r"^\s*#\s*WARNING",
    r"^\s*#\s*type:",
    r"^\s*#\s*noqa",
    r"^\s*#\s*pragma",
    r"^\s*#\s*pylint",
    r"^\s*#\s*pyright",
    r"^\s*#\s*ruff",
    r"^\s*#\s*mypy:",
    r"^\s*#\s*fmt:\s*(on|off)",
    r"^\s*#\s*isort:\s*(skip|split|off|on)",
    r"^#!",
    r"^#\s*-\*-",
    r"^#\s*coding[=:]",
    r"^\s*#\s*[-=]{3,}",
    r"^\s*#\s*region",
    r"^\s*#\s*endregion",
]

PRESERVED_RE = re.compile("|".join(PRESERVED_PATTERNS), re.IGNORECASE)


def should_preserve_comment(line: str) -> bool:
    """Check if a comment line should be preserved."""
    return bool(PRESERVED_RE.search(line))


def is_standalone_comment(line: str) -> bool:
    """Check if a line is a standalone comment (not trailing)."""
    stripped = line.strip()
    return stripped.startswith("#") and not should_preserve_comment(stripped)


def find_added_lines(old_string: str, new_string: str) -> set[str]:
    """Find lines that were added (present in new but not in old)."""
    old_lines = set(old_string.split("\n"))
    new_lines = new_string.split("\n")
    return {line for line in new_lines if line not in old_lines}


def remove_added_comment_lines(file_path: str, added_lines: set[str]) -> None:
    """Remove standalone comment lines that were added."""
    path = Path(file_path)

    if not path.exists() or path.suffix != ".py":
        return

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        result = []

        for line in lines:
            if line in added_lines and is_standalone_comment(line):
                continue
            result.append(line)

        new_content = "\n".join(result)

        if new_content != content:
            path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        print(f"remove-comments hook error: {e}", file=sys.stderr)


def main() -> None:
    """Main entry point."""
    try:
        input_data = json.load(sys.stdin)
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        file_path = tool_input.get("file_path", "")

        if not file_path or not file_path.endswith(".py"):
            return

        if tool_name == "Edit":
            old_string = tool_input.get("old_string", "")
            new_string = tool_input.get("new_string", "")
            added_lines = find_added_lines(old_string, new_string)
            if added_lines:
                remove_added_comment_lines(file_path, added_lines)

        elif tool_name == "Write":
            path = Path(file_path)
            if path.exists():
                content = path.read_text(encoding="utf-8")
                lines = content.split("\n")
                result = [line for line in lines if not is_standalone_comment(line)]
                new_content = "\n".join(result)
                if new_content != content:
                    path.write_text(new_content, encoding="utf-8")

    except Exception as e:
        print(f"remove-comments hook error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
