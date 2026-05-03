from __future__ import annotations

import json


def parse_partial_json(text: str) -> object | None:
    """Best-effort parse of possibly-incomplete JSON.

    Strips common wrappers (code fences, leading prose), then attempts ``json.loads``.
    If that fails, closes any open strings / arrays / objects based on a simple scanner
    and retries. Returns ``None`` if nothing parseable is present.
    """
    s = _strip_prose(text)
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    repaired = _close_open_structures(s)
    if repaired != s:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return None


def _strip_prose(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # drop opening fence (optional language tag)
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    brace = s.find("{")
    bracket = s.find("[")
    start = bracket if bracket != -1 and (brace == -1 or bracket < brace) else brace
    if start == -1:
        return ""
    return s[start:]


def _close_open_structures(s: str) -> str:
    stack: list[str] = []
    in_string = False
    escape = False
    last_complete = len(s)
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            if not stack:
                last_complete = i + 1
    if not stack and not in_string:
        return s[:last_complete]

    repaired = s
    if in_string:
        # trim back to the last field boundary — a dangling string would break a number too
        cut = _rfind_safe_cut(repaired)
        repaired = repaired[:cut]
        # recompute stack on trimmed string
        return _close_open_structures(repaired)

    # trim trailing trailing commas / partial literals
    trimmed = repaired.rstrip()
    while trimmed.endswith((",", ":")):
        trimmed = trimmed[:-1].rstrip()
    # drop incomplete key-without-value patterns like `"key":`
    repaired = trimmed
    # recompute unclosed brackets on the trimmed string
    stack2: list[str] = []
    in_string2 = False
    escape2 = False
    for ch in repaired:
        if in_string2:
            if escape2:
                escape2 = False
                continue
            if ch == "\\":
                escape2 = True
                continue
            if ch == '"':
                in_string2 = False
            continue
        if ch == '"':
            in_string2 = True
        elif ch in "{[":
            stack2.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack2 and stack2[-1] == ch:
            stack2.pop()
    if in_string2:
        # still in a string after trimming — give up
        return s
    return repaired + "".join(reversed(stack2))


def _rfind_safe_cut(s: str) -> int:
    """Find the last position that ends on a field boundary (``,`` or ``{`` or ``[``)."""
    for i in range(len(s) - 1, -1, -1):
        ch = s[i]
        if ch in ",{[":
            return i + 1
    return 0
