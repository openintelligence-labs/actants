"""Provider-native constrained decoding for :meth:`~actants.llm.client.LLM.extract`.

Every provider that can guarantee schema-valid output does it differently — OpenAI
takes a ``response_format`` block, Anthropic has no such parameter and instead needs a
forced single tool call, Gemini nests the schema under ``generationConfig``, Ollama
takes it as ``format``. What they share is the JSON Schema itself, so a provider
declares :attr:`~actants.llm.base.BaseLLMProvider.native_schema_mode` and this module
turns one pydantic model into the request that mode needs.

The prompt-based path is not going away: it is what runs when a provider has no native
mode, and what runs when a schema cannot be expressed in the provider's dialect.
:class:`SchemaPlan` records which of the two happened so a caller can assert on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

# Re-exported so the translators and the error they raise are importable together; the
# class lives in actants.errors with the rest of the hierarchy, so `except ActantsError`
# stays exhaustive.
from actants.errors import UnsupportedSchemaError as UnsupportedSchemaError

#: How a provider asks the model for schema-valid output on the wire.
#:
#: ``"none"`` means it cannot, and :meth:`~actants.llm.client.LLM.extract` falls back to
#: describing the schema in a system prompt.
NativeSchemaMode = Literal["none", "openai_json_schema", "anthropic_tool", "gemini", "ollama"]

NATIVE_SCHEMA_MODES: tuple[NativeSchemaMode, ...] = (
    "none",
    "openai_json_schema",
    "anthropic_tool",
    "gemini",
    "ollama",
)

#: The tool the Anthropic path forces. The model has exactly one tool available and is
#: required to call it, so its ``input_schema`` becomes the grammar for the response.
ANTHROPIC_EXTRACT_TOOL = "record_extraction"

#: Validation keywords strict mode rejects, which are *dropped* rather than treated as a
#: fallback trigger. Each is redundant with the pydantic validation that runs on the
#: response either way, so dropping one costs a hint to the model and keeps the request
#: legal — whereas sending it is a 400, since the API rejects unknown keywords rather
#: than ignoring them.
#:
#: Deliberately short. Strict mode was relaxed in 2025 to accept ``pattern``, ``format``,
#: the numeric bounds, and the array-length bounds, so those are forwarded now; only the
#: keywords still rejected are listed here.
_STRICT_DROPPED_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "patternProperties",
        "propertyNames",
        "contains",
        "minContains",
        "maxContains",
        "default",
        "examples",
    }
)

#: Keywords with no strict equivalent that also cannot be dropped, because dropping one
#: silently widens what the model may return. These decline the native path instead.
_STRICT_FATAL_KEYWORDS = frozenset(
    {"oneOf", "not", "if", "then", "else", "dependentRequired", "dependentSchemas"}
)

#: String formats strict mode recognises. An unrecognised one is rejected outright, so
#: anything else is dropped and left to pydantic.
_STRICT_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}
)

#: Keywords that carry no constraint and are safe to keep for the model's benefit.
_STRICT_ANNOTATION_KEYWORDS = frozenset({"title", "description"})


@dataclass(frozen=True)
class SchemaPlan:
    """What one ``extract`` call decided to send, and why.

    Attributes:
        native: Whether provider-native constrained decoding is being used. ``False``
            means the schema went into the prompt instead.
        mode: The provider's declared mode. Note this is the provider's *capability*,
            not what ran — a plan can have ``mode="openai_json_schema"`` and
            ``native=False`` when the schema turned out not to be strict-compatible.
        request_kwargs: Provider parameters to forward verbatim. Empty on the prompt
            path.
        reason: Why the native path was declined. ``None`` when it was taken.
        nulls_mean_default: Paths of fields strict mode had to widen to ``["T", "null"]``
            purely to express "may be absent". A conforming model may return ``null`` for
            any of them, which is *not* a value the pydantic model accepts — it means
            "use the default". Each entry is a tuple of property names from the document
            root. Empty on every non-strict path, which needs no such repair.
    """

    native: bool
    mode: NativeSchemaMode
    request_kwargs: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    nulls_mean_default: frozenset[tuple[str, ...]] = frozenset()


def build_schema_plan(
    schema: type[BaseModel],
    mode: NativeSchemaMode,
    *,
    streaming: bool = False,
) -> SchemaPlan:
    """Decide how ``schema`` should be requested from a provider in ``mode``.

    Falls back rather than raising: a schema the provider cannot express yields a plan
    with ``native=False`` carrying the reason, because a caller asked for an extraction,
    not for a particular transport.

    ``streaming`` excludes modes that cannot produce incremental text.
    :meth:`~actants.llm.client.LLM.extract_stream` yields progressively-complete objects
    parsed from a text stream, and the Anthropic tool path emits its JSON as tool-call
    input deltas rather than text — so that path is declined for streams instead of
    silently yielding nothing.
    """
    if mode == "none":
        return SchemaPlan(native=False, mode=mode, reason="provider declares no native schema mode")
    if streaming and mode == "anthropic_tool":
        return SchemaPlan(
            native=False,
            mode=mode,
            reason="a forced tool call streams as tool-call input, not as text deltas",
        )

    raw = schema.model_json_schema()
    widened: set[tuple[str, ...]] = set()
    try:
        return SchemaPlan(
            native=True,
            mode=mode,
            request_kwargs=_request_kwargs(schema.__name__, raw, mode, widened),
            nulls_mean_default=frozenset(widened),
        )
    except UnsupportedSchemaError as exc:
        return SchemaPlan(native=False, mode=mode, reason=str(exc))


def _request_kwargs(
    name: str,
    raw: dict[str, Any],
    mode: NativeSchemaMode,
    widened: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    if mode == "openai_json_schema":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _tool_safe_name(name),
                    "schema": to_strict_schema(raw, widened=widened),
                    "strict": True,
                },
            }
        }
    if mode == "anthropic_tool":
        # A ToolSpec rather than a raw dict: `tools` is a modelled parameter of
        # LLM.complete, which validates it and folds it into the cache key. The provider
        # turns `parameters` into Anthropic's `input_schema`, so the schema still lands
        # where a forced tool call needs it.
        from actants.llm.base import ToolSpec

        return {
            "tools": [
                ToolSpec(
                    name=ANTHROPIC_EXTRACT_TOOL,
                    description=f"Record the extracted {name}. Call this exactly once.",
                    parameters=to_strict_schema(raw, widened=widened),
                )
            ],
            "tool_choice": {"type": "tool", "name": ANTHROPIC_EXTRACT_TOOL},
        }
    if mode == "gemini":
        return {
            "responseMimeType": "application/json",
            "responseSchema": to_gemini_schema(raw),
        }
    if mode == "ollama":
        # Ollama takes a whole JSON Schema in `format`, not only the string "json", and
        # its llama.cpp grammar backend handles $defs/$ref — so the schema goes as-is.
        return {"format": _checked_for_ollama(raw)}
    raise UnsupportedSchemaError(f"unknown native schema mode {mode!r}")  # pragma: no cover


def _tool_safe_name(name: str) -> str:
    """OpenAI's ``json_schema.name`` accepts ``[a-zA-Z0-9_-]`` only."""
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    return cleaned or "extraction"


def to_strict_schema(
    raw: dict[str, Any],
    *,
    widened: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Rewrite a pydantic schema into OpenAI strict / Anthropic strict form.

    Strict mode is not a subset of JSON Schema that ``model_json_schema()`` happens to
    emit — every object must list all of its properties in ``required`` and set
    ``additionalProperties: false``, and a long list of validation keywords is rejected
    outright. The two rules are handled differently on purpose:

    * **Unsupported keywords are dropped.** They are redundant with the pydantic
      validation that runs on the response either way, so dropping them costs a hint to
      the model and keeps the request legal.
    * **Optional fields become nullable and required.** Strict mode has no way to say
      "may be absent"; the documented encoding is a union with ``null``, which is also
      what pydantic already emits for ``X | None``.

    That second rule widens a field the *pydantic* model may not accept as null — a
    ``priority: int = 3`` becomes ``["integer", "null"]``, and a conforming provider is
    then entitled to return null. ``widened`` collects the paths of exactly those fields
    so the caller can read a null back as "the field was absent, use its default"; see
    :attr:`SchemaPlan.nulls_mean_default`.

    Raises:
        UnsupportedSchemaError: The schema uses something with no strict equivalent —
            an unconstrained ``dict``, or a ``$ref`` cycle (strict mode forbids
            recursion, and a recursive model cannot be unrolled).
    """
    defs = raw.get("$defs", {})
    if not isinstance(defs, dict):  # pragma: no cover - pydantic always emits a dict
        raise UnsupportedSchemaError("$defs is not an object")
    converted = _strict_node(raw, defs, (), path=(), widened=widened)
    converted.pop("$defs", None)
    return converted


def _strict_node(
    node: Any,
    defs: dict[str, Any],
    seen: tuple[str, ...],
    *,
    path: tuple[str, ...] = (),
    widened: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise UnsupportedSchemaError(f"schema fragment is not an object: {node!r}")

    ref = node.get("$ref")
    if isinstance(ref, str):
        # Inlined rather than preserved: strict mode allows $ref, but inlining removes
        # the $defs-scoping question entirely and makes the cycle check trivial.
        key = _def_key(ref)
        if key in seen:
            raise UnsupportedSchemaError(
                f"recursive model {key!r}: provider strict mode does not support recursive schemas"
            )
        target = defs.get(key)
        if target is None:
            raise UnsupportedSchemaError(f"dangling $ref {ref!r}")
        merged = _strict_node(target, defs, (*seen, key), path=path, widened=widened)
        # A sibling `description` on the $ref (pydantic emits one for a documented
        # field) is the field's, and is worth keeping over the model's own.
        for keyword in _STRICT_ANNOTATION_KEYWORDS:
            if keyword in node:
                merged[keyword] = node[keyword]
        return merged

    # pydantic wraps a $ref in a single-element `allOf` whenever the field carries a
    # title or description. Strict mode rejects `allOf`, but the wrapper carries no
    # constraint of its own — so it is unwrapped rather than dropped, which would
    # otherwise discard the entire referenced subschema and leave a bare `{}`.
    all_of = node.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) != 1:
            raise UnsupportedSchemaError(
                "strict mode does not support `allOf` with more than one branch"
            )
        merged = _strict_node(all_of[0], defs, seen, path=path, widened=widened)
        for keyword in _STRICT_ANNOTATION_KEYWORDS:
            if keyword in node:
                merged[keyword] = node[keyword]
        return merged

    fatal = _STRICT_FATAL_KEYWORDS.intersection(node)
    if fatal:
        raise UnsupportedSchemaError(f"strict mode does not support {', '.join(sorted(fatal))}")

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _STRICT_DROPPED_KEYWORDS or key in {"$defs", "$ref"}:
            continue
        if key == "format":
            # An unrecognised format is a 400, not a no-op — keep only the known set.
            if value in _STRICT_FORMATS:
                out["format"] = value
        elif key == "properties":
            if not isinstance(value, dict):
                raise UnsupportedSchemaError("`properties` is not an object")
            out["properties"] = {
                k: _strict_node(v, defs, seen, path=(*path, k), widened=widened)
                for k, v in value.items()
            }
        elif key == "items":
            # Items have no property name of their own; a null inside an array is the
            # array's business, not a missing-field signal, so the path stops here.
            out["items"] = _strict_node(value, defs, seen)
        elif key == "anyOf":
            if not isinstance(value, list):
                raise UnsupportedSchemaError("`anyOf` is not a list")
            out["anyOf"] = [_strict_node(v, defs, seen) for v in value]
        elif key == "additionalProperties":
            continue  # re-derived below
        else:
            out[key] = value

    if out.get("type") == "object" or "properties" in out:
        properties = out.setdefault("properties", {})
        if not properties:
            # `dict[str, Any]` and bare `dict` land here: no properties, and strict mode
            # forbids the open-ended `additionalProperties` that would describe them.
            # There is nothing to constrain, so the native path cannot represent it.
            raise UnsupportedSchemaError(
                "object with no declared properties (e.g. a bare `dict` field) cannot be "
                "expressed in strict mode, which forbids additionalProperties"
            )
        out["additionalProperties"] = False
        # Strict mode requires every property in `required`; optionality is expressed by
        # allowing null instead. pydantic already emits `anyOf: [..., {"type": "null"}]`
        # for `X | None`, so this only widens the fields that had a non-null default.
        original_required = set(node.get("required", []) or [])
        for prop_name, prop in properties.items():
            if prop_name not in original_required:
                nullable = _nullable(prop)
                if widened is not None and nullable is not prop:
                    # Only a field this rewrite *made* nullable is a "null means absent"
                    # field. One pydantic already emitted as `X | None` accepts null as a
                    # real value, and rewriting that to the default would be wrong.
                    widened.add((*path, prop_name))
                properties[prop_name] = nullable
        out["required"] = list(properties)

    return out


def drop_defaulted_nulls(payload: Any, paths: frozenset[tuple[str, ...]]) -> Any:
    """Delete the nulls strict mode's widening invited, so defaults apply instead.

    Strict mode cannot say "this field may be absent", only "it may be null", so a
    provider doing exactly what it was told may answer ``{"priority": null}`` for a
    ``priority: int = 3``. Pydantic rightly refuses that null — but the model was not
    expressing a value, it was expressing absence. Removing the key restores what the
    schema meant, and pydantic then fills in the real default.

    Only paths in ``paths`` are touched, so a field genuinely declared ``X | None`` keeps
    its null. Returns ``payload`` unchanged when there is nothing to strip.
    """
    if not paths or not isinstance(payload, dict):
        return payload
    heads = {p[0] for p in paths}
    out = dict(payload)
    for head in heads:
        if head not in out:
            continue
        deeper = frozenset(p[1:] for p in paths if p[0] == head and len(p) > 1)
        if out[head] is None and (head,) in paths:
            del out[head]
        elif deeper:
            out[head] = drop_defaulted_nulls(out[head], deeper)
    return out


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    """Widen a schema so it also accepts ``null``, without nesting anyOf inside anyOf."""
    if "anyOf" in node:
        branches = node["anyOf"]
        if any(isinstance(b, dict) and b.get("type") == "null" for b in branches):
            return node
        return {**node, "anyOf": [*branches, {"type": "null"}]}
    node_type = node.get("type")
    if isinstance(node_type, str):
        if node_type == "null":
            return node
        return {**node, "type": [node_type, "null"]}
    if isinstance(node_type, list):
        return node if "null" in node_type else {**node, "type": [*node_type, "null"]}
    # No `type` to widen (e.g. a bare enum): wrap it rather than guess.
    return {"anyOf": [node, {"type": "null"}]}


#: Keywords Gemini's ``responseSchema`` accepts. It is an OpenAPI 3.0 Schema subset, not
#: JSON Schema, so unknown keywords are rejected rather than ignored — the safe move is
#: an allowlist. ``$ref``/``$defs`` are absent because Gemini has no ``$defs`` section;
#: nested models are inlined instead.
_GEMINI_KEYWORDS = frozenset(
    {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "items",
        "properties",
        "required",
        "propertyOrdering",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
    }
)

_GEMINI_TYPES = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}

#: The string formats Gemini recognises. Anything else (``email``, ``uri``, ...) is
#: dropped: an unrecognised ``format`` is rejected, and pydantic validates it anyway.
_GEMINI_FORMATS = {"date-time", "date", "time", "duration", "int32", "int64", "float", "double"}


def to_gemini_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a pydantic schema into Gemini's ``responseSchema`` dialect.

    Gemini takes an OpenAPI 3.0 Schema subset: uppercase type names, ``nullable`` in
    place of a union with null, no ``$defs``. Optional fields stay out of ``required``
    here — unlike strict mode, Gemini can express absence.
    """
    defs = raw.get("$defs", {})
    if not isinstance(defs, dict):  # pragma: no cover
        raise UnsupportedSchemaError("$defs is not an object")
    return _gemini_node(raw, defs, ())


def _gemini_node(node: Any, defs: dict[str, Any], seen: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise UnsupportedSchemaError(f"schema fragment is not an object: {node!r}")

    ref = node.get("$ref")
    if isinstance(ref, str):
        key = _def_key(ref)
        if key in seen:
            raise UnsupportedSchemaError(
                f"recursive model {key!r}: Gemini responseSchema does not support recursion"
            )
        target = defs.get(key)
        if target is None:
            raise UnsupportedSchemaError(f"dangling $ref {ref!r}")
        merged = _gemini_node(target, defs, (*seen, key))
        if "description" in node:
            merged["description"] = node["description"]
        return merged

    # Single-element `allOf` is pydantic's wrapper around an annotated $ref; unwrap it
    # for the same reason as in the strict converter.
    all_of = node.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) != 1:
            raise UnsupportedSchemaError(
                "Gemini responseSchema does not support `allOf` with more than one branch"
            )
        merged = _gemini_node(all_of[0], defs, seen)
        if "description" in node:
            merged["description"] = node["description"]
        return merged

    # `X | None` arrives as anyOf[T, null]; Gemini spells that `nullable: true` on T.
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        non_null = [b for b in any_of if not (isinstance(b, dict) and b.get("type") == "null")]
        if len(non_null) != len(any_of):
            if len(non_null) != 1:
                raise UnsupportedSchemaError(
                    "Gemini responseSchema cannot express a nullable union of several types"
                )
            converted = _gemini_node(non_null[0], defs, seen)
            converted["nullable"] = True
            if "description" in node:
                converted.setdefault("description", node["description"])
            return converted
        raise UnsupportedSchemaError("Gemini responseSchema does not support anyOf")

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _GEMINI_KEYWORDS:
            continue
        if key == "type":
            mapped = _GEMINI_TYPES.get(value) if isinstance(value, str) else None
            if mapped is None:
                raise UnsupportedSchemaError(f"Gemini responseSchema has no type {value!r}")
            out["type"] = mapped
        elif key == "format":
            if value in _GEMINI_FORMATS:
                out["format"] = value
        elif key == "properties":
            if not isinstance(value, dict):
                raise UnsupportedSchemaError("`properties` is not an object")
            out["properties"] = {k: _gemini_node(v, defs, seen) for k, v in value.items()}
        elif key == "items":
            out["items"] = _gemini_node(value, defs, seen)
        else:
            out[key] = value

    if out.get("type") == "OBJECT" and not out.get("properties"):
        raise UnsupportedSchemaError(
            "object with no declared properties (e.g. a bare `dict` field) cannot be "
            "expressed as a Gemini responseSchema"
        )
    return out


def _checked_for_ollama(raw: dict[str, Any]) -> dict[str, Any]:
    """Return ``raw`` unchanged, after refusing schemas Ollama's grammar cannot build.

    Ollama compiles ``format`` into a GBNF grammar. Recursion is the one shape that
    cannot be compiled into a finite grammar, so it is rejected here rather than at the
    server, where it surfaces as an opaque 500.
    """
    defs = raw.get("$defs")
    if isinstance(defs, dict):
        _assert_acyclic(raw, defs, ())
    return raw


def _assert_acyclic(node: Any, defs: dict[str, Any], seen: tuple[str, ...]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            key = _def_key(ref)
            if key in seen:
                raise UnsupportedSchemaError(
                    f"recursive model {key!r}: a recursive schema has no finite grammar"
                )
            target = defs.get(key)
            if target is None:
                raise UnsupportedSchemaError(f"dangling $ref {ref!r}")
            _assert_acyclic(target, defs, (*seen, key))
            return
        for key, value in node.items():
            if key != "$defs":
                _assert_acyclic(value, defs, seen)
    elif isinstance(node, list):
        for item in node:
            _assert_acyclic(item, defs, seen)


def _def_key(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def prompt_schema_guide(schema: type[BaseModel]) -> str:
    """The system-prompt instruction used when no native mode applies."""
    return (
        "Respond with ONLY valid JSON matching this JSON Schema. No prose, no code fences.\n"
        f"Schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"
    )
