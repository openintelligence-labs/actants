# Errors

Every exception actants raises on purpose inherits from `ActantsError`, and all of
them are importable from the top level:

```python
from actants import ActantsError, ModelNotFoundError
```

`except ActantsError` is therefore a complete catch for "actants itself refused",
as distinct from a bug in your own code or an error escaping from a provider SDK.

## Hierarchy

```text
ActantsError
├── ProviderError
│   ├── UnknownProviderError        (also ValueError)
│   ├── ProviderNotInstalledError   (also ImportError)
│   ├── MissingAPIKeyError          (also ValueError)
│   ├── ModelNotFoundError          (also ValueError)
│   ├── ToolCallsNotSupportedError  (also TypeError)
│   └── AllProvidersFailedError     (also RuntimeError)
├── ToolError
├── CacheSchemaMismatch             (also RuntimeError)
└── MCPConnectionError              (also RuntimeError)
```

Each class also inherits the builtin exception you would naturally reach for, so
`except ValueError` still catches an unknown provider name. Catch the actants class
when you want to be precise, the builtin when you want to be permissive.

## Handling them

```python
from actants import LLM, ActantsError, MissingAPIKeyError, ModelNotFoundError

llm = LLM(provider="ollama", model="llama3.2")

try:
    result = await llm.complete("hello")
except ModelNotFoundError as exc:
    # names the models the server does have, and the `ollama pull` to run
    print(exc)
except MissingAPIKeyError as exc:
    # names the env var to set
    print(exc)
except ActantsError as exc:
    print(f"actants refused: {exc}")
```

Every message names the exact problem and the exact fix — if you can hit it in your
first ten minutes, it tells you what to type next.

## Falling back across providers

`AllProvidersFailedError` carries the individual failures, so you can see why each
link in the chain gave up rather than only that the chain did:

```python
from actants import AllProvidersFailedError, FallbackProvider, LLM, OllamaProvider

chain = FallbackProvider([(OllamaProvider(), "llama3.2")])

try:
    await LLM(provider=chain).complete("hello")
except AllProvidersFailedError as exc:
    for name, err in exc.errors:
        print(f"{name} failed: {err!r}")
```

## Compatibility

The classes previously lived in `actants.llm.errors`. That path still works and
returns the identical class objects, so existing imports and `except` clauses are
unaffected; new code should prefer the top-level import.

## Reference

::: actants.errors.ActantsError

::: actants.errors.ProviderError

::: actants.errors.UnknownProviderError

::: actants.errors.ProviderNotInstalledError

::: actants.errors.MissingAPIKeyError

::: actants.errors.ModelNotFoundError

::: actants.errors.ToolCallsNotSupportedError

::: actants.tools.base.ToolError

::: actants.policies.fallback.AllProvidersFailedError

::: actants.cache.semantic.CacheSchemaMismatch
