# Embeddings

Local-first vector embeddings, defaulting to `nomic-embed-text` via Ollama
(768-dim, fast, no API key).

## Default

```python
from actants import Embeddings

emb = Embeddings()  # Ollama + nomic-embed-text
result = await emb.embed(["hello", "world"])
print(result.dimensions, len(result.vectors))  # 768 2
```

## One vector at a time

```python
vec = await emb.embed_one("the quick brown fox")
```

## Cosine similarity helper

```python
score = Embeddings.cosine(result.vectors[0], result.vectors[1])
```

Returns 0 for empty vectors, mismatched lengths, or zero-norm inputs — never
raises, so it composes safely inside batch loops.

## Switching models

```python
emb = Embeddings(model="snowflake-arctic-embed")  # any Ollama embedding model
```

## Custom provider

Subclass `BaseEmbeddingProvider`:

```python
from actants.embeddings import BaseEmbeddingProvider, EmbeddingResult, Embeddings


class MyProvider(BaseEmbeddingProvider):
    name = "mine"

    async def embed(
        self, texts, *, model: str
    ) -> EmbeddingResult: ...  # fetch / compute and return EmbeddingResult

    async def health(self) -> bool:
        return True


emb = Embeddings(provider=MyProvider())
```

## Pairing with storage

Persist vectors via `open_sqlite` (with `sqlite-vec` for ANN). See
[Storage](storage.md) and the [semantic-search cookbook](../cookbook/embeddings.md).
