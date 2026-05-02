from __future__ import annotations

import pytest

from agentic_kit.embeddings import Embeddings
from agentic_kit.testing import FakeEmbeddingProvider


@pytest.mark.asyncio
async def test_embeddings_with_fake_provider():
    fake = FakeEmbeddingProvider(dimensions=8)
    emb = Embeddings(provider=fake)

    result = await emb.embed(["hello", "world"])
    assert result.dimensions == 8
    assert len(result.vectors) == 2
    assert all(len(v) == 8 for v in result.vectors)
    assert fake.calls == [["hello", "world"]]


@pytest.mark.asyncio
async def test_embed_one_returns_single_vector():
    fake = FakeEmbeddingProvider(dimensions=4)
    emb = Embeddings(provider=fake)
    vec = await emb.embed_one("hello")
    assert len(vec) == 4


def test_cosine_similarity_basic():
    assert Embeddings.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert Embeddings.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert Embeddings.cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_handles_empty_or_mismatched():
    assert Embeddings.cosine([], [1.0]) == 0.0
    assert Embeddings.cosine([1.0, 2.0], [1.0]) == 0.0
    assert Embeddings.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


@pytest.mark.asyncio
async def test_fake_provider_is_deterministic():
    fake = FakeEmbeddingProvider(dimensions=8)
    a1 = await fake.embed(["hello"])
    a2 = await fake.embed(["hello"])
    assert a1.vectors == a2.vectors
