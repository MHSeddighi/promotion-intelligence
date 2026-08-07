"""Unit + integration tests for the product embedding / similarity service."""

import numpy as np
import pytest

from app.models.similarity import ProductSimilarityService

SIMILARITY = ProductSimilarityService()


def test_cosine_similarity_identity_and_orthogonal():
    assert ProductSimilarityService.cosine_similarity(
        np.array([1.0, 0.0]), np.array([1.0, 0.0])
    ) == pytest.approx(1.0)
    assert ProductSimilarityService.cosine_similarity(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(0.0)
    assert ProductSimilarityService.cosine_similarity(
        np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    ) == pytest.approx(-1.0)
    assert ProductSimilarityService.cosine_similarity(
        np.array([0.0, 0.0]), np.array([1.0, 0.0])
    ) == 0.0


def test_cosine_similarity_dimension_mismatch():
    with pytest.raises(ValueError):
        ProductSimilarityService.cosine_similarity(np.array([1.0]), np.array([1.0, 2.0]))


def test_similar_products_high_similarity():
    similar = SIMILARITY.find_similar_products(1005637, top_k=5)
    assert len(similar) == 5
    assert similar[0]["similarity_score"] >= similar[-1]["similarity_score"]
    assert similar[0]["similarity_score"] > 0.5


def test_unrelated_products_lower_similarity():
    # Ice (25671) and bread (26093) are in different departments: the embedding
    # cosine must be far below the perfect self-match of 1.0.
    unrelated = SIMILARITY.pair_similarity(25671, 26093)
    self_match = SIMILARITY.pair_similarity(26093, 26093)
    assert unrelated["cosine_similarity"] < 0.5
    assert unrelated["cosine_similarity"] < self_match["cosine_similarity"]
    # Top matches are always at least as similar as lower-ranked matches.
    similar = SIMILARITY.find_similar_products(1005637, top_k=10)
    all_scores = [row["similarity_score"] for row in similar]
    assert all_scores == sorted(all_scores, reverse=True)


def test_pair_similarity_same_product_is_one():
    result = SIMILARITY.pair_similarity(1005637, 1005637)
    assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)


def test_unknown_product_raises():
    with pytest.raises(KeyError):
        SIMILARITY.find_similar_products(99999999)
    with pytest.raises(KeyError):
        SIMILARITY.embedding_vector(99999999)
    with pytest.raises(KeyError):
        SIMILARITY.find_substitutes(99999999)


def test_substitutes_present():
    substitutes = SIMILARITY.find_substitutes(1005637, top_k=3)
    assert len(substitutes) <= 3
    for item in substitutes:
        assert "product_id" in item
        assert "similarity" in item


def test_invalid_top_k():
    with pytest.raises(ValueError):
        SIMILARITY.find_similar_products(1005637, top_k=0)
    with pytest.raises(ValueError):
        SIMILARITY.find_similar_products(1005637, top_k=501)


def test_health_summary():
    health = SIMILARITY.health()
    assert health["n_products_embedded"] > 1000
    assert health["embedding_dim"] > 100
