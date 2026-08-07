"""Product embedding and similarity intelligence.

Loads the artifacts produced by ``notebook/03_product_embeddings.ipynb``:

- ``product_embeddings.parquet``  -> dense per-product vectors (dim_0..dim_N)
- ``product_similarity.parquet``  -> precomputed pairwise similarity rows
- ``top_k_substitutes.parquet``   -> top substitute products per product

and exposes similarity search, substitution lookup and on-the-fly cosine
similarity for arbitrary product pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import PROJECT_ROOT

EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "product_embeddings"


class ProductSimilarityService:
    """Read-only product embedding / similarity service."""

    def __init__(self, artifact_dir: Path = EMBEDDINGS_DIR) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self._embeddings: pd.DataFrame | None = None
        self._similarity: pd.DataFrame | None = None
        self._substitutes: pd.DataFrame | None = None

    @property
    def embeddings(self) -> pd.DataFrame:
        if self._embeddings is None:
            self._embeddings = pd.read_parquet(
                self.artifact_dir / "product_embeddings.parquet"
            )
        return self._embeddings

    @property
    def similarity(self) -> pd.DataFrame:
        if self._similarity is None:
            self._similarity = pd.read_parquet(
                self.artifact_dir / "product_similarity.parquet"
            )
        return self._similarity

    @property
    def substitutes(self) -> pd.DataFrame:
        if self._substitutes is None:
            self._substitutes = pd.read_parquet(
                self.artifact_dir / "top_k_substitutes.parquet"
            )
        return self._substitutes

    @property
    def vector_columns(self) -> list[str]:
        return [column for column in self.embeddings.columns if column.startswith("dim_")]

    def embedding_vector(self, product_id: int) -> np.ndarray:
        if int(product_id) not in set(self.embeddings["PRODUCT_ID"]):
            raise KeyError(f"Product {product_id} has no embedding.")
        row = self.embeddings.loc[
            self.embeddings["PRODUCT_ID"] == int(product_id), self.vector_columns
        ].iloc[0]
        return row.to_numpy(dtype="float64")

    @staticmethod
    def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        a = np.asarray(vector_a, dtype="float64")
        b = np.asarray(vector_b, dtype="float64")
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same dimensionality.")
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_similar_products(
        self,
        product_id: int,
        top_k: int = 10,
        min_similarity: float = 0.0,
        relationship_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most similar products to ``product_id``.

        Uses the precomputed similarity table (fast) and enriches with the raw
        embedding cosine when available.
        """
        if int(product_id) not in set(self.similarity["PRODUCT_ID"]):
            raise KeyError(f"Product {product_id} has no similarity data.")
        if not 1 <= int(top_k) <= 500:
            raise ValueError("top_k must be between 1 and 500.")

        frame = self.similarity[self.similarity["PRODUCT_ID"] == int(product_id)].copy()
        if relationship_type:
            frame = frame[frame["relationship_type"] == relationship_type]
        frame = frame[frame["similarity_score"] >= min_similarity]
        frame = frame.sort_values("similarity_score", ascending=False).head(int(top_k))
        return [
            {
                "product_id": int(row["SIMILAR_PRODUCT_ID"]),
                "similarity_score": round(float(row["similarity_score"]), 6),
                "relationship_type": str(row["relationship_type"]),
                "embedding_cosine": round(float(row["embedding_cosine"]), 6),
                "metadata_similarity": round(float(row["metadata_similarity"]), 6),
                "brand_match": bool(row["brand_match"]),
                "commodity_match": bool(row["commodity_match"]),
                "sub_commodity_match": bool(row["sub_commodity_match"]),
                "department_match": bool(row["department_match"]),
                "basket_jaccard": round(float(row["basket_jaccard"]), 6),
                "household_jaccard": round(float(row["household_jaccard"]), 6),
            }
            for row in frame.to_dict(orient="records")
        ]

    def find_substitutes(
        self, product_id: int, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Return the top substitute products (cannibalization candidates)."""
        if int(product_id) not in set(self.substitutes["PRODUCT_ID"]):
            raise KeyError(f"Product {product_id} has no substitute data.")
        row = self.substitutes[
            self.substitutes["PRODUCT_ID"] == int(product_id)
        ].iloc[0]
        top_k = min(int(top_k), 10)
        results: list[dict[str, Any]] = []
        for index in range(1, top_k + 1):
            candidate = row.get(f"substitute_{index}")
            if pd.isna(candidate):
                continue
            results.append(
                {
                    "product_id": int(candidate),
                    "similarity": round(float(row.get(f"substitute_similarity_{index}", 0.0)), 6),
                    "substitute_score": round(float(row.get(f"substitute_score_{index}", 0.0)), 6),
                }
            )
        return results

    def pair_similarity(self, product_a: int, product_b: int) -> dict[str, Any]:
        """Cosine similarity between two products' embedding vectors."""
        vector_a = self.embedding_vector(product_a)
        vector_b = self.embedding_vector(product_b)
        return {
            "product_a": int(product_a),
            "product_b": int(product_b),
            "cosine_similarity": round(
                self.cosine_similarity(vector_a, vector_b), 6
            ),
        }

    def product_metadata(self, product_id: int) -> dict[str, Any]:
        """Embedding metadata summary for one product."""
        vector = self.embedding_vector(product_id)
        return {
            "product_id": int(product_id),
            "embedding_dim": int(len(vector)),
            "norm": round(float(np.linalg.norm(vector)), 6),
        }

    def health(self) -> dict[str, Any]:
        return {
            "n_products_embedded": int(self.embeddings["PRODUCT_ID"].nunique()),
            "n_similarity_rows": int(len(self.similarity)),
            "n_substitute_rows": int(len(self.substitutes)),
            "embedding_dim": len(self.vector_columns),
            "artifact_dir": str(self.artifact_dir),
        }
