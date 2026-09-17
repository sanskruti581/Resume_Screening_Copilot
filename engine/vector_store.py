"""
engine/vector_store.py
------------------------
In-memory ChromaDB vector store for resume chunks, embedded with the
'all-MiniLM-L6-v2' sentence-transformers model. Provides cosine-similarity
based retrieval of the resume chunks most relevant to a given job description.
"""

from __future__ import annotations

from typing import List, Tuple
from uuid import uuid4

import numpy as np
import chromadb
from chromadb.utils import embedding_functions


class VectorStoreError(Exception):
    pass


class ResumeVectorStore:
    """
    Wraps a single in-memory ChromaDB collection holding chunks from every
    uploaded resume, tagged by resume_id metadata so we can query per-resume.
    """

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    COLLECTION_NAME_PREFIX = "resumes"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.EMBEDDING_MODEL_NAME

        try:
            # Modern chromadb versions expose a dedicated in-memory client.
            self.client = chromadb.EphemeralClient()
        except AttributeError:
            self.client = chromadb.Client()

        try:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to load embedding model '{self.model_name}': {e}"
            )

        self.collection = None
        self._init_collection()

    def _init_collection(self):
        collection_name = f"{self.COLLECTION_NAME_PREFIX}-{uuid4().hex[:12]}"
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        """Clear all stored resume chunks."""
        self._init_collection()

    def add_resume_chunks(self, resume_id: str, chunks: List[str], filename: str) -> None:
        """Embed and store all chunks for a single resume."""
        if not chunks:
            return
        ids = [f"{resume_id}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"resume_id": resume_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        try:
            self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        except Exception as e:
            raise VectorStoreError(f"Failed to index chunks for '{filename}': {e}")

    def query_top_chunks(
        self, jd_text: str, resume_id: str, n_results: int = 5
    ) -> Tuple[List[str], float]:
        """
        Retrieve the top-matching chunks for a specific resume relative to the
        job description, and compute an average cosine-similarity based
        vector match score (0-100).
        """
        if not jd_text or not jd_text.strip():
            return [], 0.0

        try:
            results = self.collection.query(
                query_texts=[jd_text],
                n_results=n_results,
                where={"resume_id": resume_id},
            )
        except Exception as e:
            raise VectorStoreError(f"Vector query failed: {e}")

        documents = results.get("documents", [[]])
        distances = results.get("distances", [[]])

        docs = documents[0] if documents else []
        dists = distances[0] if distances else []

        if not dists:
            return docs, 0.0

        # ChromaDB cosine distance = 1 - cosine_similarity
        similarities = [max(0.0, 1.0 - d) for d in dists]
        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
        vector_score = max(0.0, min(100.0, avg_similarity * 100.0))

        return docs, vector_score
