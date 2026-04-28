"""
RAG (Retrieval-Augmented Generation) Pipeline
Handles document loading, embedding, storage, and retrieval
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import chromadb
from chromadb.config import Settings
import logging

# Suppress ChromaDB posthog telemetry errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import (
    CHROMADB_PATH, EMBEDDING_MODEL, CHUNK_SIZE,
    CHUNK_OVERLAP, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, KB_DIR
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Handles retrieval-augmented generation pipeline"""

    def __init__(self):
        """Initialize RAG pipeline with ChromaDB and embeddings"""
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.has_embedding_model = True
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.warning(f"Could not load embedding model '{EMBEDDING_MODEL}': {e}")
            logger.warning("Falling back to TF-IDF vectorizer for retrieval (offline-safe).")
            self.embedding_model = None
            self.has_embedding_model = False
            # NOTE: Do NOT fit this here — it must be fit on the full corpus in load_knowledge_base
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)

        self.using_chroma = False
        try:
            self.client = chromadb.PersistentClient(
                path=CHROMADB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name="colonial-sri-lanka-kb",
                metadata={"hnsw:space": "cosine"}
            )
            self.using_chroma = True
            logger.info(f"ChromaDB collection ready: {self.collection.name}")
        except Exception as e:
            logger.warning(f"Chroma init failed, using local fallback retriever: {e}")
            self.client = None
            self.collection = None
            self.fallback_docs: List[str] = []
            self.fallback_metadatas: List[Dict] = []
            if self.has_embedding_model:
                self.fallback_vectors: List[np.ndarray] = []
            self.fallback_matrix = None

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Chunk text into overlapping segments"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text to a dense vector.
        Uses SentenceTransformer if available, otherwise TF-IDF.
        TF-IDF vectorizer MUST have been fit on the corpus first via load_knowledge_base.
        """
        if self.has_embedding_model:
            return self.embedding_model.encode(text)
        else:
            # transform (not fit_transform) so the same vocabulary is used for
            # both documents and queries — this is what makes similarity work
            return self.vectorizer.transform([text]).toarray()[0]

    def load_knowledge_base(self, kb_dir: Path = KB_DIR, force_reload: bool = False) -> int:
        """Load documents from knowledge base directory"""
        if self.using_chroma and not force_reload and self.collection.count() > 0:
            logger.info(f"Knowledge base already loaded: {self.collection.count()} documents")
            return self.collection.count()
        if not self.using_chroma and not force_reload and len(getattr(self, 'fallback_docs', [])) > 0:
            logger.info(f"Knowledge base already loaded (fallback): {len(self.fallback_docs)} documents")
            return len(self.fallback_docs)

        kb_path = Path(kb_dir)
        if not kb_path.exists():
            logger.warning(f"Knowledge base directory not found: {kb_path}")
            return 0

        # ── Pass 1: collect all chunks and metadata ───────────────────────────
        all_chunks: List[str] = []
        all_metadatas: List[Dict] = []
        all_ids: List[str] = []

        def _harvest_file(file_path: Path):
            """Parse a .json or .txt file and append chunks to the above lists."""
            if file_path.suffix == ".json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                docs = data if isinstance(data, list) else [data]
                for doc_idx, doc in enumerate(docs):
                    doc_id = f"{file_path.stem}_{doc_idx}"
                    content = doc.get('content', doc.get('text', json.dumps(doc)))
                    for chunk_idx, chunk in enumerate(self.chunk_text(content)):
                        all_chunks.append(chunk)
                        all_ids.append(f"{doc_id}_chunk_{chunk_idx}")
                        all_metadatas.append({
                            "source": file_path.name,
                            "doc_id": doc_id,
                            "chunk_idx": chunk_idx,
                            "title": doc.get('title', 'Unknown'),
                            "period": doc.get('period', 'Unknown'),
                        })
            else:  # .txt
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                for chunk_idx, chunk in enumerate(self.chunk_text(content)):
                    all_chunks.append(chunk)
                    all_ids.append(f"{file_path.stem}_chunk_{chunk_idx}")
                    all_metadatas.append({"source": file_path.name, "chunk_idx": chunk_idx})

        for json_file in kb_path.glob("*.json"):
            try:
                _harvest_file(json_file)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        for txt_file in kb_path.glob("*.txt"):
            try:
                _harvest_file(txt_file)
            except Exception as e:
                logger.error(f"Error loading {txt_file}: {e}")

        if not all_chunks:
            logger.warning("No chunks collected — check KB directory contents.")
            return 0

        # ── Pass 2: fit TF-IDF on the FULL corpus (only needed when no SentenceTransformer) ──
        # This must happen before any call to _encode_text so that .transform() works
        # consistently for both documents and later queries.
        if not self.has_embedding_model:
            self.vectorizer.fit(all_chunks)
            logger.info("TF-IDF vectorizer fitted on full corpus.")

        # ── Pass 3: encode and store ──────────────────────────────────────────
        doc_count = 0

        if self.using_chroma:
            # Encode in one batch for SentenceTransformer (fast); TF-IDF also vectorised.
            try:
                if self.has_embedding_model:
                    embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=False)
                else:
                    embeddings = self.vectorizer.transform(all_chunks).toarray()

                # ChromaDB add() can handle batches
                BATCH = 500
                for start in range(0, len(all_chunks), BATCH):
                    end = start + BATCH
                    self.collection.add(
                        ids=all_ids[start:end],
                        embeddings=embeddings[start:end].tolist(),
                        documents=all_chunks[start:end],
                        metadatas=all_metadatas[start:end],
                    )
                    doc_count += end - start

                logger.info(f"✓ Loaded {doc_count} chunks into ChromaDB")
            except Exception as e:
                logger.error(f"Failed to add chunks to Chroma: {e}")
        else:
            # Fallback in-memory store
            self.fallback_docs = all_chunks
            self.fallback_metadatas = all_metadatas

            if self.has_embedding_model:
                vectors = self.embedding_model.encode(all_chunks, show_progress_bar=False)
                self.fallback_vectors = list(vectors)
                self.fallback_matrix = vectors  # shape (n, d)
            else:
                self.fallback_matrix = self.vectorizer.transform(all_chunks)

            doc_count = len(all_chunks)
            logger.info(f"✓ Loaded {doc_count} chunks into fallback store")

        return doc_count

    def retrieve(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Tuple[str, float, Dict]]:
        """Retrieve relevant documents for a query"""
        if not self.using_chroma:
            return self._fallback_retrieve(query, k, threshold)

        try:
            # FIX: encode the query ourselves and use query_embeddings.
            # Using query_texts requires ChromaDB to have a registered embedding
            # function; since we added documents with pre-computed embeddings,
            # there is none — so it would silently return no results.
            query_embedding = self._encode_text(query).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],   # ← was: query_texts=[query]
                n_results=min(k, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            retrieved = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    similarity = 1 - distance  # cosine distance → similarity
                    if similarity >= threshold:
                        retrieved.append((doc, similarity, metadata))

            logger.debug(f"Retrieved {len(retrieved)} documents for query: {query[:50]}...")
            return retrieved

        except Exception as e:
            logger.error(f"Error during Chroma retrieval: {e}")
            return self._fallback_retrieve(query, k, threshold)

    def format_context(self, retrieved_docs: List[Tuple[str, float, Dict]]) -> List[Dict[str, Any]]:
        """Format retrieved documents into structured context list for UI and prompts"""
        if not retrieved_docs:
            return []

        formatted_docs = []
        for i, (doc, score, metadata) in enumerate(retrieved_docs, 1):
            source = metadata.get('source', 'Unknown')
            title = metadata.get('title', '')
            period = metadata.get('period', '')

            source_info = f"[{source}]"
            if title:
                source_info += f" - {title}"
            if period:
                source_info += f" ({period})"

            formatted_docs.append({
                "id": i,
                "source_info": source_info,
                "score": score,
                "content": doc
            })

        return formatted_docs

    def add_document(self, content: str, doc_id: str, metadata: Optional[Dict] = None) -> bool:
        """Add a single document to the knowledge base"""
        try:
            chunks = self.chunk_text(content)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                chunk_metadata = (metadata.copy() if metadata else {})
                chunk_metadata["chunk_idx"] = chunk_idx

                embedding = self._encode_text(chunk)

                if self.using_chroma:
                    self.collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding.tolist()],
                        documents=[chunk],
                        metadatas=[chunk_metadata],
                    )
                else:
                    self.fallback_docs.append(chunk)
                    self.fallback_metadatas.append(chunk_metadata)
                    if self.has_embedding_model:
                        self.fallback_vectors.append(embedding)

            # Rebuild fallback matrix
            if not self.using_chroma:
                if self.has_embedding_model:
                    self.fallback_matrix = np.vstack(self.fallback_vectors)
                else:
                    self.fallback_matrix = self.vectorizer.transform(self.fallback_docs)

            logger.info(f"✓ Added document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False

    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base"""
        try:
            if self.using_chroma and self.collection is not None:
                ids_to_delete = self.collection.get()['ids']
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                logger.info(f"✓ Cleared {len(ids_to_delete)} documents from KB")
            else:
                self.fallback_docs = []
                self.fallback_metadatas = []
                if hasattr(self, 'fallback_vectors'):
                    self.fallback_vectors = []
                self.fallback_matrix = None
                logger.info("✓ Cleared fallback KB store")
            return True
        except Exception as e:
            logger.error(f"Error clearing KB: {e}")
            return False

    def _cosine_sim_dense(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if matrix is None or (hasattr(matrix, 'size') and matrix.size == 0):
            return np.array([])
        return cosine_similarity(query_vec.reshape(1, -1), matrix)[0]

    def _cosine_sim_sparse(self, query_vec, matrix) -> np.ndarray:
        return cosine_similarity(query_vec, matrix)[0]

    def _fallback_retrieve(self, query: str, k: int, threshold: float) -> List[Tuple[str, float, Dict]]:
        """Fallback retrieval using TF-IDF or dense embeddings."""
        if getattr(self, 'fallback_matrix', None) is None:
            return []

        if self.has_embedding_model:
            qvec = self.embedding_model.encode(query)
            sims = self._cosine_sim_dense(qvec, self.fallback_matrix)
        else:
            qvec = self.vectorizer.transform([query])
            sims = self._cosine_sim_sparse(qvec, self.fallback_matrix)

        idx_sorted = np.argsort(-sims)
        retrieved = []
        for idx in idx_sorted:
            if len(retrieved) >= k:
                break
            score = float(sims[idx])
            if score >= threshold:
                retrieved.append((self.fallback_docs[idx], score, self.fallback_metadatas[idx]))

        return retrieved


def initialize_rag(force_reload: bool = True) -> RAGPipeline:
    """Initialize and return RAG pipeline"""
    rag = RAGPipeline()
    if force_reload:
        rag.clear_knowledge_base()
    rag.load_knowledge_base(force_reload=force_reload)
    return rag