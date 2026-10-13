
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddingAdapter(Embeddings):
    """Connect SentenceTransformer to LangChain's embedding interface."""

    def __init__(self, model_name=EMBEDDING_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def __call__(self, text: str) -> list[float]:
        """Compatibility with code expecting a callable embedding."""
        return self.embed_query(text)


def load_embeddings():
    return LocalEmbeddingAdapter()
