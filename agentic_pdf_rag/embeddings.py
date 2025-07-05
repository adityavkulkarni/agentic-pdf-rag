import logging

from typing import List
from langchain_core.embeddings import Embeddings
from .openai_client import AzureOpenAIChatClient
from .qwen_embedding import Qwen3RetrievalSystem

logger = logging.getLogger(__name__)


class OpenAIEmbeddings(Embeddings):
    def __init__(self, azure_openai_embeddings: AzureOpenAIChatClient):
        self.azure_openai_embeddings = azure_openai_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = self.azure_openai_embeddings.create_embeddings(
            texts
        )
        return [entry.embedding for entry in results.data]

    def embed_query(self, text: str) -> List[float]:
        result = self.azure_openai_embeddings.create_embeddings(
            [text],
        )[0]
        return result.data.embedding

class Qwen3Embeddings(Embeddings):
    def __init__(self, retrieval_system: Qwen3RetrievalSystem=None):
        self.retrieval_system = retrieval_system or Qwen3RetrievalSystem()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using 'document' prompt context"""
        results = self.retrieval_system.create_embeddings(
            texts,
            prompt_name="document"
        )
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed queries using specialized query prompt"""
        result = self.retrieval_system.create_embeddings(
            [text],
            prompt_name="query"
        )[0]
        return result
