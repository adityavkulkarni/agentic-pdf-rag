import logging

from typing import List
from langchain_core.embeddings import Embeddings
from .openai_client import AzureOpenAIChatClient

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
