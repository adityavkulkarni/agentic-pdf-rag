import logging

from .. import config_manager
from ..clients import AzureOpenAIChatClient
from ..database import PostgreSQLVectorClient
from ..embedding_models import Qwen3RetrievalSystem

logger = logging.getLogger(__name__)


class DBHandler(PostgreSQLVectorClient):
    def __init__(self,
                 dbname,
                 user,
                 password,
                 host,
                 port,
                 endpoint=None,
                 api_key=None,
                 model=None,
                 api_version=None,
                 create_tables=False,
                 use_qwen3=True
                 ):
        super().__init__(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        if use_qwen3:
            self.document_table = "documents_qwen3"
            self.page_table = "pages_qwen3"
            self.agentic_embedding_table = "agentic_embedding_qwen3"
            self.semantic_embedding_table = "semantic_embedding_qwen3"
            self.docling_agentic_embedding_table = "docling_agentic_embedding_qwen3"
            self.docling_semantic_embedding_table = "docling_semantic_embedding_qwen3"

            dim = 1024
        else:
            self.document_table = "documents"
            self.page_table = "pages"
            self.agentic_embedding_table = "agentic_embedding"
            self.semantic_embedding_table = "semantic_embedding"
            self.docling_agentic_embedding_table = "docling_agentic_embedding"
            self.docling_semantic_embedding_table = "docling_semantic_embedding"
            dim = 3072
        if create_tables:
            self._create_document_table(table_name=self.document_table, vector_dim=dim)
            self._create_document_table(table_name=self.page_table, vector_dim=dim)
            self._create_embedding_table(table_name=self.agentic_embedding_table, vector_dim=dim)
            self._create_embedding_table(table_name=self.semantic_embedding_table, vector_dim=dim)
            self._create_embedding_table(table_name=self.docling_agentic_embedding_table, vector_dim=dim)
            self._create_embedding_table(table_name=self.docling_semantic_embedding_table, vector_dim=dim)
        if use_qwen3:
            self.embedding_client = Qwen3RetrievalSystem()
        else:
            self.embedding_client = AzureOpenAIChatClient(
                api_endpoint=endpoint or config_manager.config.openai_embedding_endpoint,
                api_key=api_key or config_manager.config.openai_embedding_api_key,
                api_version=api_version or config_manager.config.openai_embedding_api_version,
                model=model or config_manager.config.openai_embedding_model
            )

    def insert_document(self, document, overwrite=False):
        if overwrite:
            logger.info(f"Document {document['file_name']} already exists! Reinsert the chunks.")
            self._delete_document(table_name=self.document_table, file_name=document["file_name"])
            self._delete_document(table_name=self.page_table, file_name=document["file_name"])
            self._delete_document(table_name=self.agentic_embedding_table, file_name=document["file_name"])
            self._delete_document(table_name=self.semantic_embedding_table, file_name=document["file_name"])
        self._insert_document(
            table_name=self.document_table,
            filename=document["file_name"],
            data=document,
            summary_embedding=self.embedding_client.create_embedding_dict([document["parsed_pdf"]["summary"]])[
                document["parsed_pdf"]["summary"]],
        )
        for i, page in document["parsed_pdf"]["pages_descriptions"].items():
            self._insert_document(
                table_name=self.page_table,
                filename=f"{document["file_name"].split(".")[0]}_page{i}",
                data=page,
                summary_embedding=self.embedding_client.create_embedding_dict([page["summary"]])[page["summary"]]
            )

    def batch_insert_embeddings(self, agentic_chunks, semantic_chunks, docling=False):
        # if type(agentic_chunks.get("embedding")) is dict:
        if docling:
            agentic_embedding_table = self.docling_agentic_embedding_table
            semantic_embedding_table = self.docling_semantic_embedding_table
        else:
            agentic_embedding_table = self.agentic_embedding_table
            semantic_embedding_table = self.semantic_embedding_table
        records = []
        for chunk in agentic_chunks:
            for embedding in chunk["embedding"].values():
                records.append({
                "content": chunk["metadata"]["agentic_chunk"],
                "embedding": embedding,
                "filename": chunk["filename"],
                "summary_embedding": chunk["summary_embedding"],
                "metadata": chunk["metadata"]
            })
        self._batch_insert_embeddings(
            table_name=agentic_embedding_table,
            records=records
        )
        # if type(semantic_chunks.get("embedding")) is dict:
        records = []
        for chunk in semantic_chunks:
            for embedding in chunk["embedding"].values():
                records.append({
                "content": chunk["content"],
                "embedding": embedding,
                "filename": chunk["filename"],
                "summary_embedding": chunk["summary_embedding"],
                "metadata": chunk["metadata"]
            })
        self._batch_insert_embeddings(
            table_name=semantic_embedding_table,
            records=records
        )

    def get_documents(self, filename=None):
        if filename:
            return self._fetch_document(
                table_name=self.document_table,
                file_name=filename
            )
        else:
            return self._fetch_documents(
                table_name=self.document_table,
            )

    def get_pages(self, filename=None):
        if filename:
            pages =  self._fetch_document(
                table_name=self.page_table,
                file_name=filename
            )
        else:
            pages =  self._fetch_documents(
                table_name=self.page_table,
            )
        return [
            {
                "filename": f"{page[0].split("_page")[0]}.pdf",
                "page_no": int(page[0].split("_page")[1]),
                "title": page[1]["title"],
                "summary": page[1]["summary"],
                "text_content": page[1]["text_content"],
                "visual_elements": page[1]["visual_elements"]
            }
            for page in pages
        ]
