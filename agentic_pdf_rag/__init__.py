import os

from . import config_manager
from .agentic_pdf_parser import AgenticPDFParser
from .db_handler import DBHandler
from .openai_client import AzureOpenAIChatClient, AzureOpenAIEmbeddings
from .pdf_chunker import PDFChunker
from .rag import RetrievalEngine, GenerationEngine

__all__ = [
    "AgenticPDFParser",
    "PDFChunker",
    "DBHandler",
    "RetrievalEngine",
    "GenerationEngine",
    "RAGPipeline"
]

class RAGPipeline:
    def __init__(self, config_file=None, openai_api_key=None, openai_embeddings_api_key=None):
        if config_file is None:
            config_file = os.path.join(os.getcwd(), "config", "config.ini")
        self.config = config_manager.Config(config_file=config_file, openai_api_key=openai_api_key, openai_embeddings_api_key=openai_embeddings_api_key)
        config_manager.config = self.config

    def get_agentic_pdf_parser(self):
        return AgenticPDFParser(
                 model=self.config.agentic_pdf_parser_model,
                 openai_endpoint=self.config.openai_endpoint,
                 openai_api_key=self.config.openai_api_key,
                 openai_api_version=self.config.openai_api_version,
                 output_directory=self.config.output_directory
        )

    def get_pdf_chunker(self,
                        agentic_pdf_parser=None,
                        custom_embedding_model=None,
                        agentic_chunker_context="",
                        buffer_size=1,
                        breakpoint_threshold_type="percentile",
                        number_of_chunks=None,
                        sentence_split_regex=r"(?<=[.?!])\s+",
                        min_chunk_size=None,
                        ):
        return PDFChunker(agentic_pdf_parser=agentic_pdf_parser,
                          embedding_model=self.config.openai_embedding_model,
                          custom_embedding_model=custom_embedding_model,
                          openai_embeddings_endpoint = self.config.openai_embeddings_endpoint,
                          openai_embeddings_api_key = self.config.openai_embeddings_api_key,
                          openai_embeddings_api_version = self.config.openai_embeddings_api_version,
                          semantic_chunker_buffer_size = buffer_size,
                          semantic_chunker_breakpoint_threshold_type = breakpoint_threshold_type,
                          semantic_chunker_sentence_split_regex=sentence_split_regex,
                          semantic_chunker_min_chunk_size=min_chunk_size,
                          semantic_chunker_number_of_chunks=number_of_chunks,
                          agentic_chunker_context = agentic_chunker_context
                          )

    def get_db_handler(self):
        return DBHandler(
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            host=self.config.db_host,
            port=self.config.db_port,
        )

    def get_retrieval_engine(self, db_handler=None):
        if db_handler is None:
            db_handler = self.get_db_handler()
        return RetrievalEngine(db_handler=db_handler)

    def get_generation_engine(self, llm_client=None):
        if llm_client is None:
            llm_client = self.get_openai_client()
        return GenerationEngine(llm_client=llm_client)

    def get_openai_client(self):
        return AzureOpenAIChatClient(
            model=self.config.agentic_pdf_parser_model,
            api_key=self.config.openai_api_key,
            api_endpoint=self.config.openai_endpoint,
            api_version=self.config.openai_api_version,
        )

    def get_openai_embeddings_client(self):
        return AzureOpenAIEmbeddings(
            endpoint=self.config.openai_embeddings_endpoint,
            api_key=self.config.openai_embeddings_api_key,
            model=self.config.openai_embedding_model,
            api_version=self.config.openai_embeddings_api_version
        )
