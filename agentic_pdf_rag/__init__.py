import os

from . import config_manager
from .agentic_pdf_parser import AgenticPDFParser
from .db_handler import DBHandler
from .openai_client import AzureOpenAIChatClient
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
    def __init__(self,
                 config_file=None,
                 openai_api_key=None,
                 openai_embeddings_api_key=None,
                 init=True,
                 agentic_chunker_context="",
                 buffer_size=1,
                 breakpoint_threshold_type="percentile",
                 number_of_chunks=None,
                 sentence_split_regex=r"(?<=[.?!])\s+",
                 min_chunk_size=None,
                 ):
        if config_file is None:
            config_file = os.path.join(os.getcwd(), "config", "config.ini")
        self.config = config_manager.Config(config_file=config_file, openai_api_key=openai_api_key,
                                            openai_embeddings_api_key=openai_embeddings_api_key)
        config_manager.config = self.config
        self.use_qwen3 = self.config.use_qwen3
        self.pdf_parser = None
        self.pdf_chunker = None
        self.db_handler = None
        self.openai_client = None
        self.openai_embeddings_client = None
        self.retrieval_engine = None
        self.generator = None

        if init:
            self.pdf_parser = self.get_agentic_pdf_parser()
            self.pdf_chunker = self.get_pdf_chunker(agentic_pdf_parser=None,
                                     agentic_chunker_context=agentic_chunker_context,
                                     buffer_size=buffer_size,
                                     breakpoint_threshold_type=breakpoint_threshold_type,
                                     number_of_chunks=number_of_chunks,
                                     sentence_split_regex=sentence_split_regex,
                                     min_chunk_size=min_chunk_size)
            self.db_handler = self.get_db_handler(create_tables=True)
            self.openai_client = self.get_openai_client()
            self.openai_embeddings_client = self.get_openai_embeddings_client()
            self.retrieval_engine = self.get_retrieval_engine(db_handler=self.db_handler)
            self.generator = self.get_generation_engine(llm_client=self.openai_client)

    def get_agentic_pdf_parser(self):
        return self.pdf_parser if self.pdf_parser is not None else AgenticPDFParser(
            model=self.config.agentic_pdf_parser_model,
            openai_endpoint=self.config.openai_endpoint,
            openai_api_key=self.config.openai_api_key,
            openai_api_version=self.config.openai_api_version,
            output_directory=self.config.output_directory,
            docling_url=self.config.docling_url,
        )

    def get_pdf_chunker(self,
                        agentic_pdf_parser=None,
                        agentic_chunker_context="",
                        buffer_size=1,
                        breakpoint_threshold_type="percentile",
                        number_of_chunks=None,
                        sentence_split_regex=r"(?<=[.?!])\s+",
                        min_chunk_size=None,
                        ):
        if self.pdf_chunker:
            self.pdf_chunker.add_parsed_pdf(agentic_pdf_parser=agentic_pdf_parser, agentic_chunker_context=agentic_chunker_context,)
        return self.pdf_chunker if self.pdf_chunker else PDFChunker(agentic_pdf_parser=agentic_pdf_parser,
                          openai_embedding_model=self.config.openai_embedding_model,
                          use_qwen3=self.use_qwen3,
                          openai_embeddings_endpoint=self.config.openai_embedding_endpoint,
                          openai_embeddings_api_key=self.config.openai_embedding_api_key,
                          openai_embeddings_api_version=self.config.openai_embedding_api_version,
                          semantic_chunker_buffer_size=buffer_size,
                          semantic_chunker_breakpoint_threshold_type=breakpoint_threshold_type,
                          semantic_chunker_sentence_split_regex=sentence_split_regex,
                          semantic_chunker_min_chunk_size=min_chunk_size,
                          semantic_chunker_number_of_chunks=number_of_chunks,
                          agentic_chunker_context=agentic_chunker_context
                          )

    def get_db_handler(self, create_tables=False):
        return self.db_handler if self.db_handler else DBHandler(
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            host=self.config.db_host,
            port=self.config.db_port,
            create_tables=create_tables,
            use_qwen3=self.use_qwen3,
        )

    def get_retrieval_engine(self, db_handler=None):
        if db_handler is None:
            db_handler = self.get_db_handler()
        return self.retrieval_engine if self.retrieval_engine else RetrievalEngine(db_handler=db_handler,   use_qwen3=self.use_qwen3)

    def get_generation_engine(self, llm_client=None):
        if llm_client is None:
            llm_client = self.get_openai_client()
        return self.generator if self.generator else GenerationEngine(llm_client=llm_client)

    def get_openai_client(self):
        return self.openai_client if self.openai_client else AzureOpenAIChatClient(
            model=self.config.agentic_pdf_parser_model,
            api_key=self.config.openai_api_key,
            api_endpoint=self.config.openai_endpoint,
            api_version=self.config.openai_api_version,
        )

    def get_openai_embeddings_client(self):
        return self.openai_embeddings_client if self.openai_embeddings_client else AzureOpenAIChatClient(
            api_endpoint=self.config.openai_embedding_endpoint,
            api_key=self.config.openai_embedding_api_key,
            model=self.config.openai_embedding_model,
            api_version=self.config.openai_embedding_api_version
        )

    def parse_pdf(self, pdf_path, filename=None, custom_metadata=None):
        self.parsed_pdf = self.pdf_parser.run_pipline(pdf_file=pdf_path, file_name=filename,
                                                      custom_metadata=custom_metadata)
        return self.parsed_pdf

    def create_chunks(self, agentic_pdf_parser=None, agentic_chunker_context="", pdf_path=None, filename=None):
        if agentic_pdf_parser is None and self.parsed_pdf is None:
            self.parse_pdf(pdf_path, filename)
        self.chunks = self.pdf_chunker.run_pipline(agentic_pdf_parser=self.pdf_parser,
                                                   agentic_chunker_context=agentic_chunker_context)
        return self.chunks

    def add_document_to_db(self, parsed_pdf=None, chunks=None):
        parsed_pdf = parsed_pdf or self.parsed_pdf
        chunks = chunks or self.chunks
        self.db_handler.insert_document(parsed_pdf)
        self.db_handler.batch_insert_embeddings(
            agentic_chunks=chunks["agentic_chunks"],
            semantic_chunks=chunks["semantic_chunks"]
        )

    def add_document_to_knowledge(self,
                                  pdf_path,
                                  filename=None,
                                  custom_metadata=None,
                                  agentic_chunker_context="",
                                  ):
        self.parse_pdf(pdf_path=pdf_path, filename=filename, custom_metadata=custom_metadata)
        self.create_chunks(agentic_pdf_parser=self.pdf_parser, agentic_chunker_context=agentic_chunker_context)
        self.add_document_to_db(parsed_pdf=self.parsed_pdf, chunks=self.chunks)

    def retrieve_context(self, query):
        context = self.retrieval_engine.get_context(query=query)
        return context

    def generate_response(self, query, context="", additional_instructions=""):
        response = self.generator.generate_response(
            query=query,
            context=context,
            additional_instructions=additional_instructions
        )
        return response

    def get_final_response(self, query, additional_instructions=""):
        context = self.retrieve_context(query)
        response = self.generate_response(query, context, additional_instructions)
        return response
