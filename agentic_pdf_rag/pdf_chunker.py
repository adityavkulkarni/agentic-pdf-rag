import logging

from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config_manager import config
from .models import PDFChunkerResults
from .agentic_pdf_parser import AgenticPDFParser
from .agentic_chunker import AgenticChunker
from .embeddings import OpenAIEmbeddings, Qwen3Embeddings
from .openai_client import AzureOpenAIChatClient
from .qwen_embedding import Qwen3RetrievalSystem

logger = logging.getLogger(__name__)


class PDFChunker:
    def __init__(self,
                 agentic_pdf_parser: AgenticPDFParser=None,
                 use_qwen3=False,
                 openai_embedding_model="text-embedding-3-large",
                 openai_embeddings_endpoint = None,
                 openai_embeddings_api_key = None,
                 openai_embeddings_api_version = None,
                 semantic_chunker_buffer_size = 1,
                 semantic_chunker_breakpoint_threshold_type = "percentile",
                 semantic_chunker_sentence_split_regex=r"(?<=[.?!])\s+",
                 semantic_chunker_min_chunk_size=128,
                 semantic_chunker_number_of_chunks=10,
                 agentic_chunker_context = "",
                 docling=False,
                 ):
        openai_embeddings_api_version = openai_embeddings_api_version or config.openai_embedding_api_version
        openai_embeddings_endpoint = openai_embeddings_endpoint or config.openai_embedding_endpoint
        openai_embeddings_api_key = openai_embeddings_api_key or config.openai_embedding_api_key
        openai_embedding_model = openai_embedding_model or config.openai_embedding_model

        self.results = PDFChunkerResults()
        self.agentic_pdf_parser = None
        self.llm_client = None
        self.agentic_chunker = None
        self.metadata = None
        self.sentences = []
        self.sentences_docling = []

        if use_qwen3:
            self.embedding_client = Qwen3RetrievalSystem()
            self.embeddings = Qwen3Embeddings(retrieval_system=self.embedding_client)
        else:
            self.embedding_client = AzureOpenAIChatClient(
                api_endpoint=openai_embeddings_endpoint,
                api_key=openai_embeddings_api_key,
                api_version=openai_embeddings_api_version,
                model=openai_embedding_model
            )
            self.embeddings = OpenAIEmbeddings(self.embedding_client)

        if agentic_pdf_parser:
            self.add_parsed_pdf(agentic_pdf_parser, agentic_chunker_context, docling=docling)
        else:
            logger.info("No agentic_pdf_parser defined. Add parsed PDF object with: run_pipeline(agentic_pdf_parser)")
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            buffer_size=semantic_chunker_buffer_size,
            breakpoint_threshold_type=semantic_chunker_breakpoint_threshold_type,
            min_chunk_size=semantic_chunker_min_chunk_size,
            number_of_chunks=semantic_chunker_number_of_chunks,
            sentence_split_regex=semantic_chunker_sentence_split_regex
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=8092 * 2,  # chunk_size in characters
            chunk_overlap=100,
            length_function=len
        )

    def add_parsed_pdf(self, agentic_pdf_parser: AgenticPDFParser, agentic_chunker_context="", docling=False):
        self.agentic_pdf_parser = agentic_pdf_parser
        self.llm_client = self.agentic_pdf_parser.llm_client
        if docling:
            self.sentences = [
                x for x in self.agentic_pdf_parser.docling_sentences.keys() if len(x)
            ]
            self.metadata = self.agentic_pdf_parser.docling_sentences
        else:
            self.sentences = [
                x for x in self.agentic_pdf_parser.sentences.keys() if len(x)
            ]
            self.metadata = self.agentic_pdf_parser.sentences

        self.agentic_chunker = AgenticChunker(
            llm_client=self.llm_client,
            context=agentic_chunker_context,
            metadata=self.metadata,
        )
        logger.info(f"Got {len(self.sentences)} sentences for chunking")
        if len(self.sentences_docling):
            logger.info(f"Got {len(self.sentences_docling)} sentences from docling for chunking")

    def _get_embeddings(self, chunk):
        chunks_of_chunk = self.splitter.split_text(chunk)
        return {
            i: self.embedding_client.create_embedding_dict(input_phrases=[c])[c]
            for i, c in enumerate(chunks_of_chunk)
        }

    def get_semantic_chunks(self, sentences=None, embeddings=True, metadata=[]):
        if sentences is None:
            metadata = []
            sentences = []
            for i, chunk in enumerate(self.results.agentic_chunks):
                sentences.append(chunk["metadata"]["agentic_chunk"])
                metadata.append(chunk["metadata"] | {"sentence_id": i})
        chunks = self.semantic_chunker.create_documents(sentences, metadata)
        print(f"sentences: {len(sentences)}\nmetadata: {len(metadata)}")
        if embeddings:
            # embedding_dict = self.embedding_client.create_embedding_dict(input_phrases=chunks)
            for chunk  in chunks:
                self.results.semantic_chunks.append({
                    "embedding": self._get_embeddings(chunk.page_content),
                    "metadata": chunk.metadata,
                    "content": chunk.page_content,
                    "document_summary": self.agentic_pdf_parser.results.parsed_pdf.summary,
                    "filename": self.agentic_pdf_parser.results.file_name,
                    "summary_embedding": (
                        self.embedding_client.create_embedding_dict(
                            input_phrases=[self.agentic_pdf_parser.results.parsed_pdf.summary]
                        )[self.agentic_pdf_parser.results.parsed_pdf.summary])
                })
        else:
            self.results.semantic_chunks = chunks
        logger.info(f"Created {len(self.results.semantic_chunks)} semantic chunks")
        return self.results.semantic_chunks

    def get_agentic_chunks(self, sentences=None, embeddings=True):
        if sentences is None:
            sentences = self.sentences
        self.agentic_chunker.add_propositions(sentences)
        chunks = self.agentic_chunker.get_chunks()
        for chunk in chunks:
            if embeddings:
                # embedding_dict = self.embedding_client.create_embedding_dict(input_phrases=[chunk["metadata"]["content"]])
                chunk["embedding"] = self._get_embeddings(chunk["metadata"]["agentic_chunk"])
            # chunk["content"] = chunk["metadata"]["agentic_chunk"]
            chunk["filename"] = self.agentic_pdf_parser.results.file_name
            chunk["document_summary"] = self.agentic_pdf_parser.results.parsed_pdf.summary
            chunk["summary_embedding"] = self.embedding_client.create_embedding_dict(
                input_phrases=[self.agentic_pdf_parser.results.parsed_pdf.summary]
            )[self.agentic_pdf_parser.results.parsed_pdf.summary]
        self.results.agentic_chunks.extend(chunks)
        logger.info(f"Created {len(self.results.agentic_chunks)} agentic chunks")

    def run_pipline(self, agentic_pdf_parser=None, agentic_chunker_context=""):
        if self.agentic_pdf_parser is None:
            self.add_parsed_pdf(agentic_pdf_parser, agentic_chunker_context)
        self.get_agentic_chunks()
        self.get_semantic_chunks(
            sentences=[chunk["metadata"]["content"] for chunk in self.results.agentic_chunks],
            metadata={
                chunk["metadata"]["content"]: chunk["metadata"] for chunk in self.results.agentic_chunks
            }
        )
        return self.results.to_dict()