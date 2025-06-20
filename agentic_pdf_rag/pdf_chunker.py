import os
import logging

from pydantic import BaseModel, Field
from typing import List, Any
from langchain_experimental.text_splitter import SemanticChunker

from .config_manager import config
from .agentic_pdf_parser import AgenticPDFParser
from .agentic_chunker import AgenticChunker
from .embeddings import OpenAIEmbeddings
from .openai_client import AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)


class Results(BaseModel):
    semantic_chunks: List[Any] = Field(default_factory=list)
    agentic_chunks: List[Any] = Field(default_factory=list)

    def to_dict(self):
        return {
            "semantic_chunks": self.semantic_chunks,
            "agentic_chunks": self.agentic_chunks,
        }

class PDFChunker:
    def __init__(self,
                 agentic_pdf_parser: AgenticPDFParser=None,
                 embedding_model="text-embedding-3-large",
                 custom_embedding_model=None,
                 openai_embeddings_endpoint = None,
                 openai_embeddings_api_key = None,
                 openai_embeddings_api_version = None,
                 semantic_chunker_buffer_size = 1,
                 semantic_chunker_breakpoint_threshold_type = "percentile",
                 semantic_chunker_sentence_split_regex=r"(?<=[.?!])\s+",
                 semantic_chunker_min_chunk_size=128,
                 semantic_chunker_number_of_chunks=10,
                 agentic_chunker_context = ""
                 ):
        openai_embeddings_api_version = openai_embeddings_api_version or config.openai_embeddings_api_version
        openai_embeddings_endpoint = openai_embeddings_endpoint or config.openai_embeddings_endpoint
        openai_embeddings_api_key = openai_embeddings_api_key or config.openai_embeddings_api_key

        self.results = Results()
        self.agentic_pdf_parser = None
        self.llm_client = None
        self.agentic_chunker = None
        self.sentences = []
        self.embedding_client = AzureOpenAIEmbeddings(
            endpoint=openai_embeddings_endpoint,
            api_key=openai_embeddings_api_key,
            api_version=openai_embeddings_api_version,
            model=embedding_model
        )
        if custom_embedding_model is None:
            self.embeddings = OpenAIEmbeddings(self.embedding_client)
        else:
            self.embeddings = embedding_model

        if agentic_pdf_parser:
            self.add_parsed_pdf(agentic_pdf_parser, agentic_chunker_context)
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

    def add_parsed_pdf(self, agentic_pdf_parser: AgenticPDFParser, agentic_chunker_context=""):
        self.agentic_pdf_parser = agentic_pdf_parser
        self.llm_client = self.agentic_pdf_parser.llm_client
        self.agentic_chunker = AgenticChunker(
            llm_client=self.llm_client,
            context=agentic_chunker_context
        )
        self.sentences = [x for x in "\n\n".join(self.agentic_pdf_parser.results.parsed_pdf.pages_llm).split("\n\n") if
                          len(x)]
        logger.info(f"Got {len(self.sentences)} sentences for chunking")

    def get_semantic_chunks(self, sentences=None, embeddings=True, metadata={}):
        if sentences is None:
            sentences = self.sentences
        chunks = self.semantic_chunker.create_documents(sentences)
        chunks = [chunk.page_content for chunk in chunks]
        if embeddings:
            embedding_dict = self.embedding_client.create_embedding_dict(input_phrases=chunks)
            for chunk  in chunks:
                key = "None"
                for key in metadata:
                    if chunk in key:
                        break
                    else:
                        key = "None"
                document_summary = self.agentic_pdf_parser.results.parsed_pdf.summary
                md = metadata.get(key, {"content": chunk})
                self.results.semantic_chunks.append({
                    "embedding": embedding_dict[chunk],
                    "metadata": md,
                    "content": chunk,
                    "document_summary": self.agentic_pdf_parser.results.parsed_pdf.summary,
                    "filename": self.agentic_pdf_parser.results.file_name,
                    "summary_embedding": self.embedding_client.create_embedding_dict(
                        input_phrases=[document_summary])[document_summary]
                })
        else:
            self.results.semantic_chunks = chunks
        logger.info(f"Created {len(self.results.semantic_chunks)} semantic chunks")
        return self.results.semantic_chunks

    def get_agentic_chunks(self, embeddings=True):
        self.agentic_chunker.add_propositions(self.sentences)
        chunks = self.agentic_chunker.get_chunks()
        if embeddings:
            embedding_dict = self.embedding_client.create_embedding_dict(input_phrases=[chunk["metadata"]["content"] for chunk in chunks])
            for chunk in chunks:
                chunk["embedding"] = embedding_dict[chunk["metadata"]["content"]]
                chunk["content"] = chunk["metadata"]["content"]
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
            })
        return self.results.to_dict()