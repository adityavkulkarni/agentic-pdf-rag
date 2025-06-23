import json
import logging

from pydantic import BaseModel, Field

from .config_manager import config
from .db_handler import DBHandler
from .openai_client import AzureOpenAIChatClient

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(self, db_handler: DBHandler = None, llm_client=None):
        self.db_handler = db_handler if db_handler else DBHandler(
            dbname=config.db_name,
            user=config.db_user,
            password=config.db_password,
            host=config.db_host,
            port=config.db_port,
        )
        self.llm_client = llm_client if llm_client else AzureOpenAIChatClient(
            model=config.agentic_pdf_parser_model,
            api_key=config.openai_api_key,
            api_endpoint=config.openai_endpoint,
            api_version=config.openai_api_version,
        )

    def get_similar_chunks(self, query_embeddings):
        semantic_response = self.db_handler.similarity_search_chunks(
            table_name=self.db_handler.semantic_embedding_table,
            query_embedding=query_embeddings,
            embedding_column="embedding",
            top_k=5)
        return sorted([{
            "content": r[0],
            "metadata": r[1],
            "filename": r[2],
            "similarity": r[3]
        } for r in semantic_response if r is not None],
            key=lambda x: x["similarity"],
            reverse=True)[:5]

    def get_similar_chunk_by_summary(self, query_embeddings, top_k=5):
        semantic_response = self.db_handler.similarity_search_chunks(
            table_name=self.db_handler.semantic_embedding_table,
            query_embedding=query_embeddings,
            embedding_column="summary_embedding",
            top_k=top_k)
        return sorted([{
            "content": r[0],
            "metadata": r[1],
            "filename": r[2],
            "similarity": r[3]
        } for r in semantic_response if r is not None],
            key=lambda x: x["similarity"],
            reverse=True)[:5]

    def get_similar_document(self, query_embeddings, top_k=5):
        semantic_response = self.db_handler.similarity_search_document(
            table_name=self.db_handler.document_table,
            query_embedding=query_embeddings,
            top_k=top_k)
        return [{
            "content": r[0],
            "metadata": r[1],
            "similarity": r[2],
        } for r in semantic_response if r is not None]

    def get_document_outlines(self):
        docs = self.db_handler.get_documents()
        return "\n".join(
            [
                f"filename: {doc[0]} | "
                f"Summary: {doc[1]['parsed_pdf']['summary']} | "
                f"Entities: {','.join(doc[1]['parsed_pdf']['ner'])} | "
                f"Title: {doc[1]['parsed_pdf']['title']}"
                for doc in docs
            ]
        )

    def analyze_query(self, query):
        prompt = (
            "Classify the query into:\n"
            "- retrieval (seeks specific information from 1 or more files)\n"
            "- analysis (requires summarization/aggregation across files).\n"
            "Respond ONLY with retrieval or analysis.\n"
        )
        class QueryType(BaseModel):
            type: str = Field(..., description="Type of query: retrieval or analysis")
            class Config:
                extra = "forbid"

        query_type = json.loads(
            self.llm_client.chat_completion(
                text=prompt, feature_model=QueryType
            ).choices[0].message.content
        )["type"]
        if query_type == "retrieval":
            prompt = (
                "You are a document retrieval expert for a RAG system. Given document outline, "
                "your task is to identify all relevant filenames for the user's query. Follow this process:\n"
                "### Instructions\n"
                "1. **Analyze the query**:\n"
                "   - Identify key entities, topics, and intent\n"
                "   - Note any ambiguity or missing context\n"
                "2. **Evaluate document relevance**:\n"
                "   - For each document, check if its summary addresses:\n"
                "     * Core query topics\n"
                "     * Implicit requirements (time periods, comparisons, etc.)\n"
                "     * Domain-specific concepts\n"
                "   - Assign relevance score: Relevant/Marginal/Irrelevant\n"
                "3. **Handle ambiguity**:\n"
                "   - If query lacks context (e.g., missing timeframes, unspecified entities):\n"
                "     * Augment with likely parameters from document context\n"
                "     * Expand using synonyms and related concepts\n"
                "   - Example:  \n"
                "     Original: Sales data → Augmented: Q4 2023 sales figures for EMEA region\n\n"
                "### Document Outlines\n"
                f"{self.get_document_outlines()}\n"
                "### Query\n"
                f"{query}"
            )
            class FileNames(BaseModel):
                files: str = Field(..., description="pipe separated list of filenames")
                augmented_query: str = Field(..., description="augmented query")
                class Config:
                    extra = "forbid"

            llm_response = json.loads(
                self.llm_client.chat_completion(
                    text=prompt, feature_model=FileNames
                ).choices[0].message.content
            )
            llm_response["files"] = llm_response["files"].split("|")
        elif query_type == "analysis":
            prompt = (
                "You are a query optimization agent for a RAG system. "
                "Your task is to transform the user's analysis query into a detailed, "
                "multifaceted query that maximizes retrieval of all relevant document chunks. "
                "Follow these steps:\n\n"
                "1. **Decompose the analysis goal**: Break down the user's request into core subtopics, comparison dimensions, or aggregation criteria.\n"
                "2. **Expand context**: Based on document outlines and query include:\n"
                "   - Key entities (names, dates, concepts)\n"
                "   - Implicit requirements (e.g., time periods, metrics, contrasting elements)\n"
                "   - Synonyms and domain-specific terminology\n"
                "3. **Structure for similarity search**: \n"
                "   - Use natural language descriptions instead of keywords \n"
                "   - Emphasize relationships between concepts (e.g., impact of X on Y)\n"
                "   - Include both broad themes and specific sub-queries\n\n"
                "**User Query**:\n"
                f"{query}\n\n"
                "**Document Outline**:\n"
                f"{self.get_document_outlines()}\n\n"
            )

            class SummaryResponse(BaseModel):
                augmented_query: str = Field(..., description="augmented query")
                class Config:
                    extra = "forbid"
            llm_response = json.loads(
                self.llm_client.chat_completion(
                    text=prompt, feature_model=SummaryResponse
                ).choices[0].message.content
            )
        llm_response["query_type"] = query_type
        print(llm_response)
        return llm_response

    def get_context(self, query):
        additional_details = self.analyze_query(query)
        query_embeddings = self.db_handler.embedding_client.create_embedding_dict([query])[query]
        a_query_embeddings = self.db_handler.embedding_client.create_embedding_dict(
            [additional_details["augmented_query"]]
        )[additional_details["augmented_query"]]
        context = sorted(
            self.get_similar_chunks(query_embeddings) +
            self.get_similar_chunk_by_summary(query_embeddings) +
            self.get_similar_chunks(a_query_embeddings) +
            self.get_similar_chunk_by_summary(a_query_embeddings),
            key=lambda x: x["similarity"],
            reverse=True
        )[:5]
        return context


class GenerationEngine:
    def __init__(self, llm_client: AzureOpenAIChatClient):
        self.llm_client = llm_client

    def generate_response(self, query, context, additional_instructions=None):
        """prompt = (
            "You are a legal language model. Use the following context retrieved from a database to answer the "
            "user’s legal question or draft/review contract language. If relevant, quote or paraphrase the provided context. "
            "If the context does not fully answer the question, state so explicitly and avoid speculation.\n\n"
            "[BEGIN CONTEXT]\n"
            f"Text from contract:\n{'\n\n'.join([r['content'] for r in context])}\n\n"
            f"Elaborate text from contracts: \n{'\n\n'.join([r['metadata']['content'] for r in context])}\n\n"
            "[END CONTEXT]\n\n"
            f"User Query: {query}"
        )
        response = llm_client.chat_completion(text=prompt).choices[0].message.content"""
        additional_instructions = f"Additional Instructions: {additional_instructions}" if not additional_instructions else ""
        prompt = (
            "You are an intelligent assistant.\n"
            f"{additional_instructions}\n\n"
            "Context:\n"
            f"{context}\n\n"
            "User Query:\n"
            f"{query}\n\n"
            "Instructions:\n"
            "- Use the provided context to answer the user query as accurately and concisely as possible.\n"
            "- If the context does not contain enough information, indicate what is missing and suggest next steps.\n"
            "- Always follow any additional instructions specified above."
        )
        return self.llm_client.chat_completion(text=prompt).choices[0].message.content
