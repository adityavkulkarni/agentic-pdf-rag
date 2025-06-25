import json
import logging

from .config_manager import config
from .models import QueryType, SummaryResponse
from .db_handler import DBHandler
from .openai_client import AzureOpenAIChatClient

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(self, db_handler: DBHandler = None, llm_client=None):
        self.metadata_filter = {}
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

    def get_similar_chunks(self, query_embeddings, files=None):
        semantic_response = self.db_handler.similarity_search_chunks(
            table_name=self.db_handler.semantic_embedding_table,
            query_embedding=query_embeddings,
            embedding_column="embedding",
            top_k=5,
            files=files
        )
        return sorted([{
            "content": r[0],
            "metadata": r[1],
            "filename": r[2],
            "similarity": r[3]
        } for r in semantic_response if r is not None],
            key=lambda x: x["similarity"],
            reverse=True)[:5]

    def get_similar_chunk_by_summary(self, query_embeddings, top_k=5, files=None):
        semantic_response = self.db_handler.similarity_search_chunks(
            table_name=self.db_handler.semantic_embedding_table,
            query_embedding=query_embeddings,
            embedding_column="summary_embedding",
            top_k=top_k,
            files=files
        )
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

    def _filter_docs(self, metadata):
        for key in self.metadata_filter:
            if key not in metadata or metadata.get(key, "") != self.metadata_filter[key]:
                return False
        return True

    def get_document_outlines(self, files=None):
        docs = [
            doc for doc in self.db_handler.get_documents()
            if self._filter_docs(doc[1].get("custom_metadata", {}))
        ]
        if files:
            return [
                    f"filename: {doc[0]} | "
                    f"Summary: {doc[1]['parsed_pdf']['summary']} | "
                    f"Entities: {','.join(doc[1]['parsed_pdf']['ner'])} | "
                    f"Title: {doc[1]['parsed_pdf']['title']}"
                    for doc in docs if doc[0] in files
                ]
        else:
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
        outlines = self.get_document_outlines()
        if len(outlines) == 0:
            return {"type": "no_context"}
        prompt = (
            "**Role**: Retrieval Strategy Analyst  \n"
            "**Task**: \n1.Determine the most efficient retrieval type for the user query:  \n"
            "- `summary`: Retrieve document summaries only (for analysis tasks)  \n"
            "- `chunks`: Retrieve specific text chunks (for detail extraction)  \n"
            "2. Based on document outlines, determine all the relevant filenames "
            "and return a pipe separated list of file names\n\n"
            "### Decision Framework\n"
            "1. **Analyze query intent**:\n"
            "   - Use `summary` if the query requires:  \n"
            "     - High-level understanding (e.g., overview, compare, trends)  \n"
            "     - Synthesis across documents (e.g., summarize key points from all files)  \n"
            "     - Abstract concepts (e.g., risks, strategies, performance)  \n"
            "   - Use `chunks` if the query requires:  \n"
            "     - Specific facts/figures (e.g., Q3 sales number, exact quote)  \n"
            "     - Verbatim content (e.g., copy paragraph about X)  \n"
            "     - Location-specific data (e.g., in section 3.2, from page 14)  \n\n"
            "2. **Key decision triggers**:  \n"
            "   | Trigger Phrases         | Retrieval Type |  \n"
            "   |-------------------------|----------------|  \n"
            "   | Compare/contrast        | `summary`      |  \n"
            "   | Summarize/overview      | `summary`      |  \n"
            "   | What are the key...     | `summary`      |  \n"
            "   | Extract/show [specific] | `chunks`       |  \n"
            "   | Exact value of...       | `chunks`       |  \n"
            "   | In [file] on page...    | `chunks`       |  \n\n"
            f"Document Outlines: {outlines}\n\n"
            f"User Query: {query}"
        )
        llm_response = json.loads(
            self.llm_client.chat_completion(
                text=prompt, feature_model=QueryType
            ).choices[0].message.content
        )
        if llm_response["type"] == "summary":
            llm_response["files"] = llm_response["files"].split("|")
        else:
            prompt = (
                "You are a query optimization agent for a RAG system.\n"
                "Your task is to transform the user's analysis query into a detailed, "
                "multifaceted query that maximizes retrieval of all relevant document chunks.\n"
                "Use the document outlines to add additional context to the query to enhance retrieval.\n"
                "Identify all relevant filenames for the user's query.\n"
                "**User Query**:\n"
                f"{query}\n\n"
                "**Document Outline**:\n"
                f"{outlines}\n\n"
            )
            llm_response = json.loads(
                self.llm_client.chat_completion(
                    text=prompt, feature_model=SummaryResponse
                ).choices[0].message.content
            )
            llm_response["files"] = llm_response["files"].split("|")
            llm_response["type"] = "chunks"
        return llm_response

    def _query_context(self, query_embeddings, a_query_embeddings, top_k=5, files=None):
        return sorted(
            self.get_similar_chunks(query_embeddings, files=files) +
            self.get_similar_chunk_by_summary(query_embeddings, files=files) +
            self.get_similar_chunks(a_query_embeddings, files=files) +
            self.get_similar_chunk_by_summary(a_query_embeddings, files=files),
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]

    def get_context(self, query, top_k=5, metadata_filter={}, detailed=False):
        self.metadata_filter = metadata_filter
        additional_details = self.analyze_query(query)
        logger.info(f"Query type: {additional_details.get('type')}")
        logger.info(f"Relevant files: {additional_details.get('files')}")
        if additional_details.get("type") == "chunks":
            query_embeddings = self.db_handler.embedding_client.create_embedding_dict([query])[query]
            a_query_embeddings = self.db_handler.embedding_client.create_embedding_dict(
                [additional_details["augmented_query"]]
            )[additional_details["augmented_query"]]
            logger.info(f"Augmented query: {additional_details.get('augmented_query')}")
            context_dict = {}
            for file in additional_details["files"]:
                context_dict[file] = self._query_context(query_embeddings, a_query_embeddings, top_k=top_k, files=[file])
            results = []
            for key, value in context_dict.items():
                results += value
            if detailed:
                return results
            else:
                return '\n\n'.join([r['metadata']['content'] for r in results])
        elif additional_details.get("type") == "summary":
            return self.get_document_outlines(files=additional_details["files"])
        else:
            return None

class GenerationEngine:
    def __init__(self, llm_client: AzureOpenAIChatClient):
        self.llm_client = llm_client

    def generate_response(self, query, context, additional_instructions=None):
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
            "- Always follow any additional instructions specified above.\n"
            "- Format the response clearly. Use bullet points, tables, etc wherever necessary. Use markdown formatting.\n"
        )
        return self.llm_client.chat_completion(text=prompt).choices[0].message.content
