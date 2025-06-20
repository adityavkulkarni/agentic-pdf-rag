import logging

from .config_manager import config
from .db_handler import DBHandler
from .openai_client import AzureOpenAIChatClient

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(self, db_handler: DBHandler = None):
        self.db_handler = db_handler if db_handler else DBHandler(
            dbname=config.db_name,
            user=config.db_user,
            password=config.db_password,
            host=config.db_host,
            port=config.db_port,
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

    def analyze_query(self, query):
        # Decide if this is single file or multifile query
        # if single file, proceed with chunk analysis
        # if multifile, search with summary_query
        # return (mode, possible_filenames, summary_query)
        pass

    def get_context(self, query):
        query_embeddings = self.db_handler.embedding_client.create_embedding_dict([query])[query]
        context = sorted(
            self.get_similar_chunks(query_embeddings) +
            self.get_similar_chunk_by_summary(query_embeddings),
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
