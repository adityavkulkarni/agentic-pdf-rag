import json
import logging

from .config_manager import config
from .models import QueryType, SummaryResponse, ContextType, MultiModalResponse
from .db_handler import DBHandler
from .openai_client import AzureOpenAIChatClient

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(self, db_handler: DBHandler = None, llm_client=None, use_qwen3=False):
        self.page_sources = None
        self.pages = None
        self.custom_fields = None
        self.sources = None
        self.metadata_filter = {}
        self.db_handler = db_handler if db_handler else DBHandler(
            dbname=config.db_name,
            user=config.db_user,
            password=config.db_password,
            host=config.db_host,
            port=config.db_port,
            use_qwen3=use_qwen3
        )
        self.llm_client = llm_client if llm_client else AzureOpenAIChatClient(
            model=config.agentic_pdf_parser_model,
            api_key=config.openai_api_key,
            api_endpoint=config.openai_endpoint,
            api_version=config.openai_api_version,
        )

    def get_similar_chunks(self, query_embeddings, docling=False, files=None):
        if docling:
            semantic_embedding_table = self.db_handler.docling_semantic_embedding_table
        else:
            semantic_embedding_table = self.db_handler.semantic_embedding_table
        semantic_response = self.db_handler.similarity_search_chunks(
            table_name=semantic_embedding_table,
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

    def get_similar_chunk_by_summary(self, query_embeddings, top_k=5, docling=False, files=None):
        if docling:
            semantic_embedding_table = self.db_handler.docling_semantic_embedding_table
        else:
            semantic_embedding_table = self.db_handler.semantic_embedding_table
        semantic_response = self.db_handler.similarity_search_chunks(
            table_name=semantic_embedding_table,
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

    def get_pages(self):
        self.pages = self.db_handler.get_pages()
        return self.pages

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
        db_docs = self.db_handler.get_documents()
        docs = [
            doc for doc in db_docs
            if self._filter_docs(doc[1].get("custom_metadata", {}))
        ]
        if files:
            db_docs = [doc for doc in db_docs if doc[0].strip() in files]
            docs = [doc for doc in docs if doc[0].strip() in files]
            logger.info(f"Found {len(db_docs)} documents; Filtered {len(db_docs) - len(docs)} documents")
            return [
                f"filename: {doc[0]} | "
                f"Summary: {doc[1]['parsed_pdf']['summary']} | "
                f"Entities: {','.join(doc[1]['parsed_pdf']['ner'])} | "
                f"Title: {doc[1]['parsed_pdf']['title']}"
                for doc in docs
            ]
        else:
            logger.info(f"Found {len(db_docs)} documents; Filtered {len(db_docs) - len(docs)} documents")
            return "\n".join(
                [
                    f"filename: {doc[0]} | "
                    f"Summary: {doc[1]['parsed_pdf']['summary']} | "
                    f"Entities: {','.join(doc[1]['parsed_pdf']['ner'])} | "
                    f"Title: {doc[1]['parsed_pdf']['title']}"
                    for doc in docs
                ]
            )

    def get_document_content(self, files=None, use_docling=True):
        content_key = "docling_content" if use_docling else "pages_llm"
        logger.info(f"Getting {content_key}")
        def get_content(d):
            if use_docling:
                return d[1]["parsed_pdf"]["docling_content"]
            else:
                return "\\n".join(d[1]["parsed_pdf"]["pages_llm"])
        return {doc[0]: get_content(doc)
                for file in set(files)
                for doc in self.db_handler.get_documents(filename=file)
                }

    def analyze_query(self, query, attachment=None):
        outlines = self.get_document_outlines()
        if len(outlines) == 0:
            return {"type": "no_context"}
        attachment_string = f"Attachment:\n{attachment}" if attachment else ""
        prompt = (
            "**Role**: Retrieval Strategy Analyst  \n"
            "**Task**: \n1.Determine the most efficient retrieval type for the user query and attachment(if any):  \n"
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
            f"User Query: {query}\n"
        ) + attachment_string
        llm_response = json.loads(
            self.llm_client.chat_completion(
                text=prompt, feature_model=QueryType
            ).choices[0].message.content
        )
        if llm_response["type"] == "summary":
            llm_response["files"] = list(set([file.strip() for file in llm_response["files"].split("|")]))
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
            llm_response["files"] = list(set([file.strip() for file in llm_response["files"].split("|")]))
            llm_response["type"] = "chunks"
        return llm_response

    def _query_context(self, query_embeddings, a_query_embeddings, top_k=5, files=None):
        return sorted(
            self.get_similar_chunks(query_embeddings, files=files) +
            self.get_similar_chunk_by_summary(query_embeddings, files=files) +
            self.get_similar_chunks(a_query_embeddings, files=files) +
            self.get_similar_chunk_by_summary(a_query_embeddings, files=files) +
            self.get_similar_chunks(query_embeddings, files=files, docling=True) +
            self.get_similar_chunk_by_summary(query_embeddings, files=files, docling=True) +
            self.get_similar_chunks(a_query_embeddings, files=files, docling=True) +
            self.get_similar_chunk_by_summary(a_query_embeddings, files=files, docling=True),
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]

    def get_page_level_context(self, query, attachment=None):
        self.get_pages()
        outlines = [
            f"Filename: {page["filename"]}_page{page["page_no"]} | Summary: {page["summary"]}"
            for page in self.pages
        ]
        attachment_string = f"**Attachment**:\n{attachment}" if attachment else ""
        prompt = (
            "You are a query optimization agent for a RAG system.\n"
            "Your task is to transform the user's analysis query into a detailed, "
            "multifaceted query that maximizes retrieval of all relevant document chunks.\n"
            "Use the page outlines to add additional context to the query to enhance retrieval.\n"
            "Identify all relevant filenames for the user's query.\n"
            "**User Query**:\n"
            f"{query}\n\n"
            "**Page Outline**:\n"
            f"{"\n".join(outlines)}\n\n"
        ) + attachment_string
        llm_response = json.loads(
            self.llm_client.chat_completion(
                text=prompt, feature_model=SummaryResponse
            ).choices[0].message.content
        )
        llm_response["files"] = list(set([file.strip() for file in llm_response["files"].split("|")]))
        logging.info(f"Relevant pages: {llm_response['files']}")
        llm_response["type"] = "pages"
        self.page_sources = {file.split("_page")[0]: [] for file in llm_response["files"]}
        for page in self.pages:
            if (
                    len([x for x in llm_response["files"] if f"{page["filename"]}_" in x]) ==
                    len([x for x in self.pages if page["filename"] == x["filename"]])
            ):
                self.sources["documents"].append(page["filename"])
                continue
            if (
                    page["filename"] in self.page_sources and
                    f"{page["filename"]}_page{page["page_no"]}" in llm_response["files"]
            ):
                self.page_sources[page["filename"]].append(page["page_no"])
        for key in self.page_sources:
            self.page_sources[key] = list(set(self.page_sources[key]))
        return [
            f"text_content: {page["text_content"]}\nvisual_elements: {page["visual_elements"]}"
            for page in self.pages
            if f"{page["filename"]}_page{page["page_no"]}" in llm_response["files"]
        ]
        # return llm_response

    def evaluate_context(self, query, context, page_response):
        prompt = (
            "You are a context analyzer agent for a RAG system.\n"
            "Your task is to identify the best context for the user query.\n, "
            "Compare Document Context and Page Context to identify which is more suitable to answer the query.\n"
            "If extremely unsure then reply with hybrid(as a last option)."
            "Do not use outside knowledge.\n"
            "Return only: document or page or hybrid depending on which is better."
            "**User Query**:\n"
            f"{query}\n\n"
            "**Document Context**:\n"
            f"{context}\n\n"
            "**Page Context**:\n"
            f"{page_response}\n\n"
        )
        llm_response = json.loads(
            self.llm_client.chat_completion(
                text=prompt, feature_model=ContextType
            ).choices[0].message.content
        )
        logging.info(f"Using {llm_response["context_type"]} level context")
        for key in self.sources:
            self.sources[key] = list(set(self.sources[key]))
        if llm_response["context_type"] == "pages":
            context = page_response
            self.sources["documents"] = []
            for page in self.page_sources:
                if len(self.page_sources[page]) == 0:
                    self.sources["documents"].append(page)
                    self.page_sources.remove(page)
            self.sources = {** self.page_sources | {** self.sources }}
        elif llm_response["context_type"] == "document":
            context = llm_response
        else:
            context = f"Page level context: {page_response}\n\n Document: {llm_response}"
            self.page_sources["documents"] = [doc for doc in self.sources["documents"] if
                                              doc not in self.page_sources.keys()]
            self.sources = self.page_sources
        logging.info(f"Sources: {self.sources}")
        return context

    def get_context(self, query, top_k=5, attachment=None, metadata_filter={}, detailed=False):
        self.metadata_filter = metadata_filter
        additional_details = self.analyze_query(query, attachment)
        logger.info(f"Query type: {additional_details.get('type')}")
        logger.info(f"Relevant files: {additional_details.get('files')}")
        self.custom_fields = [
            {doc[0]: doc[1].get("custom_extraction_llm_response", {})}
            for doc in self.db_handler.get_documents() if doc[0] in additional_details.get("files")
        ]
        self.sources = {"documents": additional_details["files"]}
        page_context = self.get_page_level_context(query)
        doc_context = None
        if additional_details.get("type") == "chunks":
            query_embeddings = self.db_handler.embedding_client.create_embedding_dict([query])[query]
            a_query_embeddings = self.db_handler.embedding_client.create_embedding_dict(
                [additional_details["augmented_query"]]
            )[additional_details["augmented_query"]]
            logger.info(f"Augmented query: {additional_details.get('augmented_query')}")
            context_dict = {}
            for file in additional_details["files"]:
                context_dict[file] = self._query_context(query_embeddings, a_query_embeddings, top_k=top_k,
                                                         files=[file])
            results = []
            for key, value in context_dict.items():
                results += value
            if detailed:
                doc_context = results
            else:
                doc_context = '\n\n'.join([r['metadata']['content'] for r in results])
        elif additional_details.get("type") == "summary":
            doc_context = (
                "Document Outlines: \n"
                f"{"\n".join(self.get_document_outlines(files=additional_details["files"]))}\n\n"
                "Document Content: \n"
                f"{json.dumps(self.get_document_content(files=additional_details["files"]))}"
            )
            # return context
        return self.evaluate_context(query, doc_context, page_context)


class GenerationEngine:
    def __init__(self, llm_client: AzureOpenAIChatClient):
        self.llm_client = llm_client

    def generate_response(self, query, context, role=None, additional_instructions=None, attachment=None,
                          structured=True):
        additional_instructions = f"Additional Instructions: {additional_instructions}" if not additional_instructions else ""
        structure_instructions = ("- Summarize the response and output it in summary field\n"
                                  "- Response should be in the response field"
                                  "- If there is any content that can be use markdown, output it in the markdown field\n"
                                  if structured else "")
        prompt = (
                     f"You are an intelligent assistant. {role if role else ""}\n"
                     f"{additional_instructions}\n\n"
                     "Context:\n"
                     f"{context}\n\n"
                     "User Query:\n"
                     f"{query}\n\n"
                     "Instructions:\n"
                     "- Use the provided context to answer the user query as accurately and concisely as possible.\n"
                     "- Try to provide maximum information as possible.\n"
                     "- Always follow any additional instructions specified above.\n"
                     "- Format the response clearly. Use bullet points, tables, etc wherever necessary. Use markdown formatting.\n"
                 ) + structure_instructions

        messages = [
            {
                "role": "system",
                "content": (
                        "You are an intelligent assistant. "
                        + (role if role else "")
                        + "\n" + additional_instructions)
            },
            {
                "role": "user",
                "content": "User Query:\n" + query
            },
            {
                "role": "system",
                "content": "Context:\n" + context
            },
            {
                "role": "system",
                "content": (
                        "Instructions:\n"
                        "- Use the provided context to answer the user query as accurately and concisely as possible.\n"
                        "- Use the attachment if necessary and clearly mention if the attachment is used.\n"
                        "- Clearly mention which part of response is according to the context and which is according to attachment.\n"
                        "- Try to provide maximum information as possible.\n"
                        "- Always follow any additional instructions specified above.\n"
                        "- Format the response clearly. Use bullet points, tables, etc wherever necessary. Use markdown formatting.\n"
                        + structure_instructions
                )
            }
        ]
        if attachment:
            messages.append(
                {
                    "role": "user",
                    "content": "Attachment contents:\n" + attachment
                }
            )
        return json.loads(
            self.llm_client.chat_completion(
                text=messages, feature_model=MultiModalResponse
            ).choices[0].message.content
        )
