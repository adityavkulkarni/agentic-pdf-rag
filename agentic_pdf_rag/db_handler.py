import os
import psycopg2
import logging

from psycopg2.extras import Json
from psycopg2.extras import execute_values

from . import config_manager
from .openai_client import AzureOpenAIChatClient
from .qwen_embedding import Qwen3RetrievalSystem

logger = logging.getLogger(__name__)

class PostgreSQLVectorClient:
    def __init__(self, dbname, user, password, host, port):
        try:
            self.conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
            self.cur = self.conn.cursor()
            self._enable_extensions()
            logger.info(f"Database connection established: {user}@{host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _enable_extensions(self):
        """Enable required PostgreSQL extensions"""
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()
        logger.debug("Enabling required PostgreSQL extensions")

    def _create_document_table(self, table_name, vector_dim=3072):
        """Create table with vector column"""
        try:
            create_table_query = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id SERIAL PRIMARY KEY,
                            filename TEXT,
                            summary_embedding vector({vector_dim}),
                            data JSONB
                        );
                    """
            self.cur.execute(create_table_query)
            self.conn.commit()
            logger.info(f"Table '{table_name}' created or already exists")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            self.conn.rollback()
            raise

    def _create_embedding_table(self, table_name, vector_dim=3072):
        """Create table with vector column"""
        try:
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding vector({vector_dim}),
                    filename TEXT,
                    summary_embedding vector({vector_dim}),
                    metadata JSONB
                );
            """
            self.cur.execute(create_table_query)
            self.conn.commit()
            logger.info(f"Table '{table_name}' created or already exists")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            self.conn.rollback()
            raise

    def _insert_embedding(self, table_name, content, embedding, filename, summary_embedding, metadata=None):
        """Insert single embedding into the table"""
        insert_query = f"""
            INSERT INTO {table_name} (content, embedding, filename, summary_embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """
        self.cur.execute(insert_query, (content, embedding, filename, summary_embedding, Json(metadata)))
        self.conn.commit()
        logger.info(f"Inserted embedding for file {filename} into {table_name}")

    def _insert_document(self, table_name, filename, data, summary_embedding):
        """Insert single embedding into the table"""
        insert_query = f"""
            INSERT INTO {table_name} (filename, data, summary_embedding)
            VALUES (%s, %s, %s)
        """
        self.cur.execute(insert_query, (filename, Json(data), summary_embedding))
        self.conn.commit()
        logger.info(f"Inserted document {filename} into {table_name}")

    def _batch_insert_embeddings(self, table_name, records):
        """Batch insert multiple embeddings"""
        insert_query = f"""
            INSERT INTO {table_name} (content, embedding, filename, summary_embedding, metadata)
            VALUES %s
        """
        execute_values(
            self.cur,
            insert_query,
            [
                 (rec["content"], rec["embedding"], rec["filename"], rec["summary_embedding"],Json(rec.get("metadata")))
                for rec in records
            ]
        )
        self.conn.commit()
        logger.info(f"Batch insert into {table_name} successful")

    def similarity_search_chunks(self, table_name, query_embedding, embedding_column, top_k=5, files=None):
        """Find similar vectors using cosine similarity"""
        if files:
            files = [f"'{file}'" for file in files]
        clause = f"""WHERE filename in ({','.join(files)})""" if files else ""
        search_query = f"""
            SELECT content, metadata, filename, 1 - ({embedding_column} <=> %s::vector) AS similarity
            FROM {table_name}
            {clause}
            ORDER BY similarity DESC
            LIMIT {top_k}
        """
        self.cur.execute(search_query, (query_embedding,))
        results = self.cur.fetchall()
        return results

    def similarity_search_document(self, table_name, query_embedding, top_k=5):
        """Find similar vectors using cosine similarity"""
        search_query = f"""
            SELECT filename, data, 1 - (summary_embedding <=> %s::vector) AS similarity
            FROM {table_name}
            ORDER BY similarity DESC
            LIMIT {top_k}
        """
        self.cur.execute(search_query, (query_embedding,))
        results = self.cur.fetchall()
        return results

    def _fetch_documents(self, table_name):
        """Find similar vectors using cosine similarity"""
        search_query = f"""
            SELECT filename, data
            FROM {table_name}
        """
        self.cur.execute(search_query)
        results = self.cur.fetchall()
        logger.info(f"Fetched {len(results)} documents from {table_name}")
        return results

    def _fetch_document(self, table_name, file_name):
        """Find similar vectors using cosine similarity"""
        search_query = f"""
            SELECT filename, data
            FROM {table_name}
            WHERE filename LIKE '%{file_name}%'
        """
        self.cur.execute(search_query)
        results = self.cur.fetchall()
        logger.info(f"Fetched {len(results)} documents from {table_name}")
        return results

    def _delete_document(self, table_name, file_name):
        """Find similar vectors using cosine similarity"""
        query = f"""
            DELETE FROM {table_name}
            WHERE filename LIKE '%{file_name}%'
        """
        self.cur.execute(query)
        logger.info(f"Deleted {file_name} from {table_name}")

    def _close(self):
        self.cur.close()
        self.conn.close()


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
        for i, page in enumerate(document["parsed_pdf"]["pages_descriptions"]):
            self._insert_document(
                table_name=self.page_table,
                filename=f"{document["file_name"].split(".")[0]}_page{i+1}",
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
                "content": chunk["content"],
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
