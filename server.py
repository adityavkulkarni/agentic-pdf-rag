import os
import sys
import base64
import logging
from flask import Flask, request, jsonify
from json_schema_to_pydantic import create_model
from agentic_pdf_rag import RAGPipeline


class RAGServer:
    def __init__(self, config_path=None, openai_api_key=None, openai_embeddings_api_key=None):
        self.app = Flask(__name__)
        self._configure_logging()
        self._init_pipeline(config_path, openai_api_key, openai_embeddings_api_key)
        self._register_routes()

    @staticmethod
    def _configure_logging():
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    def _init_pipeline(self, config_path, openai_api_key, openai_embeddings_api_key):
        self.rag_pipeline = RAGPipeline(
        config_file=config_path or os.path.join(os.getcwd(), "config", "config.ini"),
        openai_api_key=openai_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
        openai_embeddings_api_key=openai_embeddings_api_key or os.getenv("AZURE_OPENAI_API_KEY_EAST2"),
        )
        self.pdf_parser = self.rag_pipeline.get_agentic_pdf_parser()
        self.pdf_chunker = self.rag_pipeline.get_pdf_chunker()
        self.db_handler = self.rag_pipeline.get_db_handler()
        self.retriever = self.rag_pipeline.get_retrieval_engine()
        self.generator = self.rag_pipeline.get_generation_engine()

    def _register_routes(self):
        self.app.add_url_rule('/get_config', view_func=self.get_config, methods=['GET'])
        self.app.add_url_rule('/set_config', view_func=self.set_config, methods=['POST'])
        self.app.add_url_rule('/parse_pdf', view_func=self.parse_pdf, methods=['POST'])
        self.app.add_url_rule('/get_chunks', view_func=self.get_chunks, methods=['POST'])
        self.app.add_url_rule('/add_document_to_context', view_func=self.add_document_to_context, methods=['POST'])
        self.app.add_url_rule('/get_documents', view_func=self.get_documents, methods=['GET'])
        self.app.add_url_rule('/get_document_by_name', view_func=self.get_document_by_name, methods=['POST'])
        self.app.add_url_rule('/get_context', view_func=self.get_context, methods=['POST'])
        self.app.add_url_rule('/get_final_response', view_func=self.get_final_response, methods=['POST'])

    def get_config(self):
        return jsonify({
            "config": self.rag_pipeline.config.to_dict()
        })

    def set_config(self):
        request_data = request.get_json()
        self.rag_pipeline.config.from_dict(request_data)
        return jsonify({
            "config": self.rag_pipeline.config.to_dict()
        })

    def parse_pdf(self):
        request_data = request.get_json()
        pdf_data = base64.b64decode(request_data['pdf'])
        pdf_file = os.path.join(self.rag_pipeline.config.output_directory, request_data["filename"])

        with open(pdf_file, 'wb') as pdf:
            pdf.write(pdf_data)

        feature_model = create_model(request_data["custom_feature_model"]) if request_data.get(
            "custom_feature_model") else None

        response = self.pdf_parser.run_pipline(
            pdf_file=pdf_file,
            file_name=request_data["filename"],
            custom_extraction_prompt=request_data.get("custom_extraction_prompt"),
            custom_feature_model=feature_model,
        )
        return jsonify(response)

    def get_chunks(self):
        request_data = request.get_json()
        context = request_data.get('agentic_chunker_context', "")
        response = self.pdf_chunker.run_pipline(
            self.pdf_parser,
            agentic_chunker_context=context
        )
        return jsonify(response)

    def add_document_to_context(self):
        request_data = request.get_json()
        pdf_data = base64.b64decode(request_data['pdf'])
        pdf_file = os.path.join(self.rag_pipeline.config.output_directory, request_data["filename"])

        with open(pdf_file, 'wb') as pdf:
            pdf.write(pdf_data)

        parsed_pdf = self.pdf_parser.run_pipline(
            pdf_file=pdf_file,
            file_name=request_data["filename"],
            custom_extraction_prompt=request_data.get("custom_extraction_prompt"),
            custom_feature_model=create_model(request_data["custom_feature_model"]) if request_data.get(
                "custom_feature_model") else None,
        )

        chunks = self.pdf_chunker.run_pipline(
            self.pdf_parser,
            agentic_chunker_context=request_data.get('agentic_chunker_context', "")
        )

        self.db_handler.insert_document(parsed_pdf)
        self.db_handler.batch_insert_embeddings(
            agentic_chunks=chunks["agentic_chunks"],
            semantic_chunks=chunks["semantic_chunks"]
        )
        return jsonify({"parsed_pdf": parsed_pdf, "chunks": chunks})

    def get_documents(self):
        data = self.db_handler.get_documents()
        return jsonify({
            "status": "success",
            "documents": {doc[0]: doc[1] for doc in data}
        })

    def get_document_by_name(self):
        request_data = request.get_json()
        filename = request_data.get('filename')
        data = self.db_handler.get_document_by_name(filename=filename)
        return jsonify({
            "status": "success",
            "documents": {doc[0]: doc[1] for doc in data}
        })

    def get_context(self):
        request_data = request.get_json()
        query = request_data.get('query')
        response = self.retriever.get_context(query=query,
                                              top_k=request_data.get("top_k", 5),
                                              metadata_filter=request_data.get("metadata_filter", {}),
                                              detailed=request_data.get("detailed", False))
        return jsonify({"context": response})

    def get_final_response(self):
        request_data = request.get_json()
        response = self.generator.generate_response(
            query=request_data.get('query'),
            context=request_data.get('context'),
            additional_instructions=request_data.get('additional_instructions')
        )
        return jsonify(response)

    def run(self, debug=True, **kwargs):
        self.app.run(debug=debug, **kwargs)


if __name__ == '__main__':
    server = RAGServer()
    server.run()
