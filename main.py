import os
import sys
import base64
import logging

from flask import Flask, request, jsonify
from json_schema_to_pydantic import create_model
from agentic_pdf_rag import RAGPipeline

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

rag_pipeline = RAGPipeline(
    config_file=os.path.join(os.getcwd(), "config", "config.ini"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_embeddings_api_key=os.getenv("AZURE_OPENAI_API_KEY_EAST2")
)
pdf_parser = rag_pipeline.get_agentic_pdf_parser()
pdf_chunker = rag_pipeline.get_pdf_chunker()
db_handler = rag_pipeline.get_db_handler()
retriever = rag_pipeline.get_retrieval_engine()
generator = rag_pipeline.get_generation_engine()
app = Flask(__name__)

@app.route('/get_config', methods=['GET'])
def get_config():
    response = {
        "status": "success",
        "config": rag_pipeline.config.to_dict()
    }
    return jsonify(response)

@app.route('/set_config', methods=['POST'])
def set_config():
    request_data = request.get_json()
    print(request_data)
    rag_pipeline.config.from_dict(request_data)
    response = {
        "status": "success",
        "config": rag_pipeline.config.to_dict()
    }
    return jsonify(response)

@app.route('/parse_pdf', methods=['POST'])
def parse_pdf():
    request_data = request.get_json()
    pdf_data = base64.b64decode(request_data['pdf'])
    # Save to a file
    pdf_file = os.path.join(rag_pipeline.config.output_directory, request_data["filename"])
    with open(pdf_file, 'wb') as pdf:
        pdf.write(pdf_data)
    if request_data.get("custom_feature_model"):
        feature_model = create_model(request_data.get("custom_feature_model"))
    else:
        feature_model = None
    response = pdf_parser.run_pipline(
        pdf_file=pdf_file,
        file_name=request_data["filename"],
        custom_extraction_prompt = request_data.get("custom_extraction_prompt"),
        custom_feature_model = feature_model,
    )
    return jsonify(response)

@app.route('/get_chunks', methods=['POST'])
def get_chunks():
    request_data = request.get_json()
    context = request_data.get('agentic_chunker_context', "")
    response = pdf_chunker.run_pipline(pdf_parser, agentic_chunker_context=context)
    return jsonify(response)

@app.route('/add_document_to_context', methods=['POST'])
def add_document_to_context():
    request_data = request.get_json()
    pdf_data = base64.b64decode(request_data['pdf'])
    # Save to a file
    pdf_file = os.path.join(rag_pipeline.config.output_directory, request_data["filename"])
    with open(pdf_file, 'wb') as pdf:
        pdf.write(pdf_data)
    parsed_pdf = pdf_parser.run_pipline(
        pdf_file=pdf_file,
        file_name=request_data["filename"],
        custom_extraction_prompt=request_data.get("custom_extraction_prompt"),
        custom_feature_model=request_data.get("custom_feature_model"),
    )
    context = request_data.get('agentic_chunker_context', "")
    chunks = pdf_chunker.run_pipline(pdf_parser, agentic_chunker_context=context)
    db_handler.insert_document(parsed_pdf)
    db_handler.batch_insert_embeddings(
        agentic_chunks=chunks["agentic_chunks"],
        semantic_chunks=chunks["semantic_chunks"]
    )
    return jsonify({"parsed_pdf": parsed_pdf, "chunks": chunks})

@app.route('/get_documents', methods=['GET'])
def get_documents():
    data = db_handler.get_documents()
    response = {
        "status": "success",
        "documents": {
            doc[0]: doc[1] for doc in data
        }
    }
    return jsonify(response)

@app.route('/get_document_by_name', methods=['POST'])
def get_document_by_name():
    request_data = request.get_json()
    filename = request_data.get('filename')
    data = db_handler.get_document_by_name(filename=filename)
    response = {
        "status": "success",
        "documents": {
            doc[0]: doc[1] for doc in data
        }
    }
    return jsonify(response)

@app.route('/get_context', methods=['POST'])
def get_context():
    request_data = request.get_json()
    query = request_data.get('query')
    response = retriever.get_context(query=query)
    print(response)
    return jsonify(response)

@app.route('/get_final_response', methods=['POST'])
def get_final_response():
    request_data = request.get_json()
    query = request_data.get('query')
    context = request_data.get('context')
    additional_instructions = request_data.get('additional_instructions')
    response = generator.generate_response(query=query, context=context, additional_instructions=additional_instructions)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

