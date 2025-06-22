import requests
import base64


class RAGClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_config(self):
        """Retrieve current configuration"""
        response = requests.get(f"{self.base_url}/get_config")
        return response.json()

    def set_config(self, config_dict):
        """Update configuration"""
        response = requests.post(f"{self.base_url}/set_config", json=config_dict)
        return response.json()

    def parse_pdf(self, pdf_path, filename, custom_extraction_prompt=None, custom_feature_model=None):
        """Process PDF file through parsing pipeline"""
        with open(pdf_path, "rb") as pdf_file:
            pdf_base64 = base64.b64encode(pdf_file.read()).decode("utf-8")

        payload = {
            "pdf": pdf_base64,
            "filename": filename
        }
        if custom_extraction_prompt:
            payload["custom_extraction_prompt"] = custom_extraction_prompt
        if custom_feature_model:
            payload["custom_feature_model"] = custom_feature_model

        response = requests.post(f"{self.base_url}/parse_pdf", json=payload)
        return response.json()

    def get_chunks(self, agentic_chunker_context):
        """Retrieve document chunks"""
        payload = {"agentic_chunker_context": agentic_chunker_context}
        response = requests.post(f"{self.base_url}/get_chunks", json=payload)
        return response.json()

    def add_document_to_context(self, pdf_path, filename, agentic_chunker_context=None, custom_extraction_prompt=None,
                                custom_feature_model=None):
        """Add document to processing context"""
        with open(pdf_path, "rb") as pdf_file:
            pdf_base64 = base64.b64encode(pdf_file.read()).decode("utf-8")

        payload = {
            "pdf": pdf_base64,
            "filename": filename
        }
        if agentic_chunker_context:
            payload["agentic_chunker_context"] = agentic_chunker_context
        if custom_extraction_prompt:
            payload["custom_extraction_prompt"] = custom_extraction_prompt
        if custom_feature_model:
            payload["custom_feature_model"] = custom_feature_model

        response = requests.post(f"{self.base_url}/add_document_to_context", json=payload)
        return response.json()

    def get_documents(self):
        """Retrieve all stored documents"""
        response = requests.get(f"{self.base_url}/get_documents")
        return response.json()

    def get_document_by_name(self, filename):
        """Retrieve document by filename"""
        payload = {"filename": filename}
        response = requests.post(f"{self.base_url}/get_document_by_name", json=payload)
        return response.json()

    def get_context(self, query):
        """Retrieve context for a query"""
        payload = {"query": query}
        response = requests.post(f"{self.base_url}/get_context", json=payload)
        return response.json()

    def get_final_response(self, query, context, additional_instructions=None):
        """Generate final response using context"""
        payload = {
            "query": query,
            "context": context
        }
        if additional_instructions:
            payload["additional_instructions"] = additional_instructions

        response = requests.post(f"{self.base_url}/get_final_response", json=payload)
        return response.json()


if __name__ == '__main__':
    client = RAGClient("http://localhost:5000")

    # Query the system
    context = client.get_context(query="Summarize the production sharing agreement")
    response = client.get_final_response(
        query="Summarize the production sharing agreement",
        context=f"Chunks from contract:\n{'\n\n'.join([r['content'] for r in context])}\n\n",
        additional_instructions=(
            "You are a legal language model. Use the following context retrieved from a database to answer the "
            "user’s legal question or draft/review contract language. If relevant, quote or paraphrase the provided context."
        )
    )
    print(response)
