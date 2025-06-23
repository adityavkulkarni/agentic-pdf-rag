import os

from agentic_pdf_rag import RAGPipeline

rag_pipeline = RAGPipeline(
    config_file=os.path.join(os.getcwd(), "config", "config.ini"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_embeddings_api_key=os.getenv("AZURE_OPENAI_API_KEY_EAST2")
)


"""parsed_pdf = rag_pipeline.parse_pdf(pdf_path="uts_to_h.pdf")
chunks = rag_pipeline.create_chunks(parsed_pdf)
rag_pipeline.add_document_to_db(parsed_pdf, chunks)

# Direct process and add to db
rag_pipeline.add_document_to_knowledge(pdf_path="uts_to_h.pdf")"""

context = rag_pipeline.retrieve_context(query="Compare the Production Sharing Agreement (PSA) #2906 and Production Sharing Agreement (PSA) #2905")
response = rag_pipeline.generate_response(
            query="Compare the Production Sharing Agreement (PSA) #2906 and Production Sharing Agreement (PSA) #2905",
            context=context,
            additional_instructions='additional_instructions'
)
import pprint
pprint.pprint(response)
# Direct generate_response
# response = rag_pipeline.get_final_response(query="This is a sample query", additional_instructions='additional_instructions')
# print(response)