import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


class AzureOpenAIChatClient:
    def __init__(self, api_endpoint: str, api_key: str, model: str, api_version: str|None = None):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.api_version = api_version
        self.model = model
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_endpoint,
            api_version=self.api_version
        )

    def chat_completion(
        self,
        text,
        feature_model=None,
        seed=77,
        temperature=0,
        stream=False,
        image_url=None,
        image_base64=None
    ):
        if type(text) == str:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
            if image_url:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            elif image_base64:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
        else:
            messages = text
        response_format = None
        if feature_model:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "Extraction_Response",
                    "strict": True,
                    "schema": feature_model.model_json_schema()
                }
            }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
            seed=seed,
            temperature=temperature,
            stream=stream
        )
        return response


class AzureOpenAIEmbeddings:
    def __init__(self, endpoint: str, api_key: str, model: str, api_version: str|None = None):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key
        )
        self.model = model

    def create_embeddings(self, input_phrases: list[str]):
        return self.client.embeddings.create(
            input=input_phrases,
            model=self.model
        )

    def create_embedding_dict(self, input_phrases: list[str]):
        # Get the embeddings response
        response = self.client.embeddings.create(
            input=input_phrases,
            model=self.model
        )
        # Extract the embeddings from the response
        embeddings = [item.embedding for item in response.data]
        # Map each phrase to its embedding
        return dict(zip(input_phrases, embeddings))


if __name__ == "__main__":
    text = """
    Summarize below text and identify all the named entities. Follow the response format. 
    
    Dr. Emily Carter recently published a groundbreaking research paper in the journal Nature, following a study conducted in collaboration with scientists from Stanford University. Microsoft has announced plans to invest $2 billion in sustainable technology by the year 2030. Paris, the capital of France, is renowned for its historic landmarks such as the Eiffel Tower. Last month, Elon Musk, CEO of Tesla, spoke at the World Economic Forum. These developments highlight the ongoing advancements in both science and technology across the globe.
    """
    import pprint
    client = AzureOpenAIEmbeddings(model="text-embedding-3-large",
                                   api_key="",
                                   endpoint="https://ul-openai-finetune-dev.openai.azure.com/",
                                   api_version="2024-02-01")
    response = client.create_embedding_dict(text)
    pprint.pprint(response)
