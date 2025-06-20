import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


class AzureOpenAIChatClient:
    def __init__(self, api_endpoint: str, api_key: str, model: str, api_version: str|None = None):
        self.model = model
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=api_endpoint,
            api_version=api_version
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

"""class AzureOpenAIEmbeddings:
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
        return dict(zip(input_phrases, embeddings))"""
