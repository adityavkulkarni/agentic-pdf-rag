import gc
import logging
import torch
import types
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

torch.classes.__path__ = []
torch.classes._path = types.SimpleNamespace()
logger = logging.getLogger(__name__)


class Qwen3RetrievalSystem:
    def __init__(self, embed_model_name="Qwen/Qwen3-Embedding-0.6B",
                 rerank_model_name="tomaarsen/Qwen3-Reranker-0.6B-seq-cls"):
        """
        Initialize Qwen3 embedding and reranking models
        """
        if torch.cuda.device_count() == 2:
            self.device = torch.device("cuda:1")
            # Limit to 50% of GPU 1's memory
        else:
            self.device = torch.device("cuda:0")
            # Limit to 50% of GPU 1's memory
        logger.info(f"Using device: {self.device}")
        # Initialize embedding model
        self.embed_model = SentenceTransformer(
            embed_model_name,
            model_kwargs={
                "attn_implementation": None,
            },
            tokenizer_kwargs={"padding_side": "left"},
            device=self.device,
        )

        # Initialize reranking model
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
        """self.rerank_model = AutoModel.from_pretrained(rerank_model_name,
                                                      torch_dtype=torch.float16)"""

    def create_embeddings(self, input_phrases: list[str], prompt_name="query"):
        """
        Create embeddings in OpenAI-compatible format.
        """
        # Generate embeddings for all input phrases
        embeddings = self.embed_model.encode(
            input_phrases,
            prompt_name=prompt_name,
            batch_size=4,
            device=self.device,
        )
        # Format output to match OpenAI's API
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [emb.tolist() for emb in embeddings]

    def create_embedding_dict(self, input_phrases: list[str], prompt_name="query"):
        # Get the embeddings response
        response = self.create_embeddings(input_phrases=input_phrases, prompt_name=prompt_name)
        # Extract the embeddings from the response
        #embeddings = [item["embedding"] for item in response.data]
        # Map each phrase to its embedding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return dict(zip(input_phrases, response))

    '''def rerank_documents(self, query, stored_data, task_description="Retrieve relevant documents"):
        """
        Rerank documents based on relevance to query using stored metadata
        """
        # Prepare input texts with task context
        inputs = [
            f"Task: {task_description}\nQuery: {query}\nDocument: {item['text']}"
            for item in stored_data
        ]

        # Tokenize and move inputs to model's device
        encoded_inputs = self.rerank_tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)  # Move all tensor values to GPU

        # Ensure model is on correct device
        self.rerank_model = self.rerank_model.to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.rerank_model(**encoded_inputs)

        # Calculate relevance scores from CLS embeddings
        scores = torch.norm(outputs.last_hidden_state[:, 0, :], dim=1)

        # Attach scores and sort
        for i, item in enumerate(stored_data):
            item["score"] = scores[i].item()

        return sorted(stored_data, key=lambda x: x["score"], reverse=True)'''


if __name__ == "__main__":
        embedding_model = Qwen3RetrievalSystem()
        print(embedding_model.create_embeddings(["This is an example", "This is an example"]))