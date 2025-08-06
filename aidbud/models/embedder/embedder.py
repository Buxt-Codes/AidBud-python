import os
import tiktoken
from typing import List, Dict, Union, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from ...config import Config
import numpy as np

class Embedder:
    def __init__(self, config: Config = Config()):
        self.embedding_model = SentenceTransformer(config.rag["embedder"])
        self.tokenizer = tiktoken.get_encoding(config.rag["tokeniser"])
        self.max_token_length = config.rag["embedder_max_tokens"]
        print(f"Initialised embedding model: {config.rag['embedder']} (max_tokens={self.max_token_length})")

    def _chunk_text(self, text: str, max_tokens: int = -1, overlap: int = 50) -> List[str]:
        if max_tokens == -1:
            max_tokens = self.max_token_length
        tokens = self.tokenizer.encode(text)
        chunks = []
        if len(tokens) <= max_tokens:
            return [text]

        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks

    def _chunk_response(self, response: Dict[str, str], max_tokens: int = -1, overlap: int = 50) -> List[str]:
        if max_tokens == -1:
            max_tokens = self.max_token_length

        query_tokens = response_tokens = pcard_tokens = []
        query_tokens = self.tokenizer.encode(str(response["query"]))
        if response.get("response"):
            response_tokens = self.tokenizer.encode(str(response["response"]))
        if response.get("pcard"):
            pcard_tokens = self.tokenizer.encode(str(response["pcard"]))

        if len(query_tokens) + len(response_tokens) + len(pcard_tokens) <= max_tokens:
            return [str(response)]

        chunks = []

        if response.get("response"):
            response_chunks = []
            for i in range(0, len(response_tokens), max_tokens - len(query_tokens) - 50 - overlap):
                chunk_tokens = response_tokens[i:i + max_tokens - len(query_tokens) - 50]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                response_chunks.append(chunk_text)
            for response_chunk in response_chunks:
                chunks.append(str({"query": response["query"], "response": response_chunk}))
        
        if response.get("pcard"):
            pcard_chunks = []
            for i in range(0, len(pcard_tokens), max_tokens - len(query_tokens) - 50 - overlap):
                chunk_tokens = pcard_tokens[i:i + max_tokens - len(query_tokens) - 50]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                pcard_chunks.append(chunk_text)
            for pcard in pcard_chunks:
                chunks.append(str({"query": response["query"], "pcard": pcard}))
        
        return chunks
    
    def _chunk_attachment(self, attachment: Dict[str, Any], max_tokens: int = -1, overlap: int = 50) -> List[str]:
        if max_tokens == -1:
            max_tokens = self.max_token_length
        description_tokens = self.tokenizer.encode(str(attachment["description"]))

        if len(description_tokens) <= max_tokens:
            return [self.tokenizer.decode(description_tokens)]

        chunks = []
        for i in range(0, len(description_tokens), max_tokens - overlap):
            chunk_tokens = description_tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def _get_embeddings(self, texts: Union[str, List[str]]) -> List[np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self.embedding_model.encode(texts).tolist()

            return [np.array(vec, dtype=np.float32) for vec in embeddings]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [] * len(texts)
    
    def embed_response(self, response: Dict[str, str]) -> Tuple[List[np.ndarray], List[str]]:
        response_chunks = self._chunk_response(response)
        embedded_text = self._get_embeddings(response_chunks)
        return embedded_text, response_chunks
    
    def embed_attachment(self, attachment: Dict[str, Any]) -> Tuple[List[np.ndarray], List[str]]:
        attachment_chunks = self._chunk_attachment(attachment)
        embedded_attachment = self._get_embeddings(attachment_chunks)
        return embedded_attachment, attachment_chunks

    def embed(self, text: str) -> Tuple[List[np.ndarray], List[str]]:
        chunks = self._chunk_text(text)
        embedded_text = self._get_embeddings(chunks)
        return embedded_text, chunks