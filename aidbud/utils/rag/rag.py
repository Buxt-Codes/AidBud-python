from models.embedder import embedder
from config import Config
import os
import tiktoken
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
import chromadb

class rag:
    def __init__(self, config: Config = Config()):
        self.chroma_client = chromadb.PersistentClient(path=config.rag["db_path"])
        self.response_collection = self.chroma_client.get_or_create_collection(
            name="text_queries"
        )
        self.attachment_collection = self.chroma_client.get_or_create_collection(
            name="attachment_queries"
        )
        self.embedder = embedder(config)
        print("RAG pipeline initialized")
    
    def _autonumber(collection) -> str:
        existing_ids = collection.get(include=[])["ids"]
        numeric_ids = [int(id_) for id_ in existing_ids if str(id_).isdigit()]
        next_id = (max(numeric_ids) + 1) if numeric_ids else 1
        return str(next_id)
    
    def insert_response(self, response: Dict[str, str], conversation_id: int):
        response_embeddings, response_chunks = self.embedder.embed_response(response)
        start_id = int(self._autonumber(self.text_collection))
        ids = [str(start_id + i) for i in range(len(response_embeddings))]
        if all(response_embeddings):
            self.text_collection.add(
                    embeddings=response_embeddings,
                    documents=response_chunks,
                    metadatas=[{"conversation_id": conversation_id} for i in response_embeddings],
                    ids=ids
                )
    
    def insert_attachment(self, attachment: Dict[str, Any], conversation_id: int):
        attachment_embeddings, attachment_chunks = self.embedder.embed_attachment(attachment)
        start_id = int(self._autonumber(self.attachment_collection))
        ids = [str(start_id + i) for i in range(len(attachment_embeddings))]
        if all(attachment_embeddings):
            self.text_collection.add(
                    embeddings=attachment_embeddings,
                    documents=attachment_chunks,
                    metadatas=[{"conversation_id": conversation_id, "paths": str(attachment["paths"])} for i in attachment_embeddings],
                    ids=ids
                )
            
