from ...models import Embedder
from ...config import Config
import os
import tiktoken
from typing import List, Dict, Union, Any, Tuple
from sentence_transformers import SentenceTransformer
import chromadb

class RAG:
    def __init__(self, config: Config = Config()):
        self.chroma_client = chromadb.PersistentClient(path=config.rag["db_path"])
        self.response_collection = self.chroma_client.get_or_create_collection(
            name="text_queries"
        )
        self.attachment_collection = self.chroma_client.get_or_create_collection(
            name="attachment_queries"
        )
        self.embedder = Embedder(config)
        print("RAG pipeline initialized")
    
    def _autonumber(self, collection) -> str:
        existing_ids = collection.get(include=[])["ids"]
        numeric_ids = [int(id_) for id_ in existing_ids if str(id_).isdigit()]
        next_id = (max(numeric_ids) + 1) if numeric_ids else 1
        return str(next_id)
    
    def insert_response(self, response: Dict[str, str], conversation_id: int):
        response_embeddings, response_chunks = self.embedder.embed_response(response)
        start_id = int(self._autonumber(self.response_collection))
        ids = [str(start_id + i) for i in range(len(response_embeddings))]
        if all(response_embeddings):
            self.response_collection.add(
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
            self.attachment_collection.add(
                    embeddings=attachment_embeddings,
                    documents=attachment_chunks,
                    metadatas=[{"conversation_id": conversation_id, "paths": str(attachment["paths"])} for i in attachment_embeddings],
                    ids=ids
                )
    
    def get_conversation_responses(self, conversation_id: int) -> Tuple[List[str], List[str]]:
        result = self.response_collection.get(
            where={
                "conversation_id": conversation_id
            }
        )
        ids = result.get("ids", [])
        documents = result.get("documents") or []
        return ids, documents

    def get_conversation_attachments(self, conversation_id: int) -> Tuple[List[str], List[str]]:
        result = self.attachment_collection.get(
            where={
                "conversation_id": conversation_id
            }
        )
        ids = result.get("ids", [])
        documents = result.get("documents") or []
        return ids, documents
    
    def get_response(self, response_id: int) -> Dict[str, Any]:
        result = self.response_collection.get(ids=[str(response_id)])
        if result is not None and result.get("ids"):
            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0]
            }
        return {}
    
    def get_attachment(self, attachment_id: int) -> Dict[str, Any]:
        result = self.attachment_collection.get(ids=[str(attachment_id)])
        if result is not None and result.get("ids"):
            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0]
            }
        return {}

    def retrieve_responses(self, query: str, conversation_id: int, k: int = 5) -> Tuple[List[str], List[str]]:
        if not query:
            result = self.response_collection.get(
                where={"conversation_id": conversation_id},
                limit=k
            )
            ids_raw = result.get("ids") or []
            ids = [str(i) for i in ids_raw]
            nested_documents = result.get("documents") or []
            flat_documents = [doc for sublist in nested_documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
            documents = [doc if isinstance(doc, str) else str(doc) for doc in flat_documents]
            return ids, documents

        query_embeddings, query_chunks = self.embedder.embed(query)
        result = self.response_collection.query(
            query_embeddings=query_embeddings[0],
            n_results=k,
            where={
                "conversation_id": conversation_id
            }
        )
        ids_raw = result.get("ids") or []
        ids = [str(i) for i in ids_raw]
        nested_documents = result.get("documents") or []
        flat_documents = [doc for sublist in nested_documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
        documents = [doc if isinstance(doc, str) else str(doc) for doc in flat_documents]

        return ids, documents

    def retrieve_attachments(self, query: str, conversation_id: int, k: int = 5) -> Tuple[List[str], List[str]]:
        if not query:
            result = self.attachment_collection.get(
                where={"conversation_id": conversation_id},
                limit=k
            )
            ids_raw = result.get("ids") or []
            ids = [str(i) for i in ids_raw]
            nested_documents = result.get("documents") or []
            flat_documents = [doc for sublist in nested_documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
            documents = [doc if isinstance(doc, str) else str(doc) for doc in flat_documents]
            return ids, documents

        query_embeddings, query_chunks = self.embedder.embed(query)
        result = self.attachment_collection.query(
            query_embeddings=query_embeddings[0],
            n_results=k,
            where={
                "conversation_id": conversation_id
            }
        )
        ids_raw = result.get("ids") or []
        ids = [str(i) for i in ids_raw]
        nested_documents = result.get("documents") or []
        flat_documents = [doc for sublist in nested_documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
        documents = [doc if isinstance(doc, str) else str(doc) for doc in flat_documents]

        return ids, documents
    
    def delete_conversation(self, conversation_id: int):
        self.response_collection.delete(where={"conversation_id": conversation_id})
        self.attachment_collection.delete(where={"conversation_id": conversation_id})

    def reset_collections(self):
        try:
            self.chroma_client.delete_collection("text_queries")
        except Exception:
            pass  

        try:
            self.chroma_client.delete_collection("attachment_queries")
        except Exception:
            pass  

        self.response_collection = self.chroma_client.get_or_create_collection(
            name="text_queries"
        )
        self.attachment_collection = self.chroma_client.get_or_create_collection(
            name="attachment_queries"
        )