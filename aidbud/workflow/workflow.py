
from ..models import LLM
from ..utils import RAG, Context, PromptBuilder, Parser
from ..config import Config
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
import mimetypes
import ast
import os
import requests
import os
import tempfile
import urllib.request
from typing import List
from moviepy import VideoFileClip, AudioFileClip

class Workflow:
    def __init__(self, context: Context, config: Config = Config()):
        self.config = config
        self.context = context
        self.llm = LLM(config)
        self.rag = RAG(config)
        self.prompt_builder = PromptBuilder(context)
        self.parser = Parser()
    
    def classify_attachments(self, attachment_paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
        image_paths = []
        video_paths = []
        audio_paths = []

        if attachment_paths is None:
            return image_paths, video_paths, audio_paths

        for path in attachment_paths:
            parsed = urlparse(path)
            is_url = parsed.scheme in ["http", "https"]

            if is_url:
                try:
                    response = requests.head(path, allow_redirects=True, timeout=5)
                    if response.status_code >= 400:
                        print(f"[SKIPPED] URL does not exist or is inaccessible: {path}")
                        continue
                except requests.RequestException:
                    print(f"[SKIPPED] Failed to check URL: {path}")
                    continue
            else:
                if not os.path.exists(path):
                    print(f"[SKIPPED] File does not exist: {path}")
                    continue

            mime_type, _ = mimetypes.guess_type(parsed.path if is_url else path)
            if mime_type is None:
                print(f"[SKIPPED] Unknown MIME type: {path}")
                continue

            if mime_type.startswith("image/"):
                image_paths.append(path)
            elif mime_type.startswith("video/"):
                video_paths.append(path)
            elif mime_type.startswith("audio/"):
                audio_paths.append(path)
            else:
                print(f"[SKIPPED] Unsupported MIME type: {mime_type} - {path}")

        return image_paths, video_paths, audio_paths
    
    def run(self, conversation_id: int, query: str, attachment_paths: List[str] = None):
        response_ids, response_contexts = self.rag.retrieve_responses(query, conversation_id, self.config.rag["topK"])
        attachment_ids, attachment_contexts = self.rag.retrieve_attachments(query, conversation_id, self.config.rag["topK"])
        response_context = []
        attachment_context = []
        if response_contexts:
            response_context = response_contexts
        if attachment_contexts:
            attachment_context = [str({"attachment id": attachment_ids[i], "description": attachment_contexts[i]}) for i in range(len(attachment_ids))]

        if attachment_paths:
            output = self._query(conversation_id, query, response_context, attachment_context, attachment_paths)
        else:
            output = self._query_function(conversation_id, query, response_context, attachment_context)
        
        if output.get("error"):
            return {"error": output["error"]}
        
        response_object = {"query": query}
        if output.get("pcard"):
            response_object["pcard"] = output["pcard"]
        if output.get("response"):
            response_object["response"] = output["response"]
        self.rag.insert_response(response_object, conversation_id)

        return output
    
    def _query(self, conversation_id: int, query: str, conversation_context: List[str], attachment_context: List[str], attachment_paths: List[str] = None):
        if attachment_paths:
            attachment_description = self._attachment_processing(conversation_id, query, attachment_paths)
        else:
            attachment_description = None

        prompt = self.prompt_builder.query_prompt(query, attachment_description, conversation_context, attachment_context)
        response = self.llm.generate(prompt)
        parsed_response = self.parser.parse_response(response, find_function=False)

        if parsed_response:
            output = {}
            response = parsed_response["context"].get("RESPONSE")
            if isinstance(response, str):
                output["response"] = response
            pcard = parsed_response["context"]
            validated_pcard = self._valid_pcard(pcard)
            if isinstance(validated_pcard, dict):
                output["pcard"] = validated_pcard
            return output
        else:
            return {"error": "There was an error generating a response. Please try again."}

    def _query_function(self, conversation_id: int, query: str, conversation_context: List[str], attachment_context: List[str]):
        prompt = self.prompt_builder.query_function_prompt(query, conversation_context, attachment_context)
        response = self.llm.generate(prompt)
        parsed_response = self.parser.parse_response(response, find_function=True)

        if parsed_response:
            if parsed_response.get("type") == "fcall":
                fcall = parsed_response["context"]
                validated_fcall = self._valid_fcall(fcall)
                if validated_fcall:
                    return self._function(conversation_id, query, validated_fcall, conversation_context, attachment_context)
                else:
                    return self._query(conversation_id, query, conversation_context, attachment_context)

            elif parsed_response.get("type") == "response":
                output = {}
                response = parsed_response["context"].get("RESPONSE")
                if isinstance(response, str):
                    output["response"] = response
                pcard = parsed_response["context"]
                validated_pcard = self._valid_pcard(pcard)
                if isinstance(validated_pcard, dict):
                    output["pcard"] = validated_pcard
                return output
            
        return {"error": "There was an error generating a response. Please try again."}

    def _function(self, conversation_id: int, query: str, fcall: Dict[str, Any], conversation_context: List[str], attachment_context: List[str]):
        attachment_data = self.rag.get_attachment(fcall["id"])
        if attachment_data:
            attachment_paths = ast.literal_eval(attachment_data["metadata"]["paths"])
            attachment_description = attachment_data["document"]
            image_paths, video_paths, audio_paths = self.classify_attachments(attachment_paths)
            prompt = self.prompt_builder.function_prompt(query, attachment_description, conversation_context, attachment_context)
            response = self.llm.generate(prompt, image_paths, video_paths, audio_paths)
            parsed_response = self.parser.parse_response(response, find_function=False)

            if parsed_response:
                output = {}
                response = parsed_response["context"].get("RESPONSE")
                if isinstance(response, str):
                    output["response"] = response
                pcard = parsed_response["context"]
                validated_pcard = self._valid_pcard(pcard)
                if isinstance(validated_pcard, dict):
                    output["pcard"] = validated_pcard
                    if validated_pcard.get("ATTACHMENT"):
                        attachment_data = {"description": validated_pcard.get("ATTACHMENT"), "paths": attachment_paths}
                        self.rag.update_attachment(attachment_data, conversation_id)
                        del validated_pcard["ATTACHMENT"]
                return output
            else:
                return {"error": "There was an error generating a response. Please try again."}

        else:
            return self._query(conversation_id, query, conversation_context, attachment_context)


    def _valid_pcard(self, pcard: Dict[str, Any]) -> Dict[str, Any]:
        fields = {
            "TRIAGE": pcard.get("TRIAGE"),
            "INJURY IDENTIFICATION": pcard.get("INJURY IDENTIFICATION"),
            "INJURY DESCRIPTION": pcard.get("INJURY DESCRIPTION"),
            "PATIENT DESCRIPTION": pcard.get("PATIENT DESCRIPTION"),
            "INTERVENTION PLAN": pcard.get("INTERVENTION PLAN"),
            "ATTACHMENT": pcard.get("ATTACHMENT")
        }
        validated_pcard = {}
        for field, value in fields.items():
            if isinstance(value, str):
                validated_pcard[field] = value
        if validated_pcard:
            return validated_pcard
        return None
    
    def _valid_fcall(self, fcall: Dict[str, Any]) -> Dict[str, Any]:
        if fcall["type"] == "fcall":
            fields = {
                "id": fcall.get("id"),
                "remarks": fcall.get("remarks")
            }

            validated_fcall = {}
            if isinstance(fields["id"], int):
                validated_fcall["id"] = fields["id"]
            if isinstance(fields["remarks"], str):
                validated_fcall["remarks"] = fields["remarks"]

            if len(validated_fcall) == 2:
                return validated_fcall
        return None

    def _attachment_processing(self, conversation_id: int, query: str, attachment_paths: List[str] = None):
        if attachment_paths:
            image_paths, video_paths, audio_paths = self.classify_attachments(attachment_paths)
            prompt = self.prompt_builder.attachment_prompt(query)
            response = self.llm.generate(prompt, image_paths, video_paths, audio_paths)
            description = self.parser.parse_attachment_response(response)
            
            if description:
                attachment_data = {"description": description, "paths": attachment_paths}
                self.rag.insert_attachment(attachment_data, conversation_id)
            
            return description
        return None