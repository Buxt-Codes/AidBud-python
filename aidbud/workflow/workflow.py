
from ..models import LLM
from ..utils import RAG, Context, PromptBuilder, Parser
from ..config import Config
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
import mimetypes
import ast

class Workflow:
    def __init__(self, context: Context, config: Config = Config()):
        self.context = context
        self.llm = LLM(config)
        self.rag = RAG(config)
        self.prompt_builder = PromptBuilder(context)
        self.parser = Parser()
    
    def _classify_attachments(self, attachment_paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
        image_paths = []
        video_paths = []
        audio_paths = []

        for path in attachment_paths:
            parsed = urlparse(path)
            if parsed.scheme in ["http", "https"]:
                mime_type, _ = mimetypes.guess_type(parsed.path)
            else:
                mime_type, _ = mimetypes.guess_type(path)

            if mime_type is None:
                continue  

            if mime_type.startswith("image/"):
                image_paths.append(path)
            elif mime_type.startswith("video/"):
                video_paths.append(path)
            elif mime_type.startswith("audio/"):
                audio_paths.append(path)

        return image_paths, video_paths, audio_paths
    
    def run(self, conversation_id: int, query: str, attachment_paths: List[str] = None):
        response_ids, response_contexts = self.rag.retrieve_responses(query, conversation_id, self.config.rag["topK"])
        attachment_ids, attachment_contexts = self.rag.retrieve_attachments(query, conversation_id, self.config.rag["topK"])
        response_context = response_contexts
        attachment_context = [str({"id": attachment_ids[i], "description": attachment_contexts[i]}) for i in range(len(attachment_ids))]

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
        self.rag.add_response(response_object, conversation_id)

        return output
    
    def _query(self, conversation_id: int, query: str, conversation_context: List[Dict[str, Any]], attachment_context: List[Dict[str, Any]], attachment_paths: List[str] = None):
        if attachment_paths:
            attachment_description = self._attachment_processing(conversation_id, query, attachment_paths)
        else:
            attachment_description = None

        prompt = self.prompt_builder.query_prompt(query, attachment_description, conversation_context, attachment_context)
        response = self.llm.generate(prompt)
        parsed_response = self.parser.parse_response(response, find_function=False)

        output = {}
        for response_data in parsed_response:
            if response_data["type"] == "pcard":
                pcard = response_data["context"]
                validated_pcard = self._valid_pcard(pcard)
                output["pcard"] = validated_pcard
            
            if response_data["type"] == "response":
                llm_response = response_data["response"]
                output["response"] = llm_response

        if validated_pcard:
            return output
        else:
            return {"error": "There was an error generating a response. Please try again."}

    def _query_function(self, conversation_id: int, query: str, conversation_context: List[Dict[str, Any]], attachment_context: List[Dict[str, Any]]):
        prompt = self.prompt_builder.query_function_prompt(query, conversation_context, attachment_context)
        response = self.llm.generate(prompt)
        parsed_response = self.parser.parse_response(response, find_function=True)

        output = {}
        for response_data in parsed_response:
            if response_data["type"] == "fcall":
                fcall = response_data["context"]
                validated_fcall = self._valid_fcall(fcall)
                if validated_fcall:
                    return self._function(conversation_id, query, validated_fcall, conversation_context, attachment_context)
                else:
                    return self._query(conversation_id, query, conversation_context, attachment_context)

            if response_data["type"] == "pcard":
                pcard = response_data["context"]
                validated_pcard = self._valid_pcard(pcard)
                output["pcard"] = validated_pcard
            
            if response_data["type"] == "response":
                llm_response = response_data["response"]
                output["response"] = llm_response
        
        if validated_pcard:
            return output
        return {"error": "There was an error generating a response. Please try again."}

    def _function(self, conversation_id, int, query: str, fcall: Dict[str, Any], conversation_context: List[Dict[str, Any]], attachment_context: List[Dict[str, Any]]):
        attachment_data = self.rag.get_attachment(fcall["id"])
        if attachment_data:
            attachment_paths = ast.literal_eval(attachment_data["metadata"]["paths"])
            attachment_description = attachment_data["document"]
            image_paths, video_paths, audio_paths = self._classify_attachments(attachment_paths)
            prompt = self.prompt_builder.function_prompt(query, attachment_description, conversation_context, attachment_context)
            response = self.llm.generate(prompt, image_paths, video_paths, audio_paths)
            parsed_response = self.parser.parse_response(response, find_function=False)

            output = {}
            for response_data in parsed_response:
                if response_data["type"] == "pcard":
                    pcard = response_data["context"]
                    validated_pcard = self._valid_pcard(pcard)
                    output["pcard"] = validated_pcard
                
                if response_data["type"] == "response":
                    llm_response = response_data["response"]
                    output["response"] = llm_response

            if validated_pcard:
                return output
            return {"error": "There was an error generating a response. Please try again."}

        else:
            return self._query(conversation_id, query, conversation_context, attachment_context)


    def _valid_pcard(self, pcard: Dict[str, Any]) -> Dict[str, Any]:
        if pcard["type"] == "pcard":
            fields = {
                "TRIAGE": pcard.get("TRIAGE"),
                "IDENTIFIED_INJURY": pcard.get("IDENTIFIED_INJURY"),
                "IDENTIFIED_INJURY_DESCRIPTION": pcard.get("IDENTIFIED_INJURY_DESCRIPTION"),
                "PATIENT_INJURY_DESCRIPTION": pcard.get("PATIENT_INJURY_DESCRIPTION"),
                "INTERVENTION_PLAN": pcard.get("INTERVENTION_PLAN"),
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
            image_paths, video_paths, audio_paths = self._classify_attachments(attachment_paths)
            prompt = self.prompt_builder.attachment_prompt(query)
            response = self.llm.generate(prompt, image_paths, video_paths, audio_paths)
            description = self.parser.parse_attachment_response(response)
            
            if description:
                attachment_data = {"description": description, "paths": attachment_paths}
                self.rag.insert_attachment(attachment_data, conversation_id)
            
            return description
        return None