import os
import io
import base64
import time
import requests
import torch
import cv2
import numpy as np
import soundfile as sf
from PIL import Image
from typing import List, Dict, Any, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import json
import re
from ..utils.context import CurrentSituationContext, FirstAidAvailableContext, TriageContext
from ..utils.rag.rag import RAG

class PromptBuilder:
    def __init__(
        self, 
        rag: RAG,
        current_situation: CurrentSituationContext, 
        first_aid_available: FirstAidAvailableContext, 
        triage: TriageContext
    ):
        self.current_situation = current_situation
        self.first_aid_available = first_aid_available
        self.triage = triage
    
    def insert_query(self, prompt: str, query: str) -> str:
        if query:
            prompt = prompt.replace("[QUERY]", f"\n** Query:**\n{query}\n""")
        else:
            prompt = prompt.replace("[QUERY]", "")
        return prompt

    def insert_triage(self, prompt: str) -> str:
        if self.triage.enabled:
            prompt = prompt.replace("[TRIAGE]", f"\n**Triage:**\n{self.triage.protocol}\n")
        else:
            prompt = prompt.replace("[TRIAGE]", "")
        return prompt

    def insert_first_aid(self, prompt: str) -> str:
        if self.first_aid_available.enabled:
            prompt = prompt.replace("[FIRST AID AVAILABILITY]", f"\n**First Aid:**\nWhere IMMEDIATE means basic first aid is readily available, NON-IMMEDIATE means basic first aid is not readily available, and UNAVAILABLE means basic first aid is not available.\n{self.first_aid_available.protocol}\n")
        else:
            prompt = prompt.replace("[FIRST AID AVAILABILITY]", "")
        return prompt
    
    def insert_current_situation(self, prompt: str) -> str:
        if self.current_situation.enabled:
            prompt = prompt.replace("[CURRENT SITUATION]", f"\n**Current Situation:**\n{self.current_situation.protocol}\n")
        else:
            prompt = prompt.replace("[CURRENT SITUATION]", "")
        return prompt
    
    def insert_attachment_description(self, prompt: str, attachment_description: str = None) -> str:
        if attachment:
            prompt = prompt.replace("[ATTACHMENT DESCRIPTION]", f"\n**Attachment Description:**\n{attachment_description}\n")
        else:
            prompt = prompt.replace("[ATTACHMENT DESCRIPTION]", "")
        return prompt

    def insert_conversation_context(self, prompt: str, response_context: List[str] = None, attachment_context: List[str] = None) -> str:
        if response_context:
            response_context_section = "\n**Relevant past conversation history:**\n" + "\n".join(response_context)
        
        if attachment_context:
            attachment_context_section = "\n**Relevant context from past attachments:**\n" + "\n".join(attachment_context)
        
        if response_context_section and attachment_context_section:
            prompt = prompt.replace("[CONVERSATION CONTEXT]", f"{response_context_section}\n{attachment_context_section}\n")
        elif response_context_section:
            prompt = prompt.replace("[CONVERSATION CONTEXT]", f"{response_context_section}\n")
        elif attachment_context_section:
            prompt = prompt.replace("[CONVERSATION CONTEXT]", f"{attachment_context_section}\n")
        else:
            prompt = prompt.replace("[CONVERSATION CONTEXT]", "")
        return prompt

    def attachment_prompt(self, query: str = None) -> str:
        with open("attachment_prompt.txt", "r", encoding="utf-8") as file:
            prompt = file.read()
        prompt = self.insert_query(prompt, query)
        return prompt
    
    def query_function_prompt(self, query: str = None, response_context: List[str] = None, attachment_context: List[str] = None) -> str:
        if self.triage.enabled:
            with open("query_function_triage_prompt.txt", "r", encoding="utf-8") as file:
                prompt = file.read()
                prompt = self.insert_triage(prompt)
        else:
            with open("query_function_prompt.txt", "r", encoding="utf-8") as file:
                prompt = file.read()
        prompt = self.insert_query(prompt, query)
        prompt = self.insert_conversation_context(prompt, response_context, attachment_context)
        prompt = self.insert_first_aid(prompt)
        prompt = self.insert_current_situation(prompt)
        return prompt
    
    def function_prompt(self, query: str = None, attachment_description: str = None, response_context: List[str] = None, attachment_context: List[str] = None) -> str:
        with open("function_prompt.txt", "r", encoding="utf-8") as file:
            prompt = file.read()

    def build_query_prompt(self, query: str, attachment_description: str = None, attachment_transcription: str = None, rag_text_context: str = None, rag_attachment_context: str = None) -> str:
        prompt = f"User Query: {query}\n\n"
        if attachment_description:
            prompt += f"Context from new attachments (Description): {attachment_description}\n"
        if attachment_transcription:
            prompt += f"Context from new attachments (Transcription): {attachment_transcription}\n"
        if rag_text_context:
            prompt += f"Relevant past conversation history: {rag_text_context}\n"
        if rag_attachment_context:
            prompt += f"Relevant context from past attachments: {rag_attachment_context}\n"
        prompt += "\nBased on all the provided information, please provide a comprehensive answer to the user's query."
        return prompt