from typing import List
from ..context import Context

class PromptBuilder:
    def __init__(
        self,
        context: Context
    ):
        self.current_situation = context.currentsituation_context
        self.first_aid_available = context.firstaidavail_context
        self.triage = context.triage_context
    
    def insert_query(self, prompt: str, query: str) -> str:
        if query:
            prompt = prompt.replace("[QUERY]", f"\n** Query:**\n{query}\n""", 1)
        else:
            prompt = prompt.replace("[QUERY]", "", 1)
        return prompt

    def insert_triage(self, prompt: str) -> str:
        if self.triage.enabled:
            prompt = prompt.replace("[TRIAGE]", f"\n**Triage:**\n{self.triage.protocol}\n", 1)
        else:
            prompt = prompt.replace("[TRIAGE]", "", 1)
        return prompt

    def insert_first_aid(self, prompt: str) -> str:
        if self.first_aid_available.enabled:
            prompt = prompt.replace("[FIRST AID AVAILABILITY]", f"\n**First Aid:**\nWhere IMMEDIATE means basic first aid is readily available, NON-IMMEDIATE means basic first aid is not readily available, and UNAVAILABLE means basic first aid is not available.\n{self.first_aid_available.protocol}\n", count=1)
        else:
            prompt = prompt.replace("[FIRST AID AVAILABILITY]", "", 1)
        return prompt
    
    def insert_current_situation(self, prompt: str) -> str:
        if self.current_situation.enabled:
            prompt = prompt.replace("[CURRENT SITUATION]", f"\n**Current Situation:**\n{self.current_situation.situation}\n", 1)
        else:
            prompt = prompt.replace("[CURRENT SITUATION]", "", 1)
        return prompt
    
    def insert_attachment_description(self, prompt: str, attachment_description: str = None) -> str:
        if attachment_description:
            prompt = prompt.replace("[ATTACHMENT DESCRIPTION]", f"\n**Attachment Description:**\n{attachment_description}\n", 1)
        else:
            prompt = prompt.replace("[ATTACHMENT DESCRIPTION]", "", 1)
        return prompt

    def insert_conversation_context(self, prompt: str, response_context: List[str] = None, attachment_context: List[str] = None) -> str:
        response_context_section = None
        attachment_context_section = None
        if response_context:
            response_context_section = "\n**Relevant past conversation history:**\n" + "\n".join(response_context)
        
        if attachment_context:
            attachment_context_section = "\n**Relevant context from past attachments:**\n" + "\n".join(attachment_context)
        
        if response_context_section and attachment_context_section:
            prompt = prompt.replace("[CONVERSATION CONTEXT]", f"{response_context_section}\n{attachment_context_section}\n", 1)
        elif response_context_section:
            prompt = prompt.replace("[CONVERSATION CONTEXT]", f"{response_context_section}\n", 1)
        elif attachment_context_section:
            prompt = prompt.replace("[CONVERSATION CONTEXT]", f"{attachment_context_section}\n", 1)
        else:
            prompt = prompt.replace("[CONVERSATION CONTEXT]", "", 1)
        return prompt

    def attachment_prompt(self, query: str = None) -> str:
        with open(r"aidbud\utils\prompt\templates\attachment.txt", "r", encoding="utf-8") as file:
            prompt = file.read()
        return prompt
    
    def query_function_prompt(self, query: str = None, response_context: List[str] = None, attachment_context: List[str] = None) -> str:
        if self.triage.enabled:
            with open(r"aidbud\utils\prompt\templates\triage_query_function.txt", "r", encoding="utf-8") as file:
                prompt = file.read()
                prompt = self.insert_triage(prompt)
        else:
            with open(r"aidbud\utils\prompt\templates\query_function.txt", "r", encoding="utf-8") as file:
                prompt = file.read()
        prompt = self.insert_query(prompt, query)
        prompt = self.insert_conversation_context(prompt, response_context, attachment_context)
        prompt = self.insert_first_aid(prompt)
        prompt = self.insert_current_situation(prompt)
        return prompt
    
    def function_prompt(self, query: str = None, attachment_description: str = None, response_context: List[str] = None, attachment_context: List[str] = None) -> str:
        if self.triage.enabled:
            with open(r"aidbud\utils\prompt\templates\triage_function.txt", "r", encoding="utf-8") as file:
                prompt = file.read()
                prompt = self.insert_triage(prompt)
        else:
            with open(r"aidbud\utils\prompt\templates\function.txt", "r", encoding="utf-8") as file:
                prompt = file.read()
        prompt = self.insert_query(prompt, query)
        prompt = self.insert_attachment_description(prompt, attachment_description)
        prompt = self.insert_conversation_context(prompt, response_context, attachment_context)
        prompt = self.insert_first_aid(prompt)
        prompt = self.insert_current_situation(prompt)
        return prompt
    
    def query_prompt(self, query: str = None, attachment_description: str = None, response_context: List[str] = None, attachment_context: List[str] = None) -> str:
        if self.triage.enabled:
            with open(r"aidbud\utils\prompt\templates\triage_query.txt", "r", encoding="utf-8") as file:
                prompt = file.read()
                prompt = self.insert_triage(prompt)
        else:
            with open(r"aidbud\utils\prompt\templates\query.txt", "r", encoding="utf-8") as file:
                prompt = file.read()
        prompt = self.insert_query(prompt, query)
        prompt = self.insert_attachment_description(prompt, attachment_description)
        prompt = self.insert_conversation_context(prompt, response_context, attachment_context)
        prompt = self.insert_first_aid(prompt)
        prompt = self.insert_current_situation(prompt)
        return prompt
