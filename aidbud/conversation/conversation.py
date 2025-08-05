from typing import List, Dict

class Conversation:
    def __init__(self):
        self.current_conversation = 0
        self.reset()
    
    def reset(self):
        self.current_conversation += 1
        self.messages = []
        self.pcard = {}
    
    def add_message(self, message, role="user", attachment_paths = List[str]):
        self.messages.append({"role": role, "content": message, "attachment_paths": attachment_paths})
    
    def get_messages(self):
        return self.messages
    
    def add_pcard(self, pcard):
        self.pcard = pcard
    
    def get_pcard(self):
        return self.pcard
    
    def update_pcard(self, pcard: Dict[str, str]):
        self.pcard.update(pcard)