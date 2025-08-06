from typing import List, Dict
from IPython.display import Markdown, display

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
        allowed_keys = {
            "TRIAGE",
            "INJURY IDENTIFICATION",
            "INJURY DESCRIPTION",
            "PATIENT DESCRIPTION",
            "INTERVENTION PLAN"
        }

        for key in pcard:
            if key not in allowed_keys:
                print(f"Warning: '{key}' is not a recognized Patient Card field and will be ignored.")

        filtered_pcard = {k: v for k, v in pcard.items() if k in allowed_keys}
        self.pcard.update(filtered_pcard)
    
    def display_pcard(self):
        if self.pcard:
            lines = []
            for key, value in self.pcard.items():
                lines.append(f">#### **{key}**")
                for line in str(value).splitlines():
                    lines.append(f">#### {line}")
                lines.append(">####")  
                lines.append(">####")  
            while lines and lines[-1].strip() == ">####":
                lines.pop()
            if lines:
                display(Markdown("## Patient Card Details"))
                display(Markdown(f"### 📝\n" + "\n".join(lines)))
        else:
            print("No Patient Card to show.")