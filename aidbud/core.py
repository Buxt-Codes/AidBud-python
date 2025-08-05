from .config import Config
from .utils import Context
from .workflow import Workflow
from typing import List 

class AidBud:
    def __init__(self):
        self.config = Config()
        self.context = Context()
        
    def initialise(self):
        self.workflow = Workflow(context=self.context, config=self.config)
    
    def run(self, conversation_id: int, query: str, attachment_paths: List[str] = None):
        return self.workflow.run(conversation_id=conversation_id, query=query, attachment_paths=attachment_paths)