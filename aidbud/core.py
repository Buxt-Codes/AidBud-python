from .config import Config
from .utils import Context
from .workflow import Workflow
from .conversation import Conversation
from typing import List 
from IPython.display import Audio, Image, Video, Markdown, display

class AidBud:
    def __init__(self):
        self.config = Config()
        self.context = Context()
        self.conversation = Conversation()
        
    def initialise(self):
        self.workflow = Workflow(context=self.context, config=self.config)
        self.workflow.rag.reset_collections() # RESET COLLECTIONS FOR NOW
        self.context.reset() # RESET CONTEXT FOR NOW TOO
    
    def new_conversation(self):
        self.conversation.reset()
    
    def reset(self):
        self.initialise()
        self.new_conversation()
    
    def query(self, query: str = None, attachment_paths: List[str] = None):
        if query == None and attachment_paths == None:
            raise ValueError("At least query or attahment_paths must be provided.")
        
        video_paths, audio_paths, image_paths = self.workflow.classify_attachments(attachment_paths)
        if query == None and video_paths == None and audio_paths == None and image_paths == None:
            raise ValueError("At least query or valid attachments must be provided.")
        
        output = self.workflow.run(conversation_id=self.conversation.current_conversation, query=query, attachment_paths=attachment_paths)

        if output.get("error"):
            display(Markdown("## Error"))
            display(Markdown(f"### ❌\n#### {output["error"]}"))
        
        display(Markdown("## Conversation Details"))

        for video_path in video_paths:
            display(Video(video_path))
        for audio_path in audio_paths:
            display(Audio(audio_path))
        for image_path in image_paths:
            display(Image(image_path))
        
        if query:
            display(Markdown(f"### 🙋‍♂️\n>#### {query}"))
        
        if output.get("response"):
            self.conversation.add_message(output["response"], "user", attachment_paths=attachment_paths)
            display(Markdown(f"### 🤖\n>#### {output['response']}"))

        if output.get("pcard"):
            self.conversation.update_pcard(output["pcard"])
            lines = []
            for key, value in output["pcard"].items():
                lines.append(f">#### **{key}**")
                for line in str(value).splitlines():
                    lines.append(f">### {line}")
                lines.append(">####")  
            while lines and lines[-1].strip() == ">####":
                lines.pop()
            display(Markdown(f"📝\n" + "\n".join(lines)))
        
        if self.conversation.pcard:
            lines = []
            for key, value in self.conversation.pcard.items():
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