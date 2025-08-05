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
    
    def new_conversation(self):
        self.conversation.reset()
    
    def query(self, query: str = None, attachment_paths: List[str] = None):
        if query == None and attachment_paths == None:
            raise ValueError("At least query or attahment_paths must be provided.")
        
        video_paths, audio_paths, image_paths = self.workflow.classify_attachments(attachment_paths)
        if query == None and video_paths == None and audio_paths == None and image_paths == None:
            raise ValueError("At least query or valid attachments must be provided.")
        
        output = self.workflow.run(conversation_id=self.conversation.conversation_id, query=query, attachment_paths=attachment_paths)

        if output.get("error"):
            return print(f"Error: {output['error']}")
        
        for video_path in video_paths:
            display(Video(video_path))
        for audio_path in audio_paths:
            display(Audio(audio_path))
        for image_path in image_paths:
            display(Image(image_path))
        
        display(Markdown("## Conversation Details"))
        if query:
            formatted_query = "<font size='+1' color='brown'>🙋‍♂️<blockquote>\n" + query + "\n</blockquote></font>"
            display(Markdown(formatted_query))
        
        if output.get("response"):
            self.conversation.add_message(output["response"], "user", attachment_paths=attachment_paths)
            formatted_response = "<font size='+1' color='green'>🤖<blockquote>\n" + output["response"] + "\n</blockquote></font>"
            display(Markdown(formatted_response))
        
        if output.get("pcard"):
            self.conversation.update_pcard(output["pcard"])
            pcard_changes = ""
            for key, value in output["pcard"].items():
                pcard_changes += f"**{key}**\n{value}\n\n"
            display(Markdown("<font size='+1' color='blue'>📝<blockquote>\n" + pcard_changes + "\n</blockquote></font>"))
        
        display(Markdown("## Patient Card Details"))
        pcard = ""
        for key, value in self.conversation.pcard.items():
            pcard += f"**{key}**\n{value}\n\n"
        display(Markdown("<font size='+1' color='blue'>📝<blockquote>\n" + pcard + "\n</blockquote></font>"))