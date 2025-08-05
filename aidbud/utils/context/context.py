from .currentsituation import CurrentSituationContext
from .firstaidavail import FirstAidAvailableContext
from .triage import TriageContext
from ...config import Config
import pickle

class Context:
    def __init__(self, config: Config = Config()):
        self.context_path = config.context["context_path"]
        self.load()
    
    def load(self):
        try:
            with open(self.context_path, "rb") as f:
                data = pickle.load(f)
            self.triage_context = data["triage"]
            self.firstaidavail_context = data["firstaid"]
            self.currentsituation_context = data["situation"]
        except (FileNotFoundError, pickle.PickleError, KeyError):
            self.triage_context = TriageContext()
            self.firstaidavail_context = FirstAidAvailableContext()
            self.currentsituation_context = CurrentSituationContext()
            self.save()

    def save(self):
        with open(self.context_path, "wb") as f:
            pickle.dump({
                "triage": self.triage_context,
                "firstaid": self.firstaidavail_context,
                "situation": self.currentsituation_context
            }, f)