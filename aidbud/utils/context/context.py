from .currentsituation import CurrentSituationContext
from .firstaidavail import FirstAidAvailableContext
from .triage import TriageContext
from ...config import Config
import pickle
from typing import Dict

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
    
    def reset(self):
        self.triage_context = TriageContext()
        self.firstaidavail_context = FirstAidAvailableContext()
        self.currentsituation_context = CurrentSituationContext()
        self.save()
    
    def enable_triage(self):
        self.triage_context.enable()
        self.save()
    
    def disable_triage(self):
        self.triage_context.disable()
        self.save()
    
    def enable_first_aid(self):
        self.firstaidavail_context.enable()
        self.save()
    
    def disable_first_aid(self):
        self.firstaidavail_context.disable()
        self.save()
    
    def enable_current_situation(self):
        self.currentsituation_context.enable()
        self.save()
    
    def disable_current_situation(self):
        self.currentsituation_context.disable()
        self.save()
    
    def set_current_situation(self, situation: str):
        self.currentsituation_context.set_situation(situation)
        self.save()
    
    def set_first_aid(self, availability: str):
        self.firstaidavail_context.set_availability(availability)
        self.save()
    
    def set_triage(self, triage: Dict[str, str]):
        self.triage_context.update_protocol(triage)
        self.save()