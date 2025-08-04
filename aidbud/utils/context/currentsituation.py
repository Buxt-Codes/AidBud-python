class CurrentSituationContext:
    def __init__(self):
        self.enabled = False
        self.current_situation = ""
    
    def set_situation(self, situation: str):
        self.situation = situation
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False