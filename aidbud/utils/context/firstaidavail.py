class FirstAidAvailableContext:
    def __init__(self):
        self.enabled = False
        self.current_availability = "Immediate"
    
    def set_availability(self, availability: str):
        if availability not in ["Immediate", "Non-Immediate", "Unavailable"]:
            raise ValueError(f"Invalid availability: {availability}, must be one of ['Immediate', 'Non-Immediate', 'Unavailable']")
        
        self.current_availability = availability
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False