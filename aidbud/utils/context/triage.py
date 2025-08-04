class TriageContext:
    def __init__(self):
        self.enabled = False
        self.protocol = {}
    
    def update_protocol(self, protocol: Dict[str, str]):
        self.protocol = protocol
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
