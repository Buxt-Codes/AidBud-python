from jsonfinder import jsonfinder
import re
from typing import Optional, List, Dict, Any, Union, Tuple

class Parser:
    def parse_response(self, response: str, find_function: bool = True) -> List[Dict[str, Any]]:
        if find_function:
            fcall = self._find_fcall(response)

            if isinstance(fcall, dict):
                return [{"type": "fcall", "context": fcall}]
            
            if fcall:
                return [{"type": "fcall", "context": None}]
        
        pcard, new_response = self._find_pcard(response)

        if isinstance(pcard, dict):
            return [{"type": "pcard", "context": pcard}, {"type": "response", "response": new_response}]

        if pcard:
            return [{"type": "pcard", "context": None}, {"type": "response", "response": new_response}]
        
        return [{"type": "response", "response": new_response}]

    def parse_attachment_response(self, response: str) -> Dict[str, str]:
        for match in jsonfinder(response):
            if isinstance(match, dict):
                if match.get("description"):
                    return {"description": str(match.get("description"))}
        
        return None
        
    
    def _find_fcall(self, response: str) -> Optional[Dict]:
        fcall_pattern = re.compile(r"\[FCALL\](.*?)\[/FCALL\]", re.DOTALL)
        fcalls = fcall_pattern.findall(response)

        if not fcalls:
            return None
        
        first_block = fcalls[0]

        for match in jsonfinder(first_block):
            return match.value
        
        return None

    def _find_pcard(self, response: str) -> Tuple[Optional[Dict], str]:
        pcard_pattern = re.compile(r"\[PCARD\](.*?)\[/PCARD\]", re.DOTALL)
        pcalls = pcard_pattern.findall(response)

        if not pcalls:
            return None, response
        
        first_block = pcalls[0]

        for match in jsonfinder(first_block):
            new_response = pcard_pattern.sub("[EDITED PATIENT CARD]", response)
            return match.value, new_response
        
        return None, response