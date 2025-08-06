from jsonfinder import jsonfinder
import re
from typing import Optional, List, Dict, Any, Union, Tuple

class Parser:
    def parse_response(self, response: str, find_function: bool = True) -> Optional[Dict[str, Any]]:
        if find_function:
            fcall = self._find_fcall(response)

            if isinstance(fcall, dict):
                ID = fcall.get("ID")
                REMARKS = fcall.get("REMARKS")
                if isinstance(ID, int):
                    if isinstance(REMARKS, str):
                        return {"type": "fcall", "context": {"id": ID, "remarks": REMARKS}}
                    else:
                        return {"type": "fcall", "context": {"id": ID, "remarks": ""}}
            
            if fcall:
                return {"type": "fcall", "context": None}
        
        response = self._find_pcard(response)

        if response:
            return {"type": "response", "context": response}
        
        return None

    def parse_attachment_response(self, response: str) -> str:
        for start, end, match in jsonfinder(response):
            if isinstance(match, dict):
                if match.get("description"):
                    return str(match.get("description"))
        
        return None
        
    
    def _find_fcall(self, response: str) -> Optional[Dict]:
        for start, end, match in jsonfinder(response):
            if isinstance(match, dict):
                if match.get("ID"):
                    return match
        
        return None

    def _find_pcard(self, response: str) -> Optional[Dict]:
        for start, end, match in jsonfinder(response):
            if isinstance(match, dict):
                return match
        
        return None