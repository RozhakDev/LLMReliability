import json
import re
from typing import Dict, Any, Optional

def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM output, handling markdown fences."""
    if isinstance(raw_text, dict):
        return raw_text
        
    try:
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        text_to_parse = json_match.group(0) if json_match else cleaned_text
        
        try:
            return json.loads(text_to_parse)
        except json.JSONDecodeError:
            text_to_parse_fixed = text_to_parse.replace("'", '"')
            return json.loads(text_to_parse_fixed)
    except Exception:
        return None
