from openagentkit.interfaces.base_speech_model import BaseSpeechModel
from openagentkit.models.responses import OpenAgentResponse
from loguru import logger
import json
import base64

class OutputHandler:
    def __init__(self,
                 speech_client: BaseSpeechModel = None,
                 event_handler = None,
                 *args,
                 **kwargs):
        self.speech_client = speech_client
        self.event_handler = event_handler
    
    def handle_output(self, output: OpenAgentResponse, speech: bool = False) -> OpenAgentResponse:
        """Handle the output from the AI model."""
        if speech and self.speech_client:
            try:
                speech_bytes = self.speech_client.text_to_speech(output)
                output.audio = base64.b64encode(speech_bytes).decode("utf-8") if speech_bytes else "No audio available."
            except Exception as e:
                output.audio = "No audio available."
                logger.error(f"Error: {e}")
            return output
        else:
            return output
    
    def save_history_to_json(self, history: list, filename: str):
        """Save the conversation history to a JSON file."""
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def load_history_from_json(self, filename: str):
        """Load the conversation history from a JSON file."""
        with open(filename, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history