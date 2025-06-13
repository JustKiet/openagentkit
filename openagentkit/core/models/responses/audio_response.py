from pydantic import BaseModel

class AudioResponse(BaseModel):
    id: str
    data: str
    transcription: str