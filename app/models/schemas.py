from pydantic import BaseModel

class ChatResponse(BaseModel):
    transcription: str
    answer: str
    audio_reply: str
