from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional, Literal, Union

class ReatimeNoiseReductionConfig(BaseModel):
    type: Optional[Literal["near_field", "far_field"]] = None

class RealtimeInputAudioTranscriptionConfig(BaseModel):
    language: Optional[str] = None
    model: Literal["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"] = "whisper-1"
    prompt: Optional[str] = None

class RealtimeToolDetail(BaseModel):
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    type: Literal["function"] = "function"

class RealtimeTurnDetectionConfig(BaseModel):
    create_response: bool = True
    eagerness: Literal["low", "medium", "high", "auto"] = "auto"
    interrupt_response: bool = True
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    threshold: float = 0.5
    type: Literal["server_vad"] = "server_vad"

    @field_validator("threshold")
    def validate_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v

class RealtimeSessionPayload(BaseModel):
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alow"] = "pcm16"
    input_audio_noise_reduction: Optional[ReatimeNoiseReductionConfig] = None
    input_audio_transcription: Optional[RealtimeInputAudioTranscriptionConfig] = None
    instructions: Optional[str] = None
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = None
    modalities: Optional[List[Literal["text", "audio"]]] = None
    model: Optional[str] = None
    output_audio_format: Optional[Literal["pcm16, g711_ulaw, g711_alow"]] = None
    temperature: Optional[float] = 0.8
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    tools: Optional[List[RealtimeToolDetail]] = None
    turn_detection: Optional[RealtimeTurnDetectionConfig] = None
    voice: Optional[Literal["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]] = None

class RealtimeClientPayload(BaseModel):
    event_id: Optional[str] = None
    session: RealtimeSessionPayload
    type: Literal[
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
        "conversation.item.create",
        "conversation.item.truncate",
        "conversation.item.delete",
        "response.create",
        "response.cancel",
        "transcription_session.update",
    ]