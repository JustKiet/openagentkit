from pydantic import BaseModel
from typing import Any, Literal

class ReatimeNoiseReductionConfig(BaseModel):
    type: Literal['near_field', 'far_field'] | None

class RealtimeInputAudioTranscriptionConfig(BaseModel):
    language: str | None
    model: Literal['gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'whisper-1']
    prompt: str | None

class RealtimeToolDetail(BaseModel):
    description: str | None
    name: str | None
    parameters: dict[str, Any] | None
    type: Literal['function']

class RealtimeTurnDetectionConfig(BaseModel):
    create_response: bool
    eagerness: Literal['low', 'medium', 'high', 'auto']
    interrupt_response: bool
    prefix_padding_ms: int
    silence_duration_ms: int
    threshold: float
    type: Literal['server_vad']
    def validate_threshold(cls, v: float) -> float: ...

class RealtimeSessionPayload(BaseModel):
    input_audio_format: Literal['pcm16', 'g711_ulaw', 'g711_alow']
    input_audio_noise_reduction: ReatimeNoiseReductionConfig | None
    input_audio_transcription: RealtimeInputAudioTranscriptionConfig | None
    instructions: str | None
    max_response_output_tokens: int | Literal['inf'] | None
    modalities: list[Literal['text', 'audio']] | None
    model: str | None
    output_audio_format: Literal['pcm16, g711_ulaw, g711_alow'] | None
    temperature: float | None
    tool_choice: Literal['auto', 'none', 'required'] | None
    tools: list[RealtimeToolDetail] | None
    turn_detection: RealtimeTurnDetectionConfig | None
    voice: Literal['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse'] | None

class RealtimeClientPayload(BaseModel):
    event_id: str | None
    session: RealtimeSessionPayload
    type: Literal['session.update', 'input_audio_buffer.append', 'input_audio_buffer.commit', 'input_audio_buffer.clear', 'conversation.item.create', 'conversation.item.truncate', 'conversation.item.delete', 'response.create', 'response.cancel', 'transcription_session.update']
