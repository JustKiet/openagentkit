from pydantic import BaseModel

class CachedTokensDetails(BaseModel):
    text_tokens: int | None
    audio_tokens: int | None

class PromptTokensDetails(BaseModel):
    cached_tokens: int | None
    text_tokens: int | None
    audio_tokens: int | None
    cached_tokens_details: CachedTokensDetails | None

class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None
    text_tokens: int | None
    audio_tokens: int | None
    accepted_prediction_tokens: int | None
    rejected_prediction_tokens: int | None

class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None
    completion_tokens_details: CompletionTokensDetails | None
