from pydantic import BaseModel
from typing import Optional

class CachedTokensDetails(BaseModel):
    """
    The details of the cached tokens.

    Schema:
        ```python
        class CachedTokensDetails(BaseModel):
            text_tokens: int
            audio_tokens: int
        ```
    Where:
        - text_tokens: The cached text tokens.
        - audio_tokens: The cached audio tokens.
    """
    text_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None

class PromptTokensDetails(BaseModel):
    """
    The details of the prompt tokens.

    Schema:
        ```python
        class PromptTokensDetails(BaseModel):
            cached_tokens: int
            text_tokens: int
            audio_tokens: int
            cached_tokens_details: CachedTokensDetails
        ```
    Where:
        - `cached_tokens`: The cached tokens.
        - `text_tokens`: The text tokens.
        - `audio_tokens`: The audio tokens.
        - `cached_tokens_details`: The cached tokens details. Is a nested model of type `CachedTokensDetails`.
    """
    cached_tokens: Optional[int] = None
    text_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    cached_tokens_details: Optional[CachedTokensDetails] = None

class CompletionTokensDetails(BaseModel):
    """
    The details of the completion tokens.

    Schema:
        ```python
        class CompletionTokensDetails(BaseModel):
            reasoning_tokens: int
            audio_tokens: int
            accepted_prediction_tokens: int
            rejected_prediction_tokens: int
        ```
    Where:
        - `reasoning_tokens`: The reasoning tokens.
        - `audio_tokens`: The audio tokens.
        - `accepted_prediction_tokens`: The accepted prediction tokens.
        - `rejected_prediction_tokens`: The rejected prediction tokens.
    """
    reasoning_tokens: Optional[int] = None
    text_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None

class UsageResponse(BaseModel):
    """
    The usage response for completion models.

    Schema:
        ```python
        class UsageResponse(BaseModel):
            prompt_tokens: int
            completion_tokens: int
            total_tokens: int
            prompt_tokens_details: PromptTokensDetails
            completion_tokens_details: CompletionTokensDetails
        ```
    Where:
        - `prompt_tokens`: The prompt tokens.
        - `completion_tokens`: The completion tokens.
        - `total_tokens`: The total tokens.
        - `prompt_tokens_details`: The prompt tokens details.
        - `completion_tokens_details`: The completion tokens details.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[PromptTokensDetails] = None
    completion_tokens_details: Optional[CompletionTokensDetails] = None
