from pydantic import BaseModel

class PromptTokensDetails(BaseModel):
    """
    The details of the prompt tokens.

    Schema:
        ```python
        class PromptTokensDetails(BaseModel):
            cached_tokens: int
            audio_tokens: int
        ```
    Where:
        - `cached_tokens`: The cached tokens.
        - `audio_tokens`: The audio tokens.
    """
    cached_tokens: int
    audio_tokens: int

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
    reasoning_tokens: int
    audio_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int

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
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails
