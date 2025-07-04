from _typeshed import Incomplete
from openagentkit.core.interfaces import BaseChunker as BaseChunker, BaseEmbeddingModel as BaseEmbeddingModel
from openagentkit.core.models.io.embeddings import EmbeddingSplits as EmbeddingSplits, EmbeddingUnit as EmbeddingUnit
from typing import Literal

BREAKPOINT_DEFAULTS: dict[Literal['percentile', 'standard_deviation', 'interquartile', 'gradient'], float]

class SemanticTextChunker(BaseChunker):
    embedding_model: Incomplete
    breakpoint_threshold_type: Incomplete
    breakpoint_threshold_amount: Incomplete
    regex_pattern: Incomplete
    buffer_size: Incomplete
    def __init__(self, embedding_model: BaseEmbeddingModel[int] | BaseEmbeddingModel[str], breakpoint_threshold_type: Literal['percentile', 'standard_deviation', 'interquartile', 'gradient'] = 'percentile', breakpoint_threshold_amount: int | None = None, regex_pattern: str = '(?<=[.?!])\\s+', buffer_size: int = 1) -> None: ...
    def get_chunks(self, text: str) -> list[str]: ...
