from openagentkit.core.models.io.embeddings import EmbeddingUnit, EmbeddingSplits
from openagentkit.core.interfaces import BaseChunker, BaseEmbeddingModel
from scipy.spatial.distance import cosine
from typing import Literal, Optional, Union
import numpy as np
import re

BREAKPOINT_DEFAULTS: dict[Literal["percentile", "standard_deviation", "interquartile", "gradient"], float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}

class SemanticTextChunker(BaseChunker):
    """
    A semantic text chunker. This chunker will split the text into chunks based on the cosine similarity of the embeddings of the splits.
    It will then combine the splits into chunks based on the breakpoint threshold.

    The breakpoint threshold type can be one of the following:
    - `percentile`: The percentile of the cosine similarity scores to use as the breakpoint threshold.
    - `standard_deviation`: The standard deviation of the cosine similarity scores to use as the breakpoint threshold.
    - `interquartile`: The interquartile range of the cosine similarity scores to use as the breakpoint threshold.
    - `gradient`: The gradient of the cosine similarity scores to use as the breakpoint threshold.

    All credit goes to Greg Kamradt for the original implementation: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    
    Args:
        embedding_model: The embedding model to use.
        breakpoint_threshold_type: The type of breakpoint threshold to use.
        breakpoint_threshold_amount: The amount of breakpoint threshold to use.
        regex_pattern: The regex pattern to use.
        buffer_size: The buffer size to use.

    Returns:
        list[str]: A list of chunks.
    """
    def __init__(self, 
                 embedding_model: Union[BaseEmbeddingModel[int], BaseEmbeddingModel[str]],
                 breakpoint_threshold_type: Literal[
                     "percentile", 
                     "standard_deviation", 
                     "interquartile", 
                     "gradient"
                 ] = "percentile",
                 breakpoint_threshold_amount: Optional[int] = None,
                 regex_pattern: str = r'(?<=[.?!])\s+',
                 buffer_size: int = 1):
        self.embedding_model = embedding_model
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount

        if self.breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[self.breakpoint_threshold_type]

        self.regex_pattern = re.compile(regex_pattern)
        self.buffer_size = buffer_size

    def _regex_split(self, text: str) -> list[EmbeddingSplits]:
        """
        Split the text into splits based on the regex pattern.

        Args:
            text: The text to split.

        Returns:
            list[EmbeddingSplits]: A list of splits.
        """
        splits = re.split(self.regex_pattern, text)
        return [
            EmbeddingSplits(
                content=content,
                index=idx,
                combined_splits=None,
            )
            for idx, content in enumerate(splits) if content.strip()
        ]
    
    def _combine_splits(self, splits: list[EmbeddingSplits], buffer_size: int = 1):
        """
        Combine the splits into chunks based on the buffer size.

        Args:
            splits: The splits to combine.
            buffer_size: The buffer size to use.

        Returns:
            list[EmbeddingSplits]: A list of combined splits.
        """
        for i in range(len(splits)):

            # Create a string that will hold the splits which are joined
            combined_string = ''

            # Add splits before the current one, based on the buffer size.
            for j in range(i - buffer_size, i):
                # Check if the index j is not negative (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_string string
                    combined_string += splits[j].content + ' '

            # Add the current sentence
            combined_string += splits[i].content

            # Add splits after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the splits list
                if j < len(splits):
                    # Add the sentence at index j to the combined_string string
                    combined_string += ' ' + splits[j].content

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            splits[i].combined_splits = combined_string

        return splits
    
    def _calculate_cosine_similarities(self, embedding_units: list[EmbeddingUnit]) -> list[float]:
        """
        Calculate the cosine similarity of the embeddings of the splits.

        Args:
            embedding_units: The embeddings of the splits.

        Returns:
            list[float]: A list of cosine similarity scores.
        """
        similarity_scores: list[float] = []
        for i in range(len(embedding_units) - 1):
            embedding_current = embedding_units[i].embedding
            embedding_next = embedding_units[i + 1].embedding
            
            # Calculate cosine similarity by simply take 1 minus the distance between the embedding pairs
            similarity = 1 - cosine(embedding_current, embedding_next)

            # Append cosine similarity to the list
            similarity_scores.append(similarity)

        return similarity_scores
    
    def _calculate_breakpoint_threshold(self, distances: list[float]) -> float:
        """
        Calculate the breakpoint threshold based on the breakpoint threshold type.

        Args:
            distances: The distances of the splits.

        Returns:
            float: The breakpoint threshold.
        """
        match self.breakpoint_threshold_type:
            case "percentile":
                return float(np.percentile(distances, self.breakpoint_threshold_amount)) # type: ignore
            
            case "standard_deviation":
                return float(np.mean(distances) + self.breakpoint_threshold_amount * np.std(distances)) # type: ignore
            
            case "interquartile":
                q1, q3 = np.percentile(distances, [25, 75])
                iqr = q3 - q1
                return np.mean(distances) + self.breakpoint_threshold_amount * iqr
            
            case "gradient":
                # Calculate the threshold based on the distribution of gradient of distance array
                distance_gradient = np.gradient(distances, range(0, len(distances)))
                return float(np.percentile(distance_gradient, self.breakpoint_threshold_amount)) # type: ignore
            
            case _:
                raise ValueError(f"Got unexpected `breakpoint_threshold_type`: {self.breakpoint_threshold_type}")
        
    def get_chunks(self, text: str) -> list[str]:
        """
        Get the chunks from the text.

        Args:
            text: The text to get the chunks from.

        Returns:
            list[str]: A list of chunks.
        """
        splits = self._regex_split(text)
            
        splits = self._combine_splits(splits, self.buffer_size)

        if len(splits) <= 1:
            combined_text = ' '.join([d.content for d in splits])
            return [combined_text] if combined_text.strip() else []

        embeddings: list[EmbeddingUnit] = self.embedding_model.encode_texts(
            texts=[split.combined_splits for split in splits if split.combined_splits],
            include_metadata=False
        ) # type: ignore

        distances = self._calculate_cosine_similarities(embeddings) # type: ignore

        breakpoint_distance_threshold = self._calculate_breakpoint_threshold(distances)

        # Find indices where the distance is above the threshold
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

        chunks: list[str] = []

        if not indices_above_thresh:
            combined_text = ' '.join([d.content for d in splits])
            if combined_text.strip():
                chunks.append(combined_text)
            return chunks
            
        # Iterate over pairs of breakpoints
        for i in range(len(indices_above_thresh) - 1):
            start = indices_above_thresh[i]
            end = indices_above_thresh[i + 1]
            
            group: list[EmbeddingSplits] = splits[start:end + 1]
            combined_text = ' '.join([d.content for d in group])
            chunks.append(combined_text)

        # Optional: if there's remaining content after the last breakpoint
        last_index = indices_above_thresh[-1]
        if last_index + 1 < len(embeddings):
            group: list[EmbeddingSplits] = splits[last_index + 1:]
            combined_text = ' '.join([d.content for d in group])
            chunks.append(combined_text)

        return chunks