from openagentkit.core.interfaces import BaseChunker
import re

class RecursiveTextChunker(BaseChunker):
    """
    A text chunker that splits text into chunks based on a specified character limit.
    It uses a recursive approach to break text into smaller chunks based on the specified separator.
    """

    def __init__(self, 
                 chunk_size: int = 100,
                 separator: str = "\n",
                 size_flexibility: float = 0.2):
        """
        Initializes the RecursiveTextChunker with a specified chunk size.

        Args:
            chunk_size (int): The target size of each chunk in characters.
            separator (str): The separator to use for splitting the text.
            size_flexibility (float): How flexible the chunk size can be (0.2 = 20% above target size).
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")
        if size_flexibility < 0:
            raise ValueError("Size flexibility must be non-negative.")
        
        self._chunk_size = chunk_size
        self._separator = separator
        self._size_flexibility = size_flexibility

    @property
    def chunk_size(self) -> int:
        """
        Returns the chunk size.

        Returns:
            int: The chunk size.
        """
        return self._chunk_size
    
    def get_chunks(self, text: str) -> list[str]:
        """
        Splits the input text into chunks respecting the separator boundaries when possible.
        Allows flexible chunk sizes to better respect natural boundaries.

        Args:
            text (str): The input text to be chunked.

        Returns:
            list: A list of chunks.
        """     
        # If text fits in a single chunk, return it as is
        if len(text) <= self._chunk_size:
            return [text]
        
        # Split the text at each separator, keeping the separators
        split_parts: list[str] = []
        last_end = 0
        for match in re.finditer(re.escape(self._separator), text):
            # Get the text before separator and the separator itself
            end_pos = match.end()
            split_parts.append(text[last_end:end_pos])
            last_end = end_pos
        
        # Add any remaining text
        if last_end < len(text):
            split_parts.append(text[last_end:])
        
        # Now combine these parts into chunks according to the size constraints
        result: list[str] = []
        current_chunk = ""
        
        for part in split_parts:
            # If adding this part still keeps us under the chunk size, add it
            if len(current_chunk) + len(part) <= self._chunk_size:
                current_chunk += part
            # If we're under the flexible maximum, still add it
            elif len(current_chunk) + len(part) <= int(self._chunk_size * (1 + self._size_flexibility)):
                current_chunk += part
            # Otherwise start a new chunk
            else:
                if current_chunk:
                    result.append(current_chunk)
                
                # Handle parts that are larger than the chunk size
                if len(part) > self._chunk_size:
                    # For large parts, we need to split them further
                    start = 0
                    while start < len(part):
                        end = min(start + self._chunk_size, len(part))
                        result.append(part[start:end])
                        start = end
                else:
                    current_chunk = part
        
        # Add the last chunk if there is one
        if current_chunk:
            result.append(current_chunk)
            
        return result
    