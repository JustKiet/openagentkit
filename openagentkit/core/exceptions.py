INVALID_TOOL_SCHEMA_MESSAGE = """
Invalid tool schema provided.
The tool schema must be a dictionary with the following structure:

{
    "type": "function",
    "name": "tool_name",
    "description": "A brief description of what the tool does.",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of the parameter."
            },
            ...
        },
        "required": ["param_name", ...],
        "additionalProperties": False
    }
    "strict": True
}
"""

class InvalidToolSchemaError(Exception):
    """Exception raised for errors in the tool schema."""
    def __init__(self, message: str = INVALID_TOOL_SCHEMA_MESSAGE):
        super().__init__(message)

class ToolCallError(Exception):
    """Exception raised for errors during tool calls."""
    def __init__(self, message: str):
        super().__init__(message)

class OperationNotAllowedError(Exception):
    """Exception raised when an operation is not allowed."""
    def __init__(self, message: str):
        super().__init__(message)