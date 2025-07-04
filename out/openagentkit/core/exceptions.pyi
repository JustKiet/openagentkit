INVALID_TOOL_SCHEMA_MESSAGE: str

class InvalidToolSchemaError(Exception):
    def __init__(self, message: str = ...) -> None: ...

class ToolCallError(Exception):
    def __init__(self, message: str) -> None: ...

class OperationNotAllowedError(Exception):
    def __init__(self, message: str) -> None: ...
