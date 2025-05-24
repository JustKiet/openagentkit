from functools import update_wrapper
from typing import Callable, Literal, Any, Optional, Dict, TypeVar, overload, Union
from pydantic import create_model
import inspect

# --------------------------------
# Tool implementation
# --------------------------------
class Tool:
    """
    Wrapper that makes a function into a tool with a schema.
    """
    __tool_wrapped__ = True

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        schema: Dict[str, Any],
    ):
        self._func = func
        self.schema = schema
        update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Tool {self._func.__name__}>"
    
    def __name__(self) -> str:
        return self._func.__name__

# --------------------------------
# tool decorator with overloads for proper typing
# --------------------------------
T = TypeVar("T", bound=Callable[..., Any])

@overload
def tool(func: T) -> Tool: ... # type: ignore

@overload
def tool(
    *,
    description: str = "",
    schema_type: Literal["OpenAI", "OpenAIRealtime"] = "OpenAI",
    add_tool_notification: bool = False,
    notification_message_guide: str = (
        "The notification that you say to the user when you are executing this tool. "
        "If you execute multiple tools, you must include all the tool names in this notification too and all the notifications must be the same."
    )
) -> Callable[[T], Tool]: ... # type: ignore

def tool(
    func: Optional[T] = None,
    *,
    description: str = "",
    schema_type: Literal["OpenAI", "OpenAIRealtime"] = "OpenAI",
    add_tool_notification: bool = False,
    notification_message_guide: str = (
        "The notification that you say to the user when you are executing this tool. "
        "If you execute multiple tools, you must include all the tool names in this notification too and all the notifications must be the same."
    )
) -> Union[Tool, Callable[[T], Tool]]:
    """
    Decorator to wrap a function into a Tool with OpenAI function-calling schema.
    """
    def decorator(inner_func: T) -> Tool:
        # Inspect signature and build pydantic model for parameters
        signature = inspect.signature(inner_func)
        final_description = inspect.getdoc(inner_func) or description

        model_fields = {
            name: (param.annotation, ...)
            for name, param in signature.parameters.items()
        }
        ToolArguments = create_model("ToolArguments", **model_fields)  # type: ignore
        raw_schema = ToolArguments.model_json_schema() # type: ignore
        assert isinstance(raw_schema, dict), "ToolArguments.model_json_schema() must return a dict"
        tool_arguments: Dict[str, Any] = raw_schema # type: ignore
        tool_arguments.pop("title", None)
        tool_arguments["additionalProperties"] = False

        if add_tool_notification:
            props = tool_arguments.setdefault("properties", {})
            props["_notification"] = {
                "title": "Tool Request Notification",
                "type": "string",
                "description": notification_message_guide,
            }
            req = tool_arguments.setdefault("required", [])
            req.append("_notification")

        if schema_type == "OpenAI":
            schema: Dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": inner_func.__name__,
                    "description": final_description,
                    "strict": True,
                    "parameters": tool_arguments,
                },
            }
        elif schema_type == "OpenAIRealtime":
            schema = {
                "type": "function",
                "name": inner_func.__name__,
                "description": final_description,
                "parameters": tool_arguments,
            }
        else:
            raise ValueError(f"Unsupported schema_type: {schema_type}")

        return Tool(inner_func, schema=schema)

    # If used without args: @tool
    return decorator if func is None else decorator(func)
