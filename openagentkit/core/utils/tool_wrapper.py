from functools import wraps
from typing import Annotated, Literal, Callable
from pydantic import create_model
import inspect

def tool(
    _func: Callable = None,
    *,
    description: str = "",
    schema_type: Literal["OpenAI", "OpenAIRealtime"] = "OpenAI",
    add_tool_notification: bool = False,
    notification_message_guide: str = (
        "The notification that you say to the user when you are executing this tool. "
        "If you execute multiple tools, you must include all the tool names in this notification too and all the notifications must be the same."
    )
):
    def decorator(func):
        func.__tool_wrapped__ = True

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        signature = inspect.signature(func)
        final_description = inspect.getdoc(func) or description

        model_fields = {
            name: (param.annotation, ...)
            for name, param in signature.parameters.items()
        }

        ToolArguments = create_model("ToolArguments", **model_fields)
        tool_arguments = ToolArguments.model_json_schema()
        tool_arguments.pop("title")
        tool_arguments["additionalProperties"] = False

        if add_tool_notification is True:
            tool_arguments.get("properties", {})["_notification"] = {
                "title": "Tool Request Notification",
                "type": "string",
                "description": notification_message_guide,
            }
            if tool_arguments.get("required") is None:
                tool_arguments["required"] = []

            tool_arguments.get("required", []).append("_notification")

        match schema_type:
            case "OpenAI":
                wrapper.schema = {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": final_description,
                        "strict": bool(tool_arguments),
                        "parameters": tool_arguments,
                    },
                }

            case "OpenAIRealtime":
                wrapper.schema = {
                    "type": "function",
                    "name": func.__name__,
                    "description": final_description,
                    "parameters": tool_arguments,
                }

        return wrapper
    
    if _func is None:
        return decorator
    else:
        return decorator(_func)