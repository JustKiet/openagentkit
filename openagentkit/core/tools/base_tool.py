from functools import update_wrapper
from typing import Callable, Any, Optional, Dict, TypeVar, overload, Union
from pydantic import create_model
import inspect

def build_tool_schema(
    func: Callable[..., Any], 
    name: str,
    description: str = ""
) -> Dict[str, Any]:
    signature = inspect.signature(func)
    final_description = inspect.getdoc(func) or description

    model_fields = {
        name: (param.annotation, ...)
        for name, param in signature.parameters.items()
    }

    ToolArguments = create_model("ToolArguments", **model_fields) # type: ignore
    raw_schema = ToolArguments.model_json_schema() # type: ignore
    raw_schema.pop("title", None) # type: ignore
    raw_schema["additionalProperties"] = False

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": final_description,
            "strict": True,
            "parameters": raw_schema,
        },
    }

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
        func: Optional[Callable[..., Any]] = None,
        *,
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        if func is None:
            func = self.__call__

        self._func = func

        if schema is None:
            schema = build_tool_schema(self._func, name=str(self.__class__.__name__))
        
        self.schema = schema
        update_wrapper(self, self._func)

    @classmethod
    def from_decorator(
        cls,
        func: Callable[..., Any],
        *,
        schema: Dict[str, Any],
    ) -> "Tool":
        """ 
        Create a Tool instance from a function and a schema.
        This is used internally by the `tool` decorator.
        """
        tool_instance = cls(func, schema=schema)
        return tool_instance

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(self._func):
            return self._func(*args, **kwargs)
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
) -> Callable[[T], Tool]: ... # type: ignore

def tool(
    func: Optional[T] = None,
    *,
    description: str = "",
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

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": inner_func.__name__,
                "description": final_description,
                "strict": True,
                "parameters": tool_arguments,
            },
        }

        return Tool.from_decorator(inner_func, schema=schema)

    # If used without args: @tool
    return decorator if func is None else decorator(func)
