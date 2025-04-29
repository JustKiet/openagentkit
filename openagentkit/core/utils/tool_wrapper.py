from typing import (
    List, 
    Dict,
    Set, 
    Any, 
    Annotated, 
    get_type_hints, 
    Optional, 
    Union, 
    get_origin, 
    get_args,
    TypedDict,
    _TypedDictMeta,
    Literal
    )
import inspect
from inspect import _empty
from functools import wraps
import types
import json


def type_mapper(annotation: str) -> str:
    annotation = annotation.strip().lower()
    
    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "nonetype": "null",
        "set": "array",
        "list": "array",
        "set": "array",
        "dict": "object",
    }
    return type_map.get(annotation, annotation)

def remove_none_values(d):
    if isinstance(d, dict):
        return {
            k: remove_none_values(v) if isinstance(v, (dict, list)) else v
            for k, v in d.items() if v is not None
        }
    elif isinstance(d, list):
        return [remove_none_values(i) if isinstance(i, (dict, list)) else i for i in d if i]
    return d 


def get_type_metadata(annotation, description: Optional[str] = None) -> dict:
    if isinstance(annotation, list) or get_origin(annotation) in [List, list]:
        args = get_args(annotation)

        return {
            "type": "list",
            "description": description,
            "additionalProperties": get_type_metadata(args[0]) if get_origin(args[0]) else {"type": type_mapper(args[0].__name__)},
        }

    elif isinstance(annotation, types.UnionType) or get_origin(annotation) is Union:
        args = get_args(annotation)

        args_list = []

        for arg in args:
            if arg.__name__:
                args_list.append(get_type_metadata(arg) if get_origin(arg) else {"type": type_mapper(arg.__name__)})

        return{
            "anyOf": args_list,
            "description": description,
        }

    elif isinstance(annotation, dict) or get_origin(annotation) in [Dict, dict]:
        args = get_args(annotation)

        return{
            "type": "object",
            "description": description,
            "additionalProperties": get_type_metadata(args[-1]) if get_origin(args[-1]) else {"type": type_mapper(args[-1].__name__)},
        }
    
    elif isinstance(annotation, Annotated) or get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return get_type_metadata(args[0], args[1])

    elif isinstance(annotation, _TypedDictMeta):
        return{
            "type": "object",
            "description": description,
            "additionalProperties": False,
            "properties": {
                k: get_type_metadata(v) if get_origin(v) else {"type": type_mapper(v.__name__)}
                for k, v in annotation.__annotations__.items()
            }
        }
    
    elif get_origin(annotation) is Literal:
        return{
            "type": type_mapper(type(get_args(annotation)[0]).__name__),
            "description": description,
            "enum": [arg for arg in get_args(annotation)],
        }

    else:
        return {
            "type": type_mapper(annotation.__name__),
            "description": description,
        }

def tool(
        description: str, 
        type: Literal["OpenAI", "OpenAIRealtime"] = "OpenAI",
        _notification: bool = False,
        _notification_message: str = "The notification that you say to the user when you are executing this tool. If you execute multiple tools, you must include all the tool names in the notification too."):
    """
    A decorator that automatically generates a JSON schema for a function based on its type hints.

    **IMPORTANT**: The type hints must be provided as **Annotated[<type>, <description>]**. The description must be a string that describes the type.

    **Arguments**
        description: str - A description of the function. This description must be provided and be specific to the function so the LLM can understand the purpose of the function.
        _notification: bool - Whether to include a notification in the function schema.
        _notification_message: str - The message to include in the notification.
    **Returns**
        A function that has a schema attribute that contains the JSON schema for the function.

    **Example**
    ```python
    @tool(
        description="Example desc"
    )
    def simple_function(a: Annotated[int, "Int type"], b: Annotated[str, "String type"]):
        pass
        
    print(simple_function.schema)
    ```
    """
    def decorator(func):

        func.__tool_wrapped__ = True

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        signature = inspect.signature(func)
        properties = {}
        required = []
        
        if _notification is True:
            properties = {
                "_notification": {
                    "type": "string",
                    "description": _notification_message
                }
            }
            required.append("_notification")

        for name, param in signature.parameters.items():
            annotation = param.annotation
            if annotation and annotation != _empty:
                param_type = annotation.__args__[0]
                desc = annotation.__metadata__[0]

                if param_type:
                    properties[name] = get_type_metadata(param_type, desc)

            required.append(name)

        match type:
            case "OpenAI":
                wrapper.schema = {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": description,
                        "strict": True if properties else False,
                        "parameters": {
                            "type": "object" if properties else None,
                            "properties": properties if properties else None,
                            "required": required if required else [],
                            "additionalProperties": False if properties else None,
                        },
                    }
                }
            case "OpenAIRealtime":
                wrapper.schema = {
                    "type": "function",
                    "name": func.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object" if properties else None,
                        "properties": properties if properties else None,
                        "required": required if required else [],
                    }
                }
        
        wrapper.schema = dict(remove_none_values(wrapper.schema))

        return wrapper
    return decorator

def get_tools_from_class(cls):
    tools = [
        method for _ , method in inspect.getmembers(cls, predicate=inspect.ismethod)
        if hasattr(method, "__tool_wrapped__")
    ]
    return tools

if __name__ == "__main__":
    @tool(
        description="Example desc"
    )
    def complex_function(
        #a: Annotated[str | int | float, "Union Type"],
        #b: Annotated[int, "Int type"],
        #c: Annotated[dict[str, Union[int, str]], "Dict type"],
        #d: Annotated[Set[Union[str, int]], "Set type"],
        #e: Annotated[List[Dict[str, Optional[str]]], "List of Dict type"],
        #f: Annotated[Union[float, str, bool], "Complex Union type"],
        #g: Annotated[Union[Dict[str, List[int]], List[Dict[str, str]]], "Union of Dict and List type"],
        h: Annotated[Optional[Union[Dict[str, int], dict[str, str]]], "Optional type with Union of Dict"],
        #i: Annotated[List[Union[Dict[str, str], Set[str]]], "List of Dict or Set type"],
        j: Annotated[Optional[List[Dict[str, Union[float, int]]]], "Optional List of Dicts with Union"],
        #k: Annotated[Any, "Any type"],
        #m: Annotated[List[List[int]], "List of List type"],
        #n: Annotated[Literal["a", "b", "c"], "Literal type"],
        #o: Annotated[Literal[1, 2, 3], "Literal type with int"],
        #p: Annotated[bool, "Boolean type"],
    ):
        pass

    print(json.dumps(complex_function.schema, indent=2))