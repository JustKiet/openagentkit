from openagentkit.utils.tool_wrapper import tool
from typing import Annotated
from pydantic import BaseModel

class AuthenticateResponse(BaseModel):
    authenticated: bool
    message: str

@tool(
    description="Authenticate a user",
)
def authenticate_tool(name: Annotated[str, "The name of the user"], favorite_color: Annotated[str, "The favorite color of the user for security purposes"]):
    """Authenticate a user"""
    if favorite_color.lower() == "blue":
        return AuthenticateResponse(authenticated=True, message=f"{name} is authenticated!")
    else:
        return AuthenticateResponse(authenticated=False, message=f"{name} is not authenticated!")