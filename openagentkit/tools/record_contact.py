from openagentkit.utils.tool_wrapper import tool
from typing import Annotated

@tool(
    description="Record the user's contact information"
)
def record_contact(
    name: Annotated[str, "The name of the user"], 
    email: Annotated[str, "The email of the user"], 
    phone: Annotated[str, "The phone number of the user"],
    ):
    """Record the user's contact information"""
    return f"Contact information recorded: {name}, {email}, {phone}"
