from pydantic import BaseModel

class DuckDuckGoResponse(BaseModel):
    query: str
    response: list