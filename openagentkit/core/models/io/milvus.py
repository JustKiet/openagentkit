from pydantic import BaseModel
from pymilvus import DataType # type: ignore

class MilvusField(BaseModel):
    field_name: str
    datatype: DataType

    class Config:
        extra = "allow"
    