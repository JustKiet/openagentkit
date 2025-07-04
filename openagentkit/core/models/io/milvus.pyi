from pydantic import BaseModel
from pymilvus import DataType as DataType

class MilvusField(BaseModel):
    field_name: str
    datatype: DataType
    class Config:
        extra: str
