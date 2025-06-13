from pydantic import BaseModel
from typing import Any

class CollectionCreate(BaseModel):
    """
    CollectionCreate is a Pydantic model that represents the response of a collection creation operation.
    It contains the name of the collection and the vector dimension.
    """
    collection_schema: dict[str, Any]

class CollectionRename(BaseModel):
    pass

class CollectionDrop(BaseModel):
    pass

class CollectionLoad(BaseModel):
    pass

class VectorAdd(BaseModel):
    pass

class VectorAddBatch(BaseModel):
    pass

class VectorUpdate(BaseModel):
    pass

class VectorSearch(BaseModel):
    pass

class VectorQuery(BaseModel):
    pass

class VectorDelete(BaseModel):
    pass