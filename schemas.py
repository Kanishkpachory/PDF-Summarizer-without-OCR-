from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TextComponentOut(BaseModel):
    id: int
    type: str
    content: str
    summary: Optional[str] = None
    position: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class GraphicOut(BaseModel):
    id: int
    page: int
    bbox: List[float]
    file_path: str                  # URL: "/storage/<doc_id>/images/..."
    extracted_texts: List[Dict[str, Any]]
    summary: Optional[str] = None

    class Config:
        from_attributes = True


class TableOut(BaseModel):
    id: int
    page: int
    bbox: List[float]
    data: List[List[Optional[str]]]
    extracted_texts: List[Dict[str, Any]]
    summary: Optional[str] = None

    class Config:
        from_attributes = True


class ChunkOut(BaseModel):
    id: int
    page: int
    title: str
    content: str
    questions: List[str]

    class Config:
        from_attributes = True


class DocumentOut(BaseModel):
    id: int
    filename: str
    components: List[TextComponentOut]
    graphics: List[GraphicOut]
    tables: List[TableOut]
    chunks: List[ChunkOut]

    class Config:
        from_attributes = True