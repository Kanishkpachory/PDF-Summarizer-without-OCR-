from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)

    components = relationship("TextComponent", back_populates="document")
    graphics = relationship("Graphic", back_populates="document")
    tables = relationship("Table", back_populates="document")
    chunks = relationship("Chunk", back_populates="document")

class TextComponent(Base):
    __tablename__ = "text_components"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))

    type = Column(String)          # title/subtitle/paragraph/author/...
    content = Column(Text)
    summary = Column(Text)
    position = Column(JSON)        # {"page": 1, "bbox": [x0,y0,x1,y1], ...}

    document = relationship("Document", back_populates="components")

class Graphic(Base):
    __tablename__ = "graphics"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))

    page = Column(Integer)         # 1-based
    bbox = Column(JSON)            # [x0, y0, x1, y1] in PDF coords
    file_path = Column(String)     # e.g. "storage/12/images/p1_img0.png"

    # Text objects that appear in the same region (no OCR; only PDF text layer)
    extracted_texts = Column(JSON) # [{"text": "...", "bbox": [...], "rel_bbox":[...]}]
    summary = Column(Text)

    document = relationship("Document", back_populates="graphics")

class Table(Base):
    __tablename__ = "tables"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))

    page = Column(Integer)
    bbox = Column(JSON)
    data = Column(JSON)            # list[list[str|None]]
    extracted_texts = Column(JSON) # [{"text": "...", "bbox": [...], "rel_bbox":[...]}]
    summary = Column(Text)

    document = relationship("Document", back_populates="tables")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))

    page = Column(Integer)         # Page number where chunk starts
    title = Column(String)
    content = Column(Text)
    summary = Column(Text)         # Summary for chunks > 200 words
    questions = Column(JSON)

    document = relationship("Document", back_populates="chunks")