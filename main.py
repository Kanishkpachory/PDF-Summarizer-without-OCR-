import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import get_db, engine, Base
from models import Document, TextComponent, Graphic, Table, Chunk
from extractors import extract_from_pdf
from chunker import create_chunks
from schemas import DocumentOut

Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS (dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # For production, restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted images
os.makedirs("storage", exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    temp_path = None
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Create doc row first so we have doc.id for storage paths
        doc = Document(filename=file.filename)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        doc_storage_dir = os.path.join("storage", str(doc.id))
        os.makedirs(doc_storage_dir, exist_ok=True)

        # Extract
        if file.filename.lower().endswith(".pdf"):
            try:
                data = extract_from_pdf(temp_path, storage_dir=doc_storage_dir)
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")
        else:
            db.rollback()
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Persist extracted text components
        for comp in data.get("components", []):
            try:
                db.add(TextComponent(document_id=doc.id, **comp))
            except Exception as e:
                print(f"Error adding component: {e}, component: {comp}")

        # Persist tables
        for t in data.get("tables", []):
            try:
                db.add(Table(document_id=doc.id, **t))
            except Exception as e:
                print(f"Error adding table: {e}, table: {t}")

        # Persist graphics, and convert disk path -> URL path
        for g in data.get("graphics", []):
            try:
                fp = g["file_path"].replace("\\", "/")
                if fp.startswith("storage/"):
                    g["file_path"] = "/" + fp  # becomes "/storage/<doc_id>/..."
                db.add(Graphic(document_id=doc.id, **g))
            except Exception as e:
                print(f"Error adding graphic: {e}, graphic: {g}")

        # Create chunks (currently text-focused; you can enhance to embed graphic/table refs)
        try:
            chunks = create_chunks(
                data.get("components", []),
                data.get("graphics", []),
                data.get("tables", [])
            )
            for ch in chunks:
                try:
                    db.add(Chunk(document_id=doc.id, **ch))
                except Exception as e:
                    print(f"Error adding chunk: {e}, chunk: {ch}")
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=f"Chunking failed: {str(e)}")

        db.commit()

        # Cleanup temp upload
        try:
            os.remove(temp_path)
        except Exception:
            pass

        return {"message": "Extracted", "doc_id": doc.id}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload: {e}")
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return {"message": "Extracted", "doc_id": doc.id}


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Return everything: components + graphics + tables + chunks
    return doc


@app.put("/chunks/{chunk_id}")
def edit_chunk(chunk_id: int, payload: dict, db: Session = Depends(get_db)):
    """
    Allow editing chunk content/title/summary.
    payload example: {"title": "...", "content": "...", "summary": "..."}
    """
    chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    if "title" in payload:
        chunk.title = payload["title"]
    if "content" in payload:
        chunk.content = payload["content"]
    if "summary" in payload:
        chunk.summary = payload["summary"]

    db.commit()
    return {"message": "Chunk updated"}


@app.put("/text-components/{component_id}")
def edit_text_component(component_id: int, payload: dict, db: Session = Depends(get_db)):
    """
    Allow editing text component content.
    payload example: {"content": "..."}
    """
    component = db.query(TextComponent).filter(TextComponent.id == component_id).first()
    if not component:
        raise HTTPException(status_code=404, detail="Text component not found")

    if "content" in payload:
        component.content = payload["content"]

    db.commit()
    return {"message": "Text component updated"}