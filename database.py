from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Runtime database: Use SQLite in-memory for simplicity (persists during app run; change to file path for persistence)
DATABASE_URL = "sqlite:///./runtime.db"  # Or "sqlite:///:memory:" for pure in-memory

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()