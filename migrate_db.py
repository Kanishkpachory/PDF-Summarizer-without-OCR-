"""
Migration script to handle database schema changes.
Run this before starting the server if you get database schema errors.
"""

import os
from database import engine, Base
from models import Document, TextComponent, Graphic, Table, Chunk

def migrate_database():
    """Migrate database schema by dropping and recreating tables."""
    db_file = "runtime.db"
    
    print("🔧 Starting database migration...")
    
    # Remove old database if it exists
    if os.path.exists(db_file):
        print(f"  Removing old database: {db_file}")
        os.remove(db_file)
        print("  ✓ Database removed")
    
    # Create all tables with the new schema
    print("  Creating new tables with updated schema...")
    Base.metadata.create_all(bind=engine)
    print("  ✓ Tables created successfully")
    
    print("\n✅ Database migration completed!")
    print("   You can now start the server with: uvicorn main:app --reload")

if __name__ == "__main__":
    migrate_database()
