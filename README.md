# PDF-Summarizer

A high-performance and modular document intelligence system, PDF-Summarizer without OCR is designed to extract, process, and summarize textual data from PDFs without relying on OCR, ensuring efficiency and accuracy.

---


## Table of Contents

- Overview  
- Features  
- Getting Started  
- Usage  
- Files and Directories  
- Contributing  
- License  

---

## Overview

PDF-Summarizer-without-OCR provides a streamlined and scalable approach to document processing by leveraging native PDF text extraction instead of OCR. The system is built with a modular backend architecture that enables efficient chunk-based processing, structured storage, and semantic summarization.

It is ideal for building intelligent document pipelines, research tools, and enterprise-level data processing systems.

---

## Features

- **OCR-Free Processing**: Extracts embedded text directly from PDFs for higher accuracy and performance  
- **Chunk-Based Processing**: Breaks large documents into smaller segments for efficient handling  
- **Modular Architecture**: Clean separation of extraction, processing, and storage layers  
- **Database Integration**: Persistent storage using structured schemas and ORM models  
- **Image Extraction Support**: Handles images and prepares for multimodal processing  
- **Scalable Design**: Easily extendable for APIs, AI integration, or SaaS platforms  

---

## Getting Started

To run the project locally, follow these steps:

### Clone the repository
```bash
git clone https://github.com/Kanishkpachory/PDF-Summarizer-without-OCR.git

cd PDF-Summarizer-without-OCR

python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows

pip install -r requirements.txt

python migrate_db.py

python main.py
```
---

## Usage

- Input a PDF file  
- Extract text from document  
- Chunk the content  
- Process and summarize  
- Store output in database

---

## Files and Directories

main.py — Entry point  
extractors.py — PDF text extraction  
chunker.py — Text segmentation  
text.py — Processing logic  
imageExtraction.py — Image extraction  
image_text.py — Image-text linking  

database.py — Database operations  
models.py — ORM models  
schemas.py — Validation schemas  
migrate_db.py — Database setup  

project_data.db — Main database  
runtime.db — Runtime storage  
requirements.txt — Dependencies  
.gitignore — Git exclusions  

---

## Contributing

- Fork repository  
- Create branch  
- Commit changes  
- Push and open PR  

---

## License

MIT License  

---

## Author

Kanishk Pachory  

EOF
