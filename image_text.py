"""
Upgraded text.py
----------------
Adds multimodal support to the PDF summariser by:
1. Extracting text from PDFs
2. Extracting images (figures, graphs, diagrams)
3. Summarising images using Gemini Vision
4. Merging text + figure summaries
5. Producing a figure-aware document summary

This file is designed to drop into your existing Streamlit/LangChain project
with minimal changes elsewhere.
"""

import os
import fitz  # PyMuPDF
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# -----------------------------
# 1. TEXT EXTRACTION
# -----------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -----------------------------
# 2. IMAGE EXTRACTION
# -----------------------------

def extract_images_from_pdf(pdf_path: str, output_dir: str = "images") -> list:
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_path = os.path.join(
                output_dir, f"page{page_num+1}_img{img_index}.{image_ext}"
            )
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_path)

    return image_paths

# -----------------------------
# 3. IMAGE SUMMARISATION (GEMINI VISION)
# -----------------------------

def summarize_images(image_paths: list, llm: ChatGoogleGenerativeAI) -> list:
    summaries = []

    prompt = (
        "You are analyzing a figure from a document. "
        "Describe what the visual shows, identify key variables or components, "
        "highlight trends or relationships, and state the main insight. "
        "Be concise and factual."
    )

    for image_path in image_paths:
        try:
            response = llm.invoke([
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_path}
            ])
            summaries.append(response.content)
        except Exception as e:
            summaries.append(f"Failed to summarize image {image_path}: {e}")

    return summaries

# -----------------------------
# 4. BUILD ENRICHED DOCUMENT
# -----------------------------

def build_enriched_document(text: str, image_summaries: list) -> str:
    enriched = text.strip()

    if image_summaries:
        enriched += "\n\nFIGURE & GRAPH SUMMARIES:\n"
        for idx, summary in enumerate(image_summaries, 1):
            enriched += f"Figure {idx}: {summary}\n"

    return enriched

# -----------------------------
# 5. VECTOR STORE + QA CHAIN
# -----------------------------

def process_document(enriched_text: str, api_key: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_text(enriched_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


def summarize_document(vectorstore, api_key: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        google_api_key=api_key
    )

    docs = vectorstore.similarity_search(
        "Summarize the document including insights from figures, charts, and diagrams."
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    response = chain.run(
        input_documents=docs,
        question=(
            "Provide a concise, structured summary of the document. "
            "Explicitly include insights derived from figures, graphs, and diagrams."
        )
    )

    return response

# -----------------------------
# 6. END-TO-END PIPELINE
# -----------------------------

def summarize_pdf_with_figures(pdf_path: str, api_key: str) -> str:
    # Init multimodal LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        google_api_key=api_key
    )

    # Extract content
    text = extract_text_from_pdf(pdf_path)
    images = extract_images_from_pdf(pdf_path)
    image_summaries = summarize_images(images, llm)

    # Merge content
    enriched_doc = build_enriched_document(text, image_summaries)

    # Vectorize + summarize
    vectorstore = process_document(enriched_doc, api_key)
    final_summary = summarize_document(vectorstore, api_key)

    return final_summary
