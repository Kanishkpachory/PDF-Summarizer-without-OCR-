from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
# --- CHANGED: Using the LangChain Google GenAI integration ---
from langchain_google_genai import ChatGoogleGenerativeAI 
from pypdf import PdfReader
import os
import fitz  # PyMuPDF


def process_text(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Correct BGE model
    embedding = HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        encode_kwargs={"normalize_embeddings": True}
    )

    knowledgebase = FAISS.from_texts(chunks, embedding)

    return knowledgebase


def extract_images_from_pdf(pdf_path: str, output_dir: str = "extracted_images") -> list:
    """Extract images/graphs from PDF"""
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


def summarize_images(image_paths: list, llm: ChatGoogleGenerativeAI) -> list:
    """Generate descriptions for extracted images/graphs using vision"""
    summaries = []

    prompt = (
        "You are analyzing a figure/graph from a document. "
        "Describe what the visual shows, identify key data points or trends, "
        "highlight relationships, and state the main insight. "
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
            summaries.append(f"[Unable to describe graph: {str(e)}]")

    return summaries


def summarizer(PDF):

    if PDF is not None:

        pages = PdfReader(PDF)
        text = ""

        for page in pages.pages:
            text += page.extract_text() or ""

        # Extract and describe graphs
        graph_descriptions = ""
        try:
            # Get PDF file path from uploaded file
            pdf_path = PDF.name if hasattr(PDF, 'name') else "temp.pdf"
            image_paths = extract_images_from_pdf(pdf_path)
            
            # Initialize LLM for vision analysis
            llm_vision = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.8
            )
            
            if image_paths:
                image_summaries = summarize_images(image_paths, llm_vision)
                graph_descriptions = "\n\nGRAPH AND FIGURE DESCRIPTIONS:\n"
                for idx, summary in enumerate(image_summaries, 1):
                    graph_descriptions += f"\nGraph/Figure {idx}:\n{summary}\n"
        except Exception as e:
            print(f"Error extracting graphs: {e}")
            graph_descriptions = ""

        # Combine text with graph descriptions for better context
        enriched_text = text + graph_descriptions

        knowledgebase = process_text(enriched_text)

        query = "Generate a comprehensive summary of the uploaded PDF with detailed elaborations of the embedded graphs. Also show the graphs and their descriptions separately in separate paragraphs with separate headings."

        # retrieve relevant context
        docs = knowledgebase.similarity_search(query, k=4)

        # --- CHANGED: LLM initialization to use Gemini ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Replaced gpt-4o-mini
            temperature=0.8           # Temperature tuning remains the same
        )
        
        query1 = """Generate 5-7 learning multiple choice questions based on the PDF content in a formal question paper format.

For each question, use the following format exactly:

Q1. [Question text here?]

    A) [First option]
    B) [Second option]
    C) [Third option]
    D) [Fourth option]

    Correct Answer: [Letter]
    Explanation: [Brief explanation]

---

Q2. [Next question text?]

    A) [First option]
    B) [Second option]
    C) [Third option]
    D) [Fourth option]

    Correct Answer: [Letter]
    Explanation: [Brief explanation]

Continue this format for all questions. Each question should be separated by a line "---". Make sure each option is clearly on a separate line with proper indentation."""

        docs1 = knowledgebase.similarity_search(query, k=4)

        chain = load_qa_chain(llm, chain_type="stuff")

        response1 = chain.run(
            input_documents=docs,
            question=query
        )
        
        response2 = chain.run(
            input_documents = docs1,
            question = query1
        )
        
        # Generate German translation with the actual summary
        query2 = f"""Translate the following summary to German. Keep the format and structure exactly the same:

{response1}"""
        
        response_german = chain.run(
            input_documents = docs,
            question = query2
        )
        response_summary = response1
        response_question = response2

        

        return {"response_summary":response_summary,"translation":response_german,"response_question":response_question}