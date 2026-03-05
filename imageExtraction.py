import streamlit as st
import os
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import re
import pandas as pd

from text import*


def clean_image_remove_header_footer(image_array):
    """
    Remove headers and footers from image by detecting text in top/bottom regions.
    Returns cleaned image with headers/footers cropped out.
    """
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        height, width = gray.shape
        
        # Detect text regions
        _, text_binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        text_regions = cv2.bitwise_not(text_binary)
        
        # Dilate to make text regions more prominent
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_dilated = cv2.dilate(text_regions, kernel, iterations=1)
        
        # Find contours (text regions)
        contours, _ = cv2.findContours(text_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Define header and footer thresholds
        header_threshold = height * 0.15  # Top 15%
        footer_threshold = height * 0.85  # Bottom 15%
        
        header_bottom = 0
        footer_top = height
        
        # Scan for text in header and footer regions
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area < 10:  # Skip tiny noise
                continue
            
            # Check if in header region
            if y < header_threshold:
                header_bottom = max(header_bottom, y + h)
            
            # Check if in footer region
            if y > footer_threshold:
                footer_top = min(footer_top, y)
        
        # Add small margin
        crop_top = max(0, header_bottom + 5)
        crop_bottom = min(height, footer_top - 5)
        
        # Ensure valid region
        if crop_bottom <= crop_top:
            return image_array
        
        # Crop the image
        if len(image_array.shape) == 3:
            cleaned = image_array[crop_top:crop_bottom, :, :]
        else:
            cleaned = image_array[crop_top:crop_bottom, :]
        
        return cleaned
        
    except Exception as e:
        print(f"Error cleaning image: {e}")
        return image_array


def upload_and_process_pdf(pdf):
    """Upload PDF and get processed data from backend"""
    try:
        file_bytes = pdf.getvalue()
        st.info("Uploading PDF to backend...")
        
        doc_json = requests.post("http://localhost:8000/upload",
            files={"file": (pdf.name, file_bytes, "application/pdf")},
            timeout=30
        )
        
        if doc_json.status_code != 200:
            st.error(f"Upload failed with status {doc_json.status_code}")
            st.error(f"Response: {doc_json.text}")
            return None, None
        
        response_json = doc_json.json()
        doc_id = response_json.get("doc_id")
        if not doc_id:
            st.error("No doc_id returned from upload")
            return None, None
        
        st.info("Fetching processed data from backend...")
        data = requests.get(f"http://localhost:8000/documents/{doc_id}", timeout=30)
        
        if data.status_code != 200:
            st.error(f"Failed to fetch document with status {data.status_code}")
            st.error(f"Response: {data.text}")
            return None, None
        
        response_data = data.json()
        return response_data, doc_id
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API at http://localhost:8000")
        st.error("Make sure the FastAPI server is running: `uvicorn main:app --reload`")
        return None, None
    except requests.exceptions.Timeout:
        st.error("⏱️ Backend request timed out. The server may be overloaded.")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error during API request: {e}")
        return None, None


def main():
   st.set_page_config(
       page_title="PDF Summarizer",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   
   # Initialize session state
   if "response_data" not in st.session_state:
       st.session_state.response_data = None
   if "summary_result" not in st.session_state:
       st.session_state.summary_result = None
   if "doc_id" not in st.session_state:
       st.session_state.doc_id = None
   
   st.title("📄 Data2Learn - PDF Summarizer")
   st.write("Summarize PDF files in seconds")
   st.divider()

   # Sidebar for upload
   with st.sidebar:
       st.header("📤 Upload & Process")
       pdf = st.file_uploader("Upload PDF File", type='pdf')
       
       os.environ["GOOGLE_API_KEY"] = "*"
       
       submit = st.button("🚀 Summarize", use_container_width=True)
       
       if submit:
           if pdf is None:
               st.error("Please upload a PDF file first")
           else:
               # Upload and process
               response_data, doc_id = upload_and_process_pdf(pdf)
               
               if response_data and doc_id:
                   st.session_state.response_data = response_data
                   st.session_state.doc_id = doc_id
                   
                   # Generate summary
                   st.info("Generating summary and translation...")
                   summary_result = summarizer(pdf)
                   st.session_state.summary_result = summary_result
                   st.success("✅ Processing complete!")
   
   # Create tabs for different sections
   tab1, tab2, tab3, tab4 = st.tabs(
       ["📋 Summary", "❓ Questions", "🌐 Translation", "📊 Extracted Content"]
   )
   
   if st.session_state.response_data is None:
       st.info("👈 Please upload and process a PDF using the sidebar to get started!")
       return
   
   response_data = st.session_state.response_data
   summary_result = st.session_state.summary_result or {}
   
   # Extract summary and translation
   if isinstance(summary_result, dict):
       response = summary_result.get("response_summary", "")
       response_question = summary_result.get("response_question","")
       translation = summary_result.get("translation", "")
   else:
       response = summary_result
       translation = ""
       response_question = ""
   
   # TAB 1: Summary
   with tab1:
       st.header("Summary")
       if response:
           st.write(response)
       else:
           st.info("No summary available. Process a PDF first.")
   
   # TAB 2: Questions
   with tab2:
       st.header("📚 Learning Questions & Answers")
       if "Q1" in response_question or "Q2" in response_question:
           # Split by question separator
           questions = re.split(r'---', response_question)
           
           question_count = 0
           for section in questions:
               if re.search(r'Q\d+', section):
                   question_count += 1
                   # Extract question number
                   q_match = re.search(r'Q(\d+)', section)
                   if q_match:
                       q_num = q_match.group(1)
                       
                       with st.expander(f"❓ Question {q_num}", expanded=False):
                           # Format the question nicely
                           st.markdown(section.strip())
           
           if question_count == 0:
               st.info("No questions found in the summary.")
       else:
           st.info("No questions found in the summary.")
   
   # TAB 3: Translation
   with tab3:
       st.header("🇩🇪 German Translation")
       if translation:
           st.write(translation)
       else:
           st.info("No translation available for this document.")
   
   # TAB 4: Extracted Content
   with tab4:
       # Create subtabs for different content types
       content_tabs = st.tabs(["� Reading Sequence", "🖼️ Charts & Graphs", "📑 Tables", "📝 Text Components"])
       
       # Subtab 0: Reading Sequence (Preserved Order with Spatial Relationships)
       with content_tabs[0]:
           st.subheader("Document Reading Sequence")
           st.info("Content is organized by page and reading order (top-to-bottom, left-to-right) with spatial relationships preserved.")
           
           # Organize all content by page and position
           page_content = {}
           
           # Add text components with position info
           if "components" in response_data:
               for comp in response_data["components"]:
                   page = comp.get("position", {}).get("page", comp.get("page", 1))
                   if page not in page_content:
                       page_content[page] = []
                   bbox = comp.get("position", {}).get("bbox", [0, 0, 0, 0])
                   page_content[page].append({
                       "type": "text",
                       "content": comp,
                       "y_position": bbox[1] if bbox else 0,
                       "x_position": bbox[0] if bbox else 0
                   })
           
           # Add graphics with position info
           if "graphics" in response_data:
               for graphic in response_data["graphics"]:
                   page = graphic.get("page", 1)
                   if page not in page_content:
                       page_content[page] = []
                   bbox = graphic.get("bbox", [0, 0, 0, 0])
                   page_content[page].append({
                       "type": "graphic",
                       "content": graphic,
                       "y_position": bbox[1] if bbox else 0,
                       "x_position": bbox[0] if bbox else 0
                   })
           
           # Add tables with position info
           if "tables" in response_data:
               for table in response_data["tables"]:
                   page = table.get("page", 1)
                   if page not in page_content:
                       page_content[page] = []
                   bbox = table.get("bbox", [0, 0, 0, 0])
                   page_content[page].append({
                       "type": "table",
                       "content": table,
                       "y_position": bbox[1] if bbox else 0,
                       "x_position": bbox[0] if bbox else 0
                   })
           
           # Display by page in order
           if page_content:
               for page_num in sorted(page_content.keys()):
                   with st.expander(f"📄 Page {page_num}", expanded=True):
                       # Sort content on this page by position (top-to-bottom, then left-to-right)
                       page_items = page_content[page_num]
                       page_items.sort(key=lambda x: (x["y_position"], x["x_position"]))
                       
                       for item in page_items:
                           item_type = item["type"]
                           content = item["content"]
                           
                           if item_type == "text":
                               comp_type = content.get("type", "text")
                               text_content = content.get("content", "")
                               bbox = content.get("position", {}).get("bbox", [0, 0, 0, 0])
                               
                               # Show position indicator
                               st.markdown(f"**[{comp_type.upper()}]** (Position: {bbox[0]:.1f}, {bbox[1]:.1f})")
                               st.write(text_content)
                               st.divider()
                           
                           elif item_type == "graphic":
                               img_path = content.get("file_path", "")
                               bbox = content.get("bbox", [0, 0, 0, 0])
                               img_url = f"http://localhost:8000{img_path}" if img_path.startswith("/") else f"http://localhost:8000/{img_path}"
                               
                               st.markdown(f"**[CHART/GRAPH]** (Position: {bbox[0]:.1f}, {bbox[1]:.1f})")
                               try:
                                   img_response = requests.get(img_url, timeout=10)
                                   if img_response.status_code == 200:
                                       img = Image.open(BytesIO(img_response.content))
                                       img_array = np.array(img)
                                       cleaned_array = clean_image_remove_header_footer(img_array)
                                       cleaned_img = Image.fromarray(cleaned_array)
                                       st.image(cleaned_img, use_container_width=True)
                               except Exception as e:
                                   st.warning(f"Could not load graphic: {e}")
                               st.divider()
                           
                           elif item_type == "table":
                               bbox = content.get("bbox", [0, 0, 0, 0])
                               st.markdown(f"**[TABLE]** (Position: {bbox[0]:.1f}, {bbox[1]:.1f})")
                               if "data" in content and content["data"]:
                                   try:
                                       df = pd.DataFrame(content["data"])
                                       st.dataframe(df, use_container_width=True)
                                   except Exception:
                                       st.write(content["data"])
                               st.divider()
           else:
               st.info("No content extracted from this document.")
       
       # Subtab 1: Charts & Graphs (Vector Graphics)
       with content_tabs[1]:
           st.subheader("Extracted Charts & Graphs")
           if "graphics" in response_data and response_data["graphics"]:
               # Filter to show only vector graphics (charts and graphs)
               charts_and_graphs = [g for g in response_data["graphics"] if "Vector graphic" in g.get("summary", "")]
               
               if charts_and_graphs:
                   for graphic in charts_and_graphs:
                       img_path = graphic.get("file_path", "")
                       summary = graphic.get("summary", "Chart/Graph")
                       
                       # Construct full URL for image display
                       img_url = f"http://localhost:8000{img_path}" if img_path.startswith("/") else f"http://localhost:8000/{img_path}"
                       
                       with st.expander(f"📊 Page {graphic.get('page', '?')} - Chart/Graph", expanded=False):
                           try:
                               # Download and clean the image
                               img_response = requests.get(img_url, timeout=10)
                               if img_response.status_code == 200:
                                   # Convert to PIL Image
                                   img = Image.open(BytesIO(img_response.content))
                                   img_array = np.array(img)
                                   
                                   # Clean image (remove headers/footers)
                                   cleaned_array = clean_image_remove_header_footer(img_array)
                                   cleaned_img = Image.fromarray(cleaned_array)
                                   
                                   # Display cleaned image
                                   st.image(cleaned_img, use_container_width=True)
                               else:
                                   st.warning(f"Could not load image from {img_url}")
                           except Exception as e:
                               st.warning(f"Error processing image: {e}")
                               st.image(img_url, use_container_width=True)
               else:
                   st.info("No charts or graphs extracted from this document.")
       
       # Subtab 2: Tables
       with content_tabs[2]:
           st.subheader("Extracted Tables")
           if "tables" in response_data and response_data["tables"]:
               for idx, table in enumerate(response_data["tables"]):
                   with st.expander(f"📊 Table {idx + 1} (Page {table.get('page', '?')})", expanded=False):
                       # Display table data
                       if "data" in table and table["data"]:
                           try:
                               df = pd.DataFrame(table["data"])
                               st.dataframe(df, use_container_width=True)
                           except Exception as e:
                               st.write(table["data"])
                       
                       # Display caption if available
                       if "caption" in table and table["caption"]:
                           st.caption(f"📝 {table['caption']}")
           else:
               st.info("No tables extracted from this document.")
       
       # Subtab 3: Text Components
       with content_tabs[3]:
           st.subheader("Extracted Text Components")
           if "components" in response_data and response_data["components"]:
               # Show all components (removed sentence filtering for full editability)
               all_components = response_data["components"]
               
               if all_components:
                   for comp_idx, comp in enumerate(all_components):
                       # Handle both 'content' and 'text' field names
                       text_content = comp.get("content", comp.get("text", ""))
                       comp_type = comp.get("type", "text")
                       comp_id = comp.get("id")
                       page = comp.get("page", comp.get("position", {}).get("page", "?"))
                       
                       with st.expander(f"📝 Page {page} ({comp_type}) - {len(text_content.split())} words", expanded=False):
                           col1, col2 = st.columns([4, 1])
                           
                           with col1:
                               # Edit content
                               new_content = st.text_area(
                                   "Edit text:",
                                   value=text_content,
                                   height=150,
                                   key=f"comp_content_{comp_idx}"
                               )
                           
                           with col2:
                               # Save button
                               if comp_id and st.button("💾 Save", key=f"comp_save_{comp_idx}"):
                                   try:
                                       update_payload = {"content": new_content}
                                       response = requests.put(
                                           f"http://localhost:8000/text-components/{comp_id}",
                                           json=update_payload,
                                           timeout=10
                                       )
                                       if response.status_code == 200:
                                           st.success("✅ Text saved!")
                                       else:
                                           st.error(f"Failed to save: {response.text}")
                                   except Exception as e:
                                       st.error(f"Error saving text: {e}")
               else:
                   st.info("No text components extracted from this document.")
           else:
               st.info("No text components extracted from this document.")
       
       # TAB 5: Editable Chunks
       st.divider()
       st.subheader("📚 Editable Content Chunks")
       st.info("Chunks with 150+ words. Chunks with 200+ words have summarization available.")
       
       if "chunks" in response_data and response_data["chunks"]:
           for chunk_idx, chunk in enumerate(response_data["chunks"]):
               chunk_id = chunk.get("id")
               word_count = len(chunk.get("content", "").split())
               
               with st.expander(f"📖 {chunk.get('title', 'Untitled')} (Page {chunk.get('page', '?')}) - {word_count} words", expanded=False):
                   col1, col2 = st.columns([3, 1])
                   
                   with col1:
                       # Edit title
                       new_title = st.text_input("Title:", value=chunk.get("title", ""), key=f"title_{chunk_idx}")
                       
                       # Edit content
                       new_content = st.text_area("Content:", value=chunk.get("content", ""), height=200, key=f"content_{chunk_idx}")
                       
                       # Edit or view summary
                       new_summary = st.text_area("Summary:", value=chunk.get("summary", ""), height=100, key=f"summary_{chunk_idx}")
                   
                   with col2:
                       # Save button
                       if st.button("💾 Save", key=f"save_{chunk_idx}"):
                           try:
                               update_payload = {
                                   "title": new_title,
                                   "content": new_content,
                                   "summary": new_summary
                               }
                               response = requests.put(
                                   f"http://localhost:8000/chunks/{chunk_id}",
                                   json=update_payload,
                                   timeout=10
                               )
                               if response.status_code == 200:
                                   st.success("✅ Chunk saved!")
                               else:
                                   st.error(f"Failed to save: {response.text}")
                           except Exception as e:
                               st.error(f"Error saving chunk: {e}")
                       
                       # AI Summarize button (for chunks > 200 words)
                       if word_count > 200 and st.button("✨ Auto Summarize", key=f"summarize_{chunk_idx}"):
                           try:
                               from text import ChatGoogleGenerativeAI, load_qa_chain
                               os.environ["GOOGLE_API_KEY"] = "AIzaSyC1HE0jv9niWFpF5JWwMwYNs4AVcHQygpo"
                               
                               llm = ChatGoogleGenerativeAI(
                                   model="gemini-2.5-flash",
                                   temperature=0.7
                               )
                               
                               summarization_prompt = f"""Please provide a concise summary (2-3 sentences) of the following content:

{new_content}

Summary:"""
                               
                               response = llm.invoke(summarization_prompt)
                               st.session_state[f"summary_{chunk_idx}"] = response.content
                               st.rerun()
                           except Exception as e:
                               st.error(f"Error generating summary: {e}")
               
               # Display questions if available
               if chunk.get("questions"):
                   with st.expander(f"❓ Study Questions for '{chunk.get('title', 'Untitled')}'"):
                       for q_idx, question in enumerate(chunk["questions"], 1):
                           st.write(f"**Q{q_idx}:** {question}")
       else:
           st.info("No content chunks available. Chunks need at least 150 words to be included.")
       
       # Debug section
       with st.expander("🔍 Debug - Full API Response"):
           st.json(response_data)




if __name__ == '__main__':

    main()
