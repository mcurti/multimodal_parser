# multimodal_pdf_server.py
import tkinter as tk
from tkinter import filedialog
import fitz  # PyMuPDF
import requests
import base64
from PIL import Image
import io
import os
from datetime import datetime
from typing import List, Dict, Any
import tempfile
import logging
import concurrent.futures
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.responses import JSONResponse
import uvicorn
from urllib.parse import unquote
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Configuration - loaded from environment variables
OVH_API_ENDPOINT = os.getenv("OVH_API_ENDPOINT", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions")
OVH_API_KEY = os.getenv("OVH_API_KEY", "somekey")

# Configuration for concurrent processing
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "40"))  # Limit concurrent API calls
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))               # Number of retries for failed requests
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))               # Seconds to wait between retries

app = FastAPI(title="Multimodal PDF Processor Server", version="1.0.0")

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def query_ovh_vlm_single_with_retry(prompt: str, image_base64: str, page_index: int, retry_count: int = 0) -> Dict[str, Any]:
    """Send single image to OVH VLM API with retry logic"""
    headers = {
        "Authorization": f"Bearer {OVH_API_KEY}",
        "Content-Type": "application/json"
    }
    # Proper OpenAI-compatible API format
    payload = {
        "model": "Qwen2.5-VL-72B-Instruct",  # Specify the model name
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }
    try:
        log.info(f"Sending request for page {page_index + 1} (attempt {retry_count + 1}) to OVH API...")
        response = requests.post(
            OVH_API_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=300
        )
        log.info(f"Page {page_index + 1} Status Code: {response.status_code}")
        if response.status_code == 200:
            response.raise_for_status()
            result = response.json()
            return {
                "page_index": page_index,
                "result": result.get('choices', [{}])[0].get('message', {}).get('content', 'No response from API'),
                "success": True
            }
        elif response.status_code in [500, 502, 503, 504] and retry_count < MAX_RETRIES:
            # Server-side errors that might recover with retry
            log.warning(f"Server error {response.status_code} for page {page_index + 1}, retrying... (attempt {retry_count + 1})")
            time.sleep(RETRY_DELAY * (2 ** retry_count))  # Exponential backoff
            return query_ovh_vlm_single_with_retry(prompt, image_base64, page_index, retry_count + 1)
        elif response.status_code == 400 and retry_count < MAX_RETRIES:
            # Connection issues that might recover
            log.warning(f"Connection error 400 for page {page_index + 1}, retrying... (attempt {retry_count + 1})")
            time.sleep(RETRY_DELAY * (2 ** retry_count))  # Exponential backoff
            return query_ovh_vlm_single_with_retry(prompt, image_base64, page_index, retry_count + 1)
        else:
            # Non-retryable errors or max retries reached
            error_msg = f"API Error {response.status_code}: {response.text}"
            log.error(f"Non-retryable error for page {page_index + 1}: {error_msg}")
            return {
                "page_index": page_index,
                "result": error_msg,
                "success": False
            }
    except requests.exceptions.ConnectionError as e:
        if retry_count < MAX_RETRIES:
            log.warning(f"Connection error for page {page_index + 1}, retrying... (attempt {retry_count + 1})")
            time.sleep(RETRY_DELAY * (2 ** retry_count))  # Exponential backoff
            return query_ovh_vlm_single_with_retry(prompt, image_base64, page_index, retry_count + 1)
        else:
            log.error(f"Connection error for page {page_index + 1} after {MAX_RETRIES} attempts: {str(e)}")
            return {
                "page_index": page_index,
                "result": f"Connection error after {MAX_RETRIES} attempts: {str(e)}",
                "success": False
            }
    except requests.exceptions.Timeout as e:
        if retry_count < MAX_RETRIES:
            log.warning(f"Timeout for page {page_index + 1}, retrying... (attempt {retry_count + 1})")
            time.sleep(RETRY_DELAY * (2 ** retry_count))  # Exponential backoff
            return query_ovh_vlm_single_with_retry(prompt, image_base64, page_index, retry_count + 1)
        else:
            log.error(f"Timeout for page {page_index + 1} after {MAX_RETRIES} attempts: {str(e)}")
            return {
                "page_index": page_index,
                "result": f"Timeout after {MAX_RETRIES} attempts: {str(e)}",
                "success": False
            }
    except Exception as e:
        log.error(f"Unexpected error for page {page_index + 1}: {str(e)}")
        return {
            "page_index": page_index,
            "result": f"Unexpected error: {str(e)}",
            "success": False
        }

def process_pdf_pages_concurrent(file_path: str, filename: str) -> List[Dict[str, Any]]:
    """Process PDF pages concurrently with retry logic"""
    log.info(f"Processing PDF: {filename} with concurrent approach")
    # Convert PDF to images page by page
    doc = fitz.open(file_path)
    documents = []
    # Prepare prompt for VLM
    prompt = """
    The provided image is a snapshot of a page of a multi-page file. Please convert the content to markdown format using the following guidelines:
    1. If this snapshot is a document such as single or multiple columns paper, report etc ... then perform the following:
        a. Keep the text formating as close as possible. Include all text,
        b. Do not output the markdown box around your reply.
        c. Mind the hierarchy of sections with #, ##, ###, and so on and so forth.
        d. Figure caption starts most of the time with the word "figure". Describe EACH figure in relevant details. Use the context from the page to determine which details should be accounted for.
        Regardless of the format in the document, use the following formatting:
            |FIGURE [#]: [Caption of the figure if exists]|
            |---|
            |[detailed figure description]|
        Example of figure with caption:
            |FIGURE 1.2: The circuit diagram of the prototype|
            |---|
            |The figure features a diagram which includes the electric circuit of ...|
        Example of figure with caption and multiple subfigures:
            |FIGURE 1.3]: The circuit diagram of the prototype, a) conceptual diagram, and b) detailed circuit|
            |---|
            |The figure features a diagram which includes the electric circuit of ...|
            |The subfigure a) features the conceptual diagram of ...|
            |The subfigure b) features the detailed electric circuit ...|
        e. Tables captions most of the times start with the word "table". Convert tables into markdown, please, preserve the formatting and information inside.
        f. DO NOT confuse Figures with Tables. It is okay if on a given page there are no figures or tables or both. Do not mention that in the output, your role is to recognize whatever is in the page, no additional interpretation.
        g.  Equations should be placed between the double dolar sign. In line symbols should be placed in between single dolar sign. Include the tag of the equations. Regardless of the equation format in the snapshot consider the following examples:
        Example of a stand alone equation with tag:
            $$
            a^2 = b^2 + c^2 \tag{1.1}
            $$
            notice that the tag is at the end of the equation. That should be always the case.
        Example of a stand alone equation without a tag
            $$
            a^2 = b^2 + c^2
            $$
        Example of a math symbols inline with text:
            ... where $a \in \mathbb{R}$, and ...
    2. If the snapshot is a slide follow the following guidelines
        a. Include the title of the slide if existing
        b. Describe each image, diagram, figure in detail
        c. Include each table
        d. Include each equation. Only include the tag if it exists in the slide. Follow the guidelines and examples from the document, case g.
        e. Give an overall description of the slide, use the details from a., b., c, and d.
    """
    # Prepare all page data first
    all_page_data = []
    total_pages = len(doc)
    for page_num in range(total_pages):
        try:
            page = doc.load_page(page_num)
            # Get page as image
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            image_b64 = encode_image_to_base64(image)
            all_page_data.append({
                "page_num": page_num,
                "page_object": page,
                "image_b64": image_b64
            })
        except Exception as e:
            log.error(f"Error preparing page {page_num + 1}: {e}")
            # Add error entry for failed page
            documents.append({
                "page_content": f"Error preparing page {page_num + 1}: {str(e)}",
                "metadata": {
                    "source": filename,
                    "page": page_num + 1,
                    "error": str(e),
                    "extraction_method": "error"
                }
            })
    # Process pages concurrently with limited workers
    if all_page_data:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            # Submit all page jobs
            future_to_page = {
                executor.submit(query_ovh_vlm_single_with_retry, prompt, page_data["image_b64"], page_data["page_num"]): page_data
                for page_data in all_page_data
            }
            # Collect results as they complete
            results = []
            for future in concurrent.futures.as_completed(future_to_page):
                page_data = future_to_page[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log.error(f"Page processing error: {e}")
                    # Add error entry for failed page
                    results.append({
                        "page_index": page_data["page_num"],
                        "result": f"Page processing error: {str(e)}",
                        "success": False
                    })
        # Sort results by page index to ensure correct order
        results.sort(key=lambda x: x["page_index"])
        # Create final documents list in correct order
        for result_item in results:
            page_index = result_item["page_index"]
            result = result_item["result"]
            success = result_item.get("success", True)
            # Create document entry
            documents.append({
                "page_content": result + f"\n\n*Page {page_index+1}*\n\n---\n",
                "metadata": {
                    "source": filename,
                    "page": page_index + 1,
                    "total_pages": total_pages,
                    "extraction_method": "multimodal_vlm",
                    "success": success
                }
            })
            log.info(f"Completed page {page_index + 1}/{total_pages} {'(SUCCESS)' if success else '(FAILED)'}")
    doc.close()
    return documents

@app.put("/process")
async def process_document(
    request: Request,
    authorization: str = Header(None),
    x_filename: str = Header(None, alias="X-Filename"),
    content_type: str = Header(None, alias="Content-Type")
):
    """
    Process document via external content extraction engine
    Expected by OpenWebUI's ExternalDocumentLoader
    """
    # Read raw binary data from request body
    file_data = await request.body()
    if not file_data:
        raise HTTPException(status_code=400, detail="No file data received")
    # Get filename from header
    filename = x_filename
    if filename:
        filename = unquote(filename)
    else:
        filename = "unknown_document"
    # Create temporary file from binary data
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file_data)
        temp_file_path = temp_file.name
    try:
        # Process the document based on file type
        # For this server, we focus on PDF processing
        if filename.lower().endswith('.pdf'):
            documents = process_pdf_pages_concurrent(temp_file_path, filename)
        else:
            # For non-PDF files, return basic text extraction
            with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            documents = [{
                "page_content": content,
                "metadata": {
                    "source": filename,
                    "extraction_method": "basic_text"
                }
            }]
        # Return response in expected format
        if len(documents) == 1:
            return JSONResponse(content=documents[0])
        else:
            return JSONResponse(content=documents)
    except Exception as e:
        log.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "multimodal-pdf-processor"}

@app.get("/")
async def root():
    """Root endpoint with usage information"""
    return {
        "message": "Multimodal PDF Processor Server",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST/PUT - Process documents (external content extraction)",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal PDF Processor Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    args = parser.parse_args()
    # Support environment variables
    host = os.environ.get("HOST", args.host)
    port = int(os.environ.get("PORT", args.port))
    print(f"Starting Multimodal PDF Processor Server on {host}:{port}")
    print("Configure OpenWebUI to use this endpoint as External Content Extraction Engine")
    print("Endpoint URL should be: http://<your-server-ip>:8000/process")
    # Run with uvicorn
    uvicorn.run(app, host=host, port=port)
