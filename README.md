# Description

Multimodal PDF to Markdown Converter Server - A FastAPI-based service that converts PDF documents into structured markdown format using vision-language models, designed for integration with OpenWebUI's external content extraction system.

# Multimodal PDF to Markdown Converter

A FastAPI-based server that converts PDF documents into structured markdown format using vision-language models. Designed specifically for integration with OpenWebUI's external content extraction system.

## Features

- **PDF Processing**: Converts multi-page PDFs into markdown format
- **Vision-Language Model Integration**: Uses OPENAI AI compatible endpoints for content understanding
- **Concurrent Processing**: Processes PDF pages in parallel for improved performance
- **Robust Error Handling**: Implements retry logic for API failures
- **OpenWebUI Compatible**: Follows OpenWebUI's ExternalDocumentLoader interface
- **Environment Configuration**: Securely manages API keys and configuration via environment variables

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- PyMuPDF (fitz)
- Requests
- Pillow
- python-dotenv

## Installation

```bash
pip install fastapi uvicorn pymupdf requests pillow python-dotenv
```

## Usage

1. Create a `.env` file with your configuration:
```env
OVH_API_ENDPOINT=https://your-endpoint.com/v1/chat/completions
OVH_API_KEY=your-api-key-here
MAX_CONCURRENT_REQUESTS=40
MAX_RETRIES=3
RETRY_DELAY=2
```

2. Run the server:
```bash
uvicorn multimodal_pdf_server:app --host 0.0.0.0 --port 8000
```

3. Configure OpenWebUI to use this endpoint as External Content Extraction Engine

## Endpoints

- `PUT /process` - Process documents (external content extraction)
- `GET /health` - Health check endpoint
- `GET /` - Root endpoint with usage information

## Configuration

All configuration parameters can be set via environment variables:
- `OVH_API_ENDPOINT`: The API endpoint URL
- `OVH_API_KEY`: Your API authentication key
- `MAX_CONCURRENT_REQUESTS`: Maximum number of concurrent API calls
- `MAX_RETRIES`: Number of retries for failed requests
- `RETRY_DELAY`: Base delay between retries (seconds)
