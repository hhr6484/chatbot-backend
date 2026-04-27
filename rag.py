from pypdf import PdfReader
from typing import Union, Optional

def load_pdf_text(file_path: str, max_pages: Optional[int] = None) -> str:
    """Extract text from PDF file with error handling and page limits."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            
        text = ""
        page_count = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            if max_pages and i >= max_pages:
                break
                
            page_text = page.extract_text()
            if page_text:  # Skip empty pages
                text += page_text + "\n\n"
                
        return text.strip()
        
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {file_path}: {str(e)}")