import fitz  # PyMuPDF
import uuid
import os
from typing import List, Tuple

def extract_text_from_pdf(file_path: str) -> List[Tuple[str, int]]:
    """Extract text from PDF and return list of (text_chunk, page_number) tuples."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
        
    try:
        doc = fitz.open(file_path)
        if not doc.is_pdf:
            raise ValueError("The uploaded file is not a valid PDF")
            
        chunks_with_pages = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text()
                
                # Split text into smaller chunks (e.g., paragraphs)
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                # Add each paragraph with its page number
                for paragraph in paragraphs:
                    if len(paragraph.split()) >= 10:  # Only include chunks with at least 10 words
                        chunks_with_pages.append((paragraph, page_num + 1))
            except Exception as e:
                print(f"Warning: Error processing page {page_num + 1}: {str(e)}")
                continue
                
        doc.close()
        
        if not chunks_with_pages:
            raise ValueError("No text content could be extracted from the PDF")
            
        return chunks_with_pages
        
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

def process_pdf(file_path: str) -> str:
    """Process a PDF file and add it to the vector store."""
    try:
        # Generate a unique document ID
        doc_id = str(uuid.uuid4())
        
        # Extract text chunks with page numbers
        chunks_with_pages = extract_text_from_pdf(file_path)
        
        # Get filename from path
        filename = os.path.basename(file_path)
        
        # Add to vector store
        from vector_store import vector_store
        vector_store.add_document(doc_id, chunks_with_pages, filename)
        
        return doc_id
        
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")