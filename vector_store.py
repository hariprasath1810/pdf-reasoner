import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional
from embeddings import Doc2VecEmbeddings

class FAISSVectorStore:
    def __init__(self):
        # Initialize the Doc2Vec embeddings model
        self.embeddings = Doc2VecEmbeddings(vector_size=300)
        self.dimension = 300  # Doc2Vec dimension
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store document metadata
        self.metadata: Dict[str, List[Tuple[str, int]]] = {}  # doc_id -> list of (chunk, page_num)
        self.doc_ids: List[str] = []  # To map FAISS index positions to doc_ids
        self.filenames: Dict[str, str] = {}  # doc_id -> filename mapping
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
    
    def add_document(self, doc_id: str, chunks_with_pages: List[Tuple[str, int]], filename: str) -> None:
        """Add a document's chunks to the vector store."""
        # Store metadata
        self.metadata[doc_id] = chunks_with_pages
        self.filenames[doc_id] = filename
        
        # Get embeddings for chunks
        chunks = [chunk for chunk, _ in chunks_with_pages]
        
        # Fit the model if this is the first document
        if not self.embeddings.model:
            self.embeddings.fit(chunks)
        
        # Get embeddings
        embeddings = self.embeddings.embed_batch(chunks)
        
        # Add to FAISS index
        start_idx = len(self.doc_ids)
        self.index.add(embeddings)
        
        # Store doc_ids for each embedding
        self.doc_ids.extend([doc_id] * len(chunks))
        
        # Save the updated state
        self._save_state()
    
    def search(self, query: str, doc_id: str, k: int = 5) -> List[Tuple[str, int]]:
        """Search for relevant chunks within a specific document."""
        if not self.exists(doc_id):
            return []
        
        # Get query embedding
        query_embedding = self.embeddings.embed(query).reshape(1, -1)
        
        # Search in FAISS
        D, I = self.index.search(query_embedding, len(self.doc_ids))  # Search all vectors
        
        # Filter results for the specific document and get top k
        results = []
        seen_chunks = set()  # To avoid duplicates
        
        for idx in I[0]:  # I[0] because we only have one query
            if idx >= 0 and idx < len(self.doc_ids):  # Valid index check
                result_doc_id = self.doc_ids[idx]
                if result_doc_id == doc_id:  # Only include results from the requested document
                    chunk, page = self.metadata[doc_id][idx - self._get_doc_start_idx(doc_id)]
                    if chunk not in seen_chunks:  # Avoid duplicates
                        results.append((chunk, page))
                        seen_chunks.add(chunk)
                        if len(results) >= k:
                            break
        
        return results
    
    def exists(self, doc_id: str) -> bool:
        """Check if a document exists in the vector store."""
        return doc_id in self.metadata
    
    def get_filename(self, doc_id: str) -> Optional[str]:
        """Get the filename associated with a document ID."""
        return self.filenames.get(doc_id)
    
    def _get_doc_start_idx(self, doc_id: str) -> int:
        """Get the starting index for a document in the FAISS index."""
        count = 0
        for d_id in self.doc_ids:
            if d_id == doc_id:
                return count
            count += 1
        return 0
    
    def _save_state(self) -> None:
        """Save the vector store state to disk."""
        # Save FAISS index
        faiss.write_index(self.index, 'data/faiss.index')
        
        # Save metadata
        state = {
            'metadata': self.metadata,
            'doc_ids': self.doc_ids,
            'filenames': self.filenames
        }
        with open('data/vector_store_state.pkl', 'wb') as f:
            pickle.dump(state, f)
    
    def _load_state(self) -> bool:
        """Load the vector store state from disk."""
        try:
            # Load FAISS index
            if os.path.exists('data/faiss.index'):
                self.index = faiss.read_index('data/faiss.index')
            
            # Load metadata
            if os.path.exists('data/vector_store_state.pkl'):
                with open('data/vector_store_state.pkl', 'rb') as f:
                    state = pickle.load(f)
                    self.metadata = state['metadata']
                    self.doc_ids = state['doc_ids']
                    self.filenames = state['filenames']
            return True
        except Exception as e:
            print(f"Error loading vector store state: {e}")
            return False

# Initialize the vector store
vector_store = FAISSVectorStore()
# Try to load existing state
vector_store._load_state()