import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import pickle
import os

class Doc2VecEmbeddings:
    def __init__(self, vector_size=300, min_count=2, epochs=20):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
        self.model_path = 'data/doc2vec_model'
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def _preprocess_text(self, text):
        """Preprocess text for Doc2Vec."""
        return simple_preprocess(text, deacc=True)
    
    def fit(self, texts):
        """Train the Doc2Vec model on the provided texts."""
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Create tagged documents
        tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_texts)]
        
        # Initialize and train model
        self.model = Doc2Vec(vector_size=self.vector_size, 
                            min_count=self.min_count, 
                            epochs=self.epochs)
        self.model.build_vocab(tagged_docs)
        self.model.train(tagged_docs, 
                        total_examples=self.model.corpus_count, 
                        epochs=self.epochs)
        
        # Save the model
        self._save_model()
    
    def embed(self, text):
        """Generate embedding for a single text."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        processed_text = self._preprocess_text(text)
        return self.model.infer_vector(processed_text)
    
    def embed_batch(self, texts):
        """Generate embeddings for a batch of texts."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        processed_texts = [self._preprocess_text(text) for text in texts]
        return np.array([self.model.infer_vector(doc) for doc in processed_texts])
    
    def _save_model(self):
        """Save the trained model to disk."""
        if self.model is not None:
            self.model.save(self.model_path)
    
    def _load_model(self):
        """Load a trained model from disk if it exists."""
        if os.path.exists(self.model_path):
            try:
                self.model = Doc2Vec.load(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None