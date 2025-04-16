from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pdf_processor import process_pdf
from vector_store import vector_store
from agents import summary_agent, query_agent, abstract_agent, proposed_approach_agent, keyword_agent, result_discussion_agent
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Ensure filename is safe
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ('-', '_', '.'))
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Use os.path.join for proper path handling
        file_path = os.path.join("uploads", safe_filename)
        
        # Save the uploaded file
        try:
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
            
        # Process the PDF
        try:
            doc_id = process_pdf(file_path)
        except Exception as e:
            # If PDF processing fails, clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
            
        return {
            "doc_id": doc_id,
            "message": "PDF processed successfully",
            "filename": safe_filename
        }
    except Exception as e:
        # Log the full error for debugging
        print(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}")
async def get_pdf(doc_id: str):
    try:
        # Assuming your vector_store keeps track of filenames
        filename = vector_store.get_filename(doc_id)  # You'll need to implement this
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        return FileResponse(file_path, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{doc_id}")
async def get_summary(doc_id: str):
    if not vector_store.exists(doc_id): # Assuming vector_store has an 'exists' method
        raise HTTPException(status_code=404, detail="Document not found")
    # Use a targeted query for summary
    query = "Provide a detailed technical summary of the entire research paper, covering problem, methods, results, and contributions."
    relevant_chunks = vector_store.search(query, doc_id, k=10) # Fetch top 10 relevant chunks
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="Could not find relevant content for summary.")
    return {"summary": summary_agent.summarize(relevant_chunks)}

@app.get("/abstract/{doc_id}")
async def get_abstract(doc_id: str):
    if not vector_store.exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    # Use a targeted query for abstract
    query = "Extract the abstract and main motivation or objective of this research paper."
    relevant_chunks = vector_store.search(query, doc_id, k=5) # Fetch top 5 relevant chunks (abstracts are usually short)
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="Could not find relevant content for abstract.")
    return {"abstract": abstract_agent.generate_abstract(relevant_chunks)}

@app.get("/approach/{doc_id}")
async def get_proposed_approach(doc_id: str):
    if not vector_store.exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    # Use a targeted query for proposed approach
    query = "Describe the proposed methodology, approach, algorithms, or framework presented in this paper."
    relevant_chunks = vector_store.search(query, doc_id, k=10) # Fetch top 10 relevant chunks
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="Could not find relevant content for proposed approach.")
    return {"approach": proposed_approach_agent.extract_approach(relevant_chunks)}

@app.get("/query/{doc_id}")
async def answer_query(doc_id: str, question: str):
    if not vector_store.exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    # Keep using the user's question for relevance
    relevant_chunks = vector_store.search(question, doc_id, k=10)
    if not relevant_chunks:
        # Try a broader search if the specific question yields nothing
        relevant_chunks = vector_store.search("", doc_id, k=5) # Fallback to top 5 general chunks
        if not relevant_chunks:
             raise HTTPException(status_code=404, detail="Could not find relevant content for this question.")
    return {"answer": query_agent.answer(question, relevant_chunks)}

@app.get("/keywords/{doc_id}")
async def get_keywords(doc_id: str):
    if not vector_store.exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    # Use a targeted query for keywords
    query = "Identify the main topics, concepts, and keywords of this research paper."
    relevant_chunks = vector_store.search(query, doc_id, k=10) # Fetch top 10 relevant chunks
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="Could not find relevant content for keywords.")
    return {"keywords": keyword_agent.extract_keywords(relevant_chunks)}

@app.get("/results/{doc_id}")
async def get_results_discussion(doc_id: str):
    if not vector_store.exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    # Use a targeted query for results and discussion
    query = "Extract the results, findings, and discussion sections of this research paper."
    relevant_chunks = vector_store.search(query, doc_id, k=15) # Fetch top 15 relevant chunks for comprehensive results
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="Could not find relevant content for results and discussion.")
    return {"results_discussion": result_discussion_agent.extract_results_discussion(relevant_chunks)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)