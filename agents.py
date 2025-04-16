from autogen import AssistantAgent, UserProxyAgent
import ollama

def clean_response(text):
    """Remove newline characters and join into a single paragraph."""
    return ' '.join(text.split())

class SummaryAgent(AssistantAgent):
    def __init__(self, name="SummaryAgent"):
        super().__init__(name=name)
        self.role = "Summarize the PDF content"

    def summarize(self, chunks_with_pages):
        chunks = [chunk for chunk, _ in chunks_with_pages]
        prompt = f"""
You are an expert research assistant. Summarize the research paper using only the following sections: **methodology**, **experiments**, and **results**. Avoid references, literature reviews, and conclusions.

Structure in 200–250 words:
1. **Problem Addressed**
2. **Proposed Methodology**
3. **Experimental Setup and Results**
4. **Key Contributions**

Be clear and concise. Avoid academic jargon.

--- CONTENT START ---
{' '.join(chunks)}
--- CONTENT END ---
"""
        response = ollama.generate(model="llama3:8b", prompt=prompt)
        return clean_response(response['response'])

class AbstractAgent(AssistantAgent):
    def __init__(self, name="AbstractAgent"):
        super().__init__(name=name)
        self.role = "Generate abstract and motive"

    def generate_abstract(self, chunks_with_pages):
        chunks = [chunk for chunk, _ in chunks_with_pages]
        prompt = f"""
Write a clear and concise abstract (250–300 words) for the research paper based only on **background**, **goals**, **methodology**, and **results**. Do not include references or literature reviews.

Structure:
1. **Background**
2. **Research Goals**
3. **Proposed Solution**
4. **Main Results and Contributions**

Make it accessible and academic in tone.

--- CONTENT START ---
{' '.join(chunks)}
--- CONTENT END ---
"""
        response = ollama.generate(model="llama3:8b", prompt=prompt)
        return clean_response(response['response'])

class ProposedApproachAgent(AssistantAgent):
    def __init__(self, name="ProposedApproachAgent"):
        super().__init__(name=name)
        self.role = "Explain proposed approach"

    def extract_approach(self, chunks_with_pages):
        chunks = [chunk for chunk, _ in chunks_with_pages]
        prompt = f"""
Explain the **proposed method** from the research paper in 350–400 words. Use only the methodology content. Exclude references and reviews.

Structure your answer as:
1. **Method Overview**
2. **Detailed Steps**
3. **Implementation Details**
4. **Advantages and Novelty**

Use bullet points or short paragraphs where useful. Be precise and clear.

--- CONTENT START ---
{' '.join(chunks)}
--- CONTENT END ---
"""
        response = ollama.generate(model="llama3:8b", prompt=prompt)
        return clean_response(response['response'])

class QueryAgent(AssistantAgent):
    def __init__(self, name="QueryAgent"):
        super().__init__(name=name)
        self.role = "Answer user questions based on specific sections of the paper"

    def answer(self, query, chunks_with_pages):
        chunks = [chunk for chunk, _ in chunks_with_pages]
        pages = [page for _, page in chunks_with_pages]

        # Step 1: Detect relevant sections based on user query
        section_detection_prompt = f"""
You are an intelligent assistant that classifies academic questions.

Given the user's question:
"{query}"

Decide which sections of a research paper are most relevant to answer it. Choose only from the following list (one or more):

- abstract
- background
- methodology
- proposed approach
- experiments
- results
- discussion
- literature survey
- references
- keywords

Return a comma-separated list of the most relevant sections.
        """
        section_response = ollama.generate(model="llama3:8b", prompt=section_detection_prompt)
        selected_sections = clean_response(section_response['response']).lower().split(',')

        # Step 2: Collect relevant chunks based on section names
        relevant_chunks = []
        for section in selected_sections:
            section = section.strip()
            if "abstract" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "abstract" in chunk.lower()])
            if "background" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "background" in chunk.lower()])
            if "methodology" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "methodology" in chunk.lower() or "method" in chunk.lower()])
            if "proposed approach" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "proposed" in chunk.lower() or "approach" in chunk.lower()])
            if "experiments" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "experiment" in chunk.lower()])
            if "results" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "result" in chunk.lower()])
            if "discussion" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "discussion" in chunk.lower()])
            if "literature" in section or "survey" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "literature" in chunk.lower() or "survey" in chunk.lower() or "related work" in chunk.lower()])
            if "references" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "reference" in chunk.lower() or "bibliography" in chunk.lower()])
            if "keywords" in section:
                relevant_chunks.extend([chunk for chunk in chunks if "keyword" in chunk.lower()])

        # Step 3: Fallback if nothing matched
        if not relevant_chunks:
            relevant_chunks = chunks  # fallback to entire content

        content = ' '.join(relevant_chunks)

        # Step 4: Final response generation
        answering_prompt = f"""
Use the following content from a research paper to answer the user's question.

**User's Question:** {query}

Be precise and academic. Cite page numbers where possible using [Page X].

--- CONTENT START ---
{content}
--- CONTENT END ---

Pages: {pages}
Chunks used: {len(relevant_chunks)}
        """
        response = ollama.generate(model="llama3:8b", prompt=answering_prompt)
        return clean_response(response['response'])

class KeywordAgent(AssistantAgent):
    def __init__(self, name="KeywordAgent"):
        super().__init__(name=name)
        self.role = "Extract PDF keywords"

    def extract_keywords(self, chunks_with_pages):
        chunks = [chunk for chunk, _ in chunks_with_pages]
        text = ' '.join(chunks)
        prompt = f"""
Extract 8–12 technical **keywords** from the paper, focusing only on methods and results. Exclude references and reviews.

Return only a comma-separated list like:
keyword1, keyword2, keyword3, ...

--- CONTENT START ---
{text}
--- CONTENT END ---
"""
        response = ollama.generate(model="llama3:8b", prompt=prompt)
        return clean_response(response['response'])

class ResultDiscussionAgent(AssistantAgent):
    def __init__(self, name="ResultDiscussionAgent"):
        super().__init__(name=name)
        self.role = "Summarize results and discussion"

    def extract_results_discussion(self, chunks_with_pages):
        chunks = [chunk for chunk, _ in chunks_with_pages]
        pages = [page for _, page in chunks_with_pages]
        prompt = f"""
Summarize the **results and discussion** section of the paper in 350–400 words. Focus only on evaluation, findings, and interpretation. Exclude references, reviews, and conclusions.

Structure your answer:
1. **Evaluation Metrics**
2. **Key Findings**
3. **Discussion and Interpretation**
4. **Page References** (e.g., [Page 7])

--- CONTENT START ---
{' '.join(chunks)}
--- CONTENT END ---
Pages: {pages}
"""
        response = ollama.generate(model="llama3:8b", prompt=prompt)
        return clean_response(response['response'])

# Initialize agents
summary_agent = SummaryAgent()
abstract_agent = AbstractAgent()
proposed_approach_agent = ProposedApproachAgent()
query_agent = QueryAgent()
keyword_agent = KeywordAgent()
result_discussion_agent = ResultDiscussionAgent()

# Initialize UserProxyAgent with Docker disabled
user_proxy = UserProxyAgent(name="UserProxy", code_execution_config={"use_docker": False})
