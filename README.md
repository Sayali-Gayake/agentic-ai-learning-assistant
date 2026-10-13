# Agentic AI Learning & Career Assistant

A local-first AI learning assistant that helps learners study Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), Generative AI, and related topics. It combines PDF-based Retrieval-Augmented Generation (RAG), personalized study-plan generation, quiz generation, and LangGraph-based request routing in a Streamlit interface.

The project uses a locally running LLM through Ollama and a local FAISS vector index. No paid AI API key is required. An internet connection is needed initially to install packages and download the language and embedding models.

## Features

### 1. Chat with Learning Documents
- Search a knowledge base containing multiple PDF documents.
- Ask questions about topics covered by your documents.
- Retrieve relevant text passages using embeddings and FAISS similarity search.
- Generate answers grounded in retrieved passages.
- Show source filenames and PDF page numbers.

### 2. Document Collection and PDF Uploads
- Store permanent PDFs in the `documents/` directory.
- Organize documents by subject.
- Upload additional PDFs through the Streamlit sidebar.
- Preserve source filenames and page metadata.
- Build a searchable FAISS index from permanent documents.

### 3. AI Study Plan Generator
- Generate personalized learning roadmaps.
- Customize the topic, skill level, study hours, duration, and learning goal.
- Download the generated study plan as a Markdown file.

### 4. Quiz Generator
- Generate five multiple-choice questions.
- Choose Beginner, Intermediate, or Advanced difficulty.
- Optionally use indexed PDFs as the primary source for questions.
- Submit answers and receive a score, correct answers, and explanations.
- View source documents used when PDF context is retrieved.

### 5. LangGraph Request Routing
The LangGraph workflow routes chat requests to one of four tools:

1. **Document Q&A** — retrieves information from the PDF knowledge base.
2. **Study Plan Generator** — creates a learning roadmap.
3. **Quiz Generator** — generates a quiz that can be answered in the Quiz tab.
4. **General Assistant** — handles greetings and general questions.

The current workflow selects one tool per request. It does not yet perform autonomous multi-step tool chaining.

## Technology Stack

- **Language:** Python
- **Interface:** Streamlit
- **LLM:** Llama 3.2 3B through Ollama
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS
- **RAG and PDF Processing:** LangChain, PyPDF
- **Agent Workflow:** LangGraph



## Future Improvements

- Multi-step agent workflows.
- Improved retrieval and answer evaluation.
- Persistent quiz history and learning progress.
- Automated testing and expanded error handling.
