
import io
import json
import re
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from agent import build_agent
from app_embeddings import load_embeddings
from knowledge_search import (
    search_all_knowledge,
    load_saved_knowledge_base,
)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "llama3.2:3b"

PROJECT_DIR = Path(__file__).parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
INDEX_DIR = PROJECT_DIR / "faiss_index"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVAL_K = 4


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Learning & Career Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 AI Learning & Career Assistant")

st.caption(
    "Your local AI learning assistant for Artificial Intelligence, "
    "Machine Learning, Deep Learning, Generative AI, RAG, "
    "and interview preparation."
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "messages": [],
    "uploaded_vector_store": None,
    "uploaded_indexed_files": [],
    "study_plan": "",
    "study_plan_topic": "",
    "quiz_data": None,
    "quiz_results": None,
    "quiz_topic": "",
    "quiz_sources": [],
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================================================
# LOCAL LANGUAGE MODEL AND EMBEDDINGS
# =========================================================

@st.cache_resource
def load_llm():
    """Load the local Ollama language model."""
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0.2,
    )


@st.cache_resource
def load_upload_embeddings():
    """Reuse the project's local embedding configuration."""
    return load_embeddings()


def invoke_local_llm(prompt: str) -> str:
    """Generate a response using the local Ollama model."""
    response = load_llm().invoke(prompt)
    return str(response.content)


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_uploaded_pdfs(uploaded_files):
    """
    Extract text from uploaded PDFs.

    Each extracted page retains the uploaded filename and
    its original one-based PDF page number.
    """
    documents = []
    errors = []

    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(
                io.BytesIO(uploaded_file.getvalue())
            )

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):
                text = page.extract_text() or ""

                if not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": f"Uploaded/{uploaded_file.name}",
                            "page": page_number,
                        },
                    )
                )

        except Exception as exc:
            errors.append(
                f"{uploaded_file.name}: {exc}"
            )

    return documents, errors


# =========================================================
# UPLOADED PDF INDEXING
# =========================================================

def index_uploaded_pdfs(uploaded_files):
    """Create an in-memory FAISS index for uploaded PDFs."""

    pages, errors = extract_uploaded_pdfs(uploaded_files)

    for error in errors:
        st.warning(f"Could not read PDF: {error}")

    if not pages:
        st.error(
            "No readable PDF text was found. The PDF may be empty, "
            "encrypted, or contain scanned images requiring OCR."
        )
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        st.error("No searchable text chunks were created.")
        return

    try:
        with st.spinner("Creating embeddings for uploaded PDFs..."):
            vector_store = FAISS.from_documents(
                chunks,
                load_upload_embeddings(),
            )

        st.session_state.uploaded_vector_store = vector_store
        st.session_state.uploaded_indexed_files = [
            file.name for file in uploaded_files
        ]

        st.success(
            f"Indexed {len(pages)} pages into "
            f"{len(chunks)} text chunks."
        )

    except Exception as exc:
        st.error("Could not create the uploaded-PDF index.")
        st.exception(exc)


# =========================================================
# DOCUMENT CONTEXT AND SOURCES
# =========================================================

def build_context(documents):
    """Format retrieved passages with filename and page metadata."""

    context_parts = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source: {source} | Page: {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n--- NEXT PASSAGE ---\n\n".join(context_parts)


def get_source_references(documents):
    """Return unique source filenames and page references."""

    references = []
    seen = set()

    for document in documents:
        source = str(
            document.metadata.get("source", "Unknown")
        )
        page = document.metadata.get("page", "Unknown")

        key = (source, str(page))

        if key not in seen:
            seen.add(key)

            references.append({
                "source": source,
                "page": page,
            })

    return references


# =========================================================
# RAG QUESTION ANSWERING
# =========================================================

def answer_from_documents(question):
    """
    Search permanent and uploaded PDFs, then generate an answer
    grounded in the retrieved passages.
    """

    relevant_documents = search_all_knowledge(
        question,
        uploaded_vector_store=(
            st.session_state.uploaded_vector_store
        ),
        k=RETRIEVAL_K,
    )

    if not relevant_documents:
        return (
            "I couldn't find relevant information in the indexed PDFs. "
            "Try rephrasing your question or adding a relevant document. "
            "You can also turn off document search to ask a general question."
        ), []

    context = build_context(relevant_documents)

    prompt = f"""
You are an AI learning assistant and technical instructor.

Answer the user's question using the retrieved document context.

Rules:
1. Ground document-specific claims in the supplied context.
2. Do not invent facts, quotations, or page numbers.
3. If the context does not contain the answer, say so clearly.
4. Explain concepts in clear, student-friendly language.
5. Use examples where appropriate.
6. Cite supporting passages with the exact source and page metadata.
7. Use this citation format:
   [Source: ML/notes.pdf | Page: 12]
8. Do not claim to have read information absent from the context.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    answer = invoke_local_llm(prompt)
    sources = get_source_references(relevant_documents)

    return answer, sources


def answer_general_question(question):
    """Answer a general question without PDF retrieval."""

    prompt = f"""
You are a helpful AI learning assistant.

Explain technical concepts clearly for a student learning
Artificial Intelligence, Machine Learning, Deep Learning,
Generative AI, RAG, and software engineering.

USER QUESTION:
{question}

ANSWER:
"""

    return invoke_local_llm(prompt)


# =========================================================
# QUIZ RESPONSE PARSER
# =========================================================

def parse_quiz_response(raw_response: str) -> list[dict]:
    """Parse and validate five MCQs returned by the local model."""

    raw = raw_response.strip()

    # Remove optional Markdown code fences.
    raw = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "The model did not return a valid JSON object."
        )

    data = json.loads(raw[start:end + 1])
    questions = data.get("questions")

    if not isinstance(questions, list) or len(questions) != 5:
        raise ValueError(
            "Expected exactly five quiz questions."
        )

    validated = []

    for number, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(
                f"Question {number} has an invalid format."
            )

        options = question.get("options")

        answer = str(
            question.get("answer", "")
        ).strip().upper()

        if not isinstance(options, dict):
            raise ValueError(
                f"Question {number} has invalid options."
            )

        required_options = ["A", "B", "C", "D"]

        if not all(
            key in options for key in required_options
        ):
            raise ValueError(
                f"Question {number} must contain options A-D."
            )

        if answer not in required_options:
            raise ValueError(
                f"Question {number} has an invalid answer key."
            )

        question_text = str(
            question.get("question", "")
        ).strip()

        if not question_text:
            raise ValueError(
                f"Question {number} has no question text."
            )

        validated.append({
            "question": question_text,
            "options": {
                key: str(options[key])
                for key in required_options
            },
            "answer": answer,
            "explanation": str(
                question.get(
                    "explanation",
                    "No explanation provided.",
                )
            ),
        })

    return validated


# =========================================================
# QUIZ GENERATION
# =========================================================

def generate_quiz(
    topic,
    difficulty,
    use_documents,
    uploaded_vector_store=None,
):
    """Generate five multiple-choice questions with explanations."""

    context = ""
    source_documents = []

    if use_documents:
        source_documents = search_all_knowledge(
            (
                f"Important definitions, core concepts, "
                f"examples, and facts about {topic}"
            ),
            uploaded_vector_store=uploaded_vector_store,
            k=5,
        )

        if source_documents:
            context = build_context(source_documents)

    quiz_prompt = f"""
You are an experienced technical instructor.

Create exactly five multiple-choice questions.

Topic: {topic}
Difficulty: {difficulty}

Use the document context below as the primary source when available.
Do not invent document-specific facts. If no useful context is available,
use your general knowledge appropriate to the topic and difficulty.

DOCUMENT CONTEXT:
{context if context else "No document context was retrieved."}

Return ONLY one valid JSON object with this structure:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
      }},
      "answer": "A",
      "explanation": "Explain why option A is correct."
    }}
  ]
}}

Requirements:
- Exactly five questions.
- Four options labelled A, B, C, and D.
- Exactly one correct answer per question.
- The answer must be A, B, C, or D.
- Include a clear explanation for every answer.
- Avoid duplicate questions.
- Return valid JSON only, without Markdown code fences.
"""

    raw_response = invoke_local_llm(quiz_prompt)
    questions = parse_quiz_response(raw_response)
    sources = get_source_references(source_documents)

    return questions, sources


# =========================================================
# LANGGRAPH AGENT TOOL FUNCTIONS
# =========================================================

def agent_document_qa_tool(request: str) -> str:
    """Search PDFs and answer a document-related question."""

    if not st.session_state.get("use_document_search", True):
        return answer_general_question(request)

    answer, sources = answer_from_documents(request)

    if sources:
        source_text = "\n".join(
            f"- {item['source']} | Page {item['page']}"
            for item in sources
        )

        answer += f"\n\n**Sources retrieved:**\n{source_text}"

    return answer


def agent_study_plan_tool(request: str) -> str:
    """Generate a study plan from the user's chat request."""

    prompt = f"""
You are an experienced AI and Machine Learning instructor
and career mentor.

Create a practical, detailed study plan based on the request below.

USER REQUEST:
{request}

Requirements:
1. Identify the topic, duration, level, and goal where possible.
2. If details are missing, make reasonable assumptions.
3. Organize the learning topics in a logical sequence.
4. Include concepts, exercises, and coding tasks.
5. Include a practical project where relevant.
6. Include revision and interview practice where appropriate.
7. Prefer free learning resources and local tools.
8. Use clear Markdown headings and lists.
9. State expected learning outcomes.

Generate the complete study plan now.
"""

    plan = invoke_local_llm(prompt)

    # Make the generated plan available in the Study Plan tab.
    st.session_state.study_plan = plan
    st.session_state.study_plan_topic = request[:80]

    return (
        f"## Your Study Plan\n\n{plan}\n\n"
        "You can also view this plan in the AI Study Plan tab."
    )


def agent_quiz_tool(request: str) -> str:
    """Generate a quiz and make it available in the Quiz tab."""

    topic = request.strip()

    patterns = [
        (
            r"^(?:please\s+)?(?:can you\s+)?"
            r"(?:create|generate|make)(?:\s+me)?\s+"
            r"(?:a\s+)?(?:quiz|test|mcqs?|multiple[- ]choice questions?)"
            r"(?:\s+(?:on|about|for))?\s*"
        ),
        r"^(?:please\s+)?quiz\s+me(?:\s+(?:on|about))?\s*",
        r"^(?:please\s+)?test\s+me(?:\s+(?:on|about))?\s*",
    ]

    for pattern in patterns:
        topic = re.sub(
            pattern,
            "",
            topic,
            flags=re.IGNORECASE,
        ).strip()

    topic = topic.strip(" .?!")

    if not topic:
        topic = request.strip()

    request_lower = request.lower()

    if any(
        word in request_lower
        for word in ["advanced", "difficult", "hard"]
    ):
        difficulty = "Advanced"
    elif any(
        word in request_lower
        for word in ["beginner", "easy", "basic"]
    ):
        difficulty = "Beginner"
    else:
        difficulty = "Intermediate"

    quiz, sources = generate_quiz(
        topic=topic,
        difficulty=difficulty,
        use_documents=st.session_state.get(
            "use_document_search",
            True,
        ),
        uploaded_vector_store=(
            st.session_state.uploaded_vector_store
        ),
    )

    # Reset selections and results from the previous quiz.
    for index in range(5):
        st.session_state.pop(
            f"quiz_answer_{index}",
            None,
        )

    st.session_state.quiz_data = quiz
    st.session_state.quiz_results = None
    st.session_state.quiz_topic = topic
    st.session_state.quiz_sources = sources

    return (
        f"Generated a five-question quiz about **{topic}** "
        f"at the **{difficulty}** level.\n\n"
        "Open the Quiz Generator tab to answer the questions, "
        "submit your responses, and see your score and explanations."
    )


def agent_general_tool(request: str) -> str:
    """Handle general conversation."""

    return answer_general_question(request)


@st.cache_resource
def load_agent_graph():
    """Build and cache the LangGraph graph with real application tools."""

    tool_functions = {
        "document_qa": agent_document_qa_tool,
        "study_plan": agent_study_plan_tool,
        "quiz": agent_quiz_tool,
        "general": agent_general_tool,
    }

    return build_agent(tool_functions)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("📚 Knowledge Base")

    permanent_pdfs = (
        list(DOCUMENTS_DIR.rglob("*.pdf"))
        if DOCUMENTS_DIR.exists()
        else []
    )

    index_exists = (
        (INDEX_DIR / "index.faiss").exists()
        and (INDEX_DIR / "index.pkl").exists()
    )

    st.write(
        f"PDFs in documents folder: **{len(permanent_pdfs)}**"
    )

    if index_exists:
        st.success("Saved FAISS index found.")
    else:
        st.warning(
            "Saved index not found. Run `python knowledge_base.py` "
            "to build it."
        )

    st.divider()
    st.subheader("Upload additional PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help=(
            "Uploaded PDFs are indexed for the current Streamlit "
            "session after processing."
        ),
    )

    if st.button(
        "Process uploaded PDFs",
        use_container_width=True,
    ):
        if uploaded_files:
            index_uploaded_pdfs(uploaded_files)
        else:
            st.warning("Please select at least one PDF.")

    if st.session_state.uploaded_indexed_files:
        st.caption("Indexed uploads:")

        for filename in st.session_state.uploaded_indexed_files:
            st.write(f"• {filename}")

    if st.button(
        "Clear uploaded PDF index",
        use_container_width=True,
    ):
        st.session_state.uploaded_vector_store = None
        st.session_state.uploaded_indexed_files = []
        st.success("Uploaded PDF index cleared.")
        st.rerun()

    st.divider()

    st.checkbox(
        "Use document knowledge (RAG)",
        value=True,
        key="use_document_search",
        help=(
            "Search the saved knowledge base and uploaded PDFs "
            "when answering questions or generating document-based quizzes."
        ),
    )

    if st.button(
        "Reload saved knowledge base",
        use_container_width=True,
    ):
        load_saved_knowledge_base.clear()
        st.success("Knowledge-base cache cleared.")
        st.rerun()

    st.divider()

    if st.button(
        "Clear chat history",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Model: Llama 3.2 3B\n\n"
        "Embeddings: all-MiniLM-L6-v2\n\n"
        "Runs locally using Ollama."
    )


# =========================================================
# MAIN TABS
# =========================================================

chat_tab, planner_tab, quiz_tab = st.tabs([
    "💬 Chat with Agent",
    "🗓️ AI Study Plan",
    "🧠 Quiz Generator",
])


# =========================================================
# TAB 1: CHAT WITH AGENT
# =========================================================

with chat_tab:
    st.subheader("Ask your AI Learning Assistant")

    st.caption(
        "Ask questions about your PDFs, request a study plan, "
        "or ask the assistant to quiz you."
    )

    # This container is placed above the chat input.
    chat_history_container = st.container()

    # Streamlit displays the chat input at the bottom of this tab.
    question = st.chat_input(
        "Ask about AI, request a study plan, or say 'quiz me on ML'...",
        key="main_chat_input",
    )

    if question and question.strip():
        question = question.strip()

        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "sources": [],
        })

        try:
            with st.spinner(
                "Agent is deciding which tool to use..."
            ):
                result = load_agent_graph().invoke({
                    "user_input": question,
                })

            answer = result.get(
                "response",
                "The agent did not return a response.",
            )

            selected_tool = result.get(
                "selected_tool",
                "unknown",
            )

            tool_labels = {
                "document_qa": "PDF Knowledge Search",
                "study_plan": "Study Plan Generator",
                "quiz": "Quiz Generator",
                "general": "General Assistant",
            }

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
                "selected_tool": selected_tool,
                "tool_label": tool_labels.get(
                    selected_tool,
                    selected_tool,
                ),
            })

        except Exception as exc:
            st.error("The agent encountered an error.")
            st.exception(exc)

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "Sorry, the agent couldn't complete that request. "
                    "Check the error details above."
                ),
                "sources": [],
                "selected_tool": "error",
                "tool_label": "Error",
            })

    # Render the updated conversation in the container created above.
    with chat_history_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                tool_label = message.get("tool_label")

                if (
                    message["role"] == "assistant"
                    and tool_label
                ):
                    st.caption(f"Handled by: {tool_label}")

                message_sources = message.get("sources", [])

                if message_sources:
                    with st.expander("📄 Sources used"):
                        for source in message_sources:
                            st.write(
                                f"- {source['source']} "
                                f"| Page {source['page']}"
                            )


# =========================================================
# TAB 2: AI STUDY PLAN GENERATOR
# =========================================================

with planner_tab:
    st.subheader("🗓️ AI Study Plan Generator")

    st.write(
        "Create a personalized learning roadmap based on your "
        "current level, available study time, duration, and goal."
    )

    with st.form("study_plan_form"):
        selected_topic = st.selectbox(
            "Choose a topic",
            [
                "AI Fundamentals",
                "Machine Learning",
                "Deep Learning",
                "Generative AI",
                "RAG and Vector Databases",
                "Agentic AI and LangGraph",
                "Python for AI",
                "AWS and Amazon Bedrock",
                "Custom topic",
            ],
        )

        custom_topic = st.text_input(
            "Custom topic",
            placeholder="Example: Neural Networks",
        )

        skill_level = st.selectbox(
            "Current skill level",
            [
                "Beginner",
                "Basic knowledge",
                "Intermediate",
                "Advanced",
            ],
        )

        daily_hours = st.slider(
            "Available study time per day",
            min_value=1,
            max_value=8,
            value=2,
        )

        duration = st.selectbox(
            "Study duration",
            [
                "7 days",
                "14 days",
                "30 days",
                "60 days",
                "90 days",
            ],
            index=2,
        )

        learning_goal = st.selectbox(
            "Main learning goal",
            [
                "Understand fundamentals",
                "Prepare for job interviews",
                "Build practical projects",
                "Prepare for an examination",
                "Become job-ready",
            ],
        )

        generate_plan = st.form_submit_button(
            "Generate Study Plan",
            use_container_width=True,
        )

    if generate_plan:
        topic = (
            custom_topic.strip()
            if selected_topic == "Custom topic"
            else selected_topic
        )

        if not topic:
            st.warning("Please enter your custom topic.")
        else:
            plan_prompt = f"""
You are an experienced AI and Machine Learning instructor
and career mentor.

Create a realistic, actionable study plan.

Topic: {topic}
Current level: {skill_level}
Study time: {daily_hours} hours per day
Duration: {duration}
Goal: {learning_goal}

Requirements:
1. Follow a logical learning sequence.
2. Include prerequisites when needed.
3. Divide the plan into days or weeks.
4. Include concepts, practice exercises, and coding tasks.
5. Include a practical mini-project where relevant.
6. Include revision and self-assessment.
7. Include interview preparation if relevant to the goal.
8. Make the workload realistic for the available study hours.
9. Prefer free resources and local tools where possible.
10. State expected learning outcomes.
11. Use clear Markdown headings and lists.

Generate the complete study plan now.
"""

            try:
                with st.spinner(
                    "Generating your study plan with the local model..."
                ):
                    study_plan = invoke_local_llm(plan_prompt)

                st.session_state.study_plan = study_plan
                st.session_state.study_plan_topic = topic

            except Exception as exc:
                st.error("Could not generate the study plan.")
                st.exception(exc)

    if st.session_state.study_plan:
        st.divider()
        st.markdown("### Your Personalized Study Plan")
        st.markdown(st.session_state.study_plan)

        safe_filename = "".join(
            char
            if char.isalnum() or char in "-_"
            else "_"
            for char in st.session_state.study_plan_topic
        )

        st.download_button(
            "Download Study Plan (.md)",
            data=st.session_state.study_plan,
            file_name=f"{safe_filename}_study_plan.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =========================================================
# TAB 3: QUIZ GENERATOR
# =========================================================

with quiz_tab:
    st.subheader("🧠 AI Quiz Generator")

    st.write(
        "Generate five multiple-choice questions, answer them, "
        "calculate your score, and review the explanations."
    )

    with st.form("quiz_generation_form"):
        selected_quiz_topic = st.selectbox(
            "Choose a topic",
            [
                "Artificial Intelligence",
                "Machine Learning",
                "Deep Learning",
                "Neural Networks",
                "Generative AI",
                "RAG and Vector Databases",
                "AI Agents and LangGraph",
                "AWS and Amazon Bedrock",
                "Custom Topic",
            ],
        )

        custom_quiz_topic = st.text_input(
            "Custom quiz topic",
            placeholder="Example: Logistic Regression",
        )

        quiz_level = st.selectbox(
            "Difficulty",
            [
                "Beginner",
                "Intermediate",
                "Advanced",
            ],
        )

        quiz_from_documents = st.checkbox(
            "Use my PDF knowledge base",
            value=True,
            help=(
                "Use your indexed PDFs as the primary source "
                "when generating questions."
            ),
        )

        generate_quiz_button = st.form_submit_button(
            "Generate Quiz",
            use_container_width=True,
        )

    if generate_quiz_button:
        topic = (
            custom_quiz_topic.strip()
            if selected_quiz_topic == "Custom Topic"
            else selected_quiz_topic
        )

        if not topic:
            st.warning("Please enter your custom topic.")
        else:
            try:
                with st.spinner(
                    "Generating your quiz with the local model..."
                ):
                    new_quiz, new_sources = generate_quiz(
                        topic=topic,
                        difficulty=quiz_level,
                        use_documents=quiz_from_documents,
                        uploaded_vector_store=(
                            st.session_state.uploaded_vector_store
                        ),
                    )

                # Reset old answers before displaying a new quiz.
                for index in range(5):
                    st.session_state.pop(
                        f"quiz_answer_{index}",
                        None,
                    )

                st.session_state.quiz_data = new_quiz
                st.session_state.quiz_results = None
                st.session_state.quiz_topic = topic
                st.session_state.quiz_sources = new_sources

                st.success(f"Quiz generated for: {topic}")

            except Exception as exc:
                st.error(
                    "Could not generate a valid quiz. "
                    "Please try again."
                )
                st.exception(exc)

    # Render the quiz and collect answers.
    if st.session_state.quiz_data:
        st.divider()

        st.markdown(
            f"### Quiz: {st.session_state.quiz_topic}"
        )

        with st.form("quiz_answers_form"):
            for index, question_data in enumerate(
                st.session_state.quiz_data
            ):
                st.markdown(
                    f"**Question {index + 1}: "
                    f"{question_data['question']}**"
                )

                st.radio(
                    "Choose your answer",
                    options=["A", "B", "C", "D"],
                    format_func=(
                        lambda option, opts=question_data["options"]:
                        f"{option}. {opts[option]}"
                    ),
                    key=f"quiz_answer_{index}",
                    index=None,
                    label_visibility="collapsed",
                )

                st.write("")

            submit_answers = st.form_submit_button(
                "Submit Answers",
                use_container_width=True,
            )

        if submit_answers:
            selected_answers = [
                st.session_state.get(
                    f"quiz_answer_{index}"
                )
                for index in range(5)
            ]

            if any(
                answer is None
                for answer in selected_answers
            ):
                st.warning(
                    "Please answer all five questions before submitting."
                )
            else:
                score = 0
                results = []

                for index, question_data in enumerate(
                    st.session_state.quiz_data
                ):
                    selected = selected_answers[index]
                    correct = question_data["answer"]
                    is_correct = selected == correct

                    if is_correct:
                        score += 1

                    results.append({
                        "question": question_data["question"],
                        "selected": selected,
                        "correct": correct,
                        "is_correct": is_correct,
                        "options": question_data["options"],
                        "explanation": question_data["explanation"],
                    })

                st.session_state.quiz_results = {
                    "score": score,
                    "details": results,
                }

        # Display results once submitted.
        if st.session_state.quiz_results is not None:
            results_data = st.session_state.quiz_results
            score = results_data["score"]

            st.divider()
            st.markdown("### 📊 Your Quiz Results")

            st.metric("Score", f"{score} / 5")
            st.progress(score / 5)

            st.write(f"**Percentage:** {score * 20}%")

            for index, result in enumerate(
                results_data["details"],
                start=1,
            ):
                st.markdown(
                    f"**Question {index}: {result['question']}**"
                )

                st.write(
                    f"Your answer: {result['selected']}. "
                    f"{result['options'][result['selected']]}"
                )

                st.write(
                    f"Correct answer: {result['correct']}. "
                    f"{result['options'][result['correct']]}"
                )

                if result["is_correct"]:
                    st.success("Correct!")
                else:
                    st.error("Incorrect.")

                st.info(
                    f"Explanation: {result['explanation']}"
                )

            if st.session_state.quiz_sources:
                with st.expander("📄 Documents used for this quiz"):
                    for source in st.session_state.quiz_sources:
                        st.write(
                            f"- {source['source']} "
                            f"| Page {source['page']}"
                        )
