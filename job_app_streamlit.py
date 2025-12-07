# app.py
# ---------------------------------------------------------
# AI Job Application Assistant (Final Project Version)
# ---------------------------------------------------------
# Requirements (in requirements.txt):
#   streamlit
#   crewai[tools]
#   PyPDF2
#   python-docx
#   langchain-community (if you use elsewhere)
# ---------------------------------------------------------

import os
import json
import re
import tempfile
from typing import Dict, Tuple

import streamlit as st
import PyPDF2
from docx import Document
from crewai import Agent, Task, Crew

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Streamlit Page Configuration & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Job Application Assistant",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .fit-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .high-fit {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-fit {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-fit {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------
# OpenAI API Setup
# ---------------------------------------------------------
def setup_openai_api() -> bool:
    """Configure OpenAI API from Streamlit secrets or user input."""
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_MODEL_NAME"] = st.secrets.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        return True
    else:
        with st.sidebar:
            st.subheader("🔑 API Configuration")
            api_key = st.text_input("OpenAI API Key", type="password")
            model_name = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                index=0
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                os.environ["OPENAI_MODEL_NAME"] = model_name
                return True
            else:
                st.warning("⚠️ Please enter your OpenAI API key to continue")
                return False

# ---------------------------------------------------------
# File Helpers
# ---------------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return path."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"uploaded_{uploaded_file.name}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a Word document."""
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return ""

def extract_text_from_file(path: str, ext: str) -> str:
    """Extract text based on file extension."""
    ext = ext.lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext in ["docx", "doc"]:
        return extract_text_from_docx(path)
    elif ext in ["txt", "md"]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return ""
    else:
        st.error(f"Unsupported file type: {ext}")
        return ""

def markdown_to_docx(md_text: str, filename: str) -> str:
    """
    Convert simple Markdown/plain text to a .docx file.
    Each line becomes a paragraph; headers are just plain bold text stripped of '#'.
    """
    doc = Document()
    for raw_line in md_text.split("\n"):
        line = raw_line.lstrip("#").strip()
        doc.add_paragraph(line)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    doc.save(temp_path)
    return temp_path

# ---------------------------------------------------------
# Deterministic CV Parser (No LLM)
# ---------------------------------------------------------
def parse_cv_text_to_structured_json(cv_text: str) -> Dict:
    """
    Deterministic parser: group bullets under the employer header that precedes them.
    Works best for CVs like:

    EXPERIENCE
    Instiglio – Consulting firm in international development
    Senior Associate, Nairobi, Kenya March 2024 – June 2024
    • Bullet 1
    • Bullet 2
    ...
    """
    lines = [l.strip() for l in cv_text.split("\n") if l.strip()]
    structured = {
        "positions": [],
        "education": [],
        "skills": {"technical": [], "soft": []}
    }

    current_section = None
    current_pos = None

    for line in lines:
        upper = line.upper()

        # Section detection
        if upper.startswith("EXPERIENCE"):
            current_section = "experience"
            continue
        if upper.startswith("EDUCATION"):
            current_section = "education"
            continue
        if "SPECIFIC SKILLS" in upper or upper.startswith("SKILLS"):
            current_section = "skills"
            continue

        if current_section == "experience":
            # New employer line heuristic: has a dash and not a bullet
            if not line.startswith("•") and ("–" in line or " - " in line):
                # Finish previous position
                if current_pos is not None:
                    structured["positions"].append(current_pos)
                # Start new
                current_pos = {
                    "employer": line,
                    "title": "",
                    "dates": "",
                    "location": "",
                    "responsibilities": []
                }
                continue

            # Title/dates line: first non-bullet after employer
            if current_pos is not None and not line.startswith("•") and current_pos["title"] == "":
                current_pos["title"] = line
                m = re.search(r"(\d{4}.*)$", line)
                if m:
                    current_pos["dates"] = m.group(1).strip()
                continue

            # Bullet line
            if current_pos is not None and line.startswith("•"):
                bullet = line.lstrip("•").strip()
                if bullet:
                    current_pos["responsibilities"].append(bullet)
                continue

            # Continuation line for last bullet
            if current_pos is not None and current_pos["responsibilities"]:
                current_pos["responsibilities"][-1] += " " + line
                continue

        elif current_section == "education":
            structured["education"].append(line)

        elif current_section == "skills":
            raw_pieces = re.split(r"[|,]", line)
            for p in raw_pieces:
                s = p.strip()
                if not s:
                    continue
                if any(x in s.lower() for x in ["stata", "excel", "powerpoint", "python", "software", "data"]):
                    structured["skills"]["technical"].append(s)
                else:
                    structured["skills"]["soft"].append(s)

    # Add last position if exists
    if current_pos is not None:
        structured["positions"].append(current_pos)

    return structured

# ---------------------------------------------------------
# Fit Score Helper
# ---------------------------------------------------------
def extract_fit_score(assessment_text: str) -> Tuple[int, str]:
    """Extract fit score and category from assessment text."""
    m = re.search(r"(\d+)%", assessment_text)
    if m:
        score = int(m.group(1))
    else:
        score = 50

    if score >= 75:
        category = "HIGH"
    elif score >= 50:
        category = "MEDIUM"
        pass
    else:
        category = "LOW"
    return score, category

# ---------------------------------------------------------
# crewAI Agents & Tasks
# ---------------------------------------------------------
def create_agents_and_tasks(cv_structured: Dict, job_description_text: str):
    """Create crewAI agents and tasks using structured CV JSON (no LLM parsing)."""

    cv_json = json.dumps(cv_structured, indent=2)

    # Agents
    job_description_analyzer = Agent(
        role="Job Description Analyzer",
        goal=(
            "Extract and structure key information from job descriptions: "
            "responsibilities and required skills/qualifications."
        ),
        tools=[],
        verbose=True,
        backstory=(
            "You are an expert in reading job postings and summarizing what matters. "
            "You separate responsibilities from requirements clearly."
        )
    )

    recruitment_expert = Agent(
        role="Recruitment Assessment Expert",
        goal=(
            "Assess how well the structured CV JSON matches the job description, "
            "and produce a fit score and narrative."
        ),
        tools=[],
        verbose=True,
        backstory=(
            "You are an experienced recruiter. You work from structured data (JSON) "
            "so you can see exactly which bullets belong to which employer."
        )
    )

    cv_strategist = Agent(
        role="Adaptive CV Strategist",
        goal=(
            "Rewrite the CV ONLY by reordering and lightly rephrasing bullets within each job, "
            "preserving 100% factual accuracy and original employer→bullet mapping."
        ),
        tools=[],
        verbose=True,
        backstory=(
            "You are obsessive about not mixing employers. You treat the JSON as ground truth: "
            "each position’s bullets must stay under that position."
        )
    )

    cover_letter_writer = Agent(
        role="Adaptive Cover Letter Writer",
        goal=(
            "Write a 250–400 word cover letter using the structured CV JSON. "
            "Every specific achievement must be attributed to the correct employer as in the JSON."
        ),
        tools=[],
        verbose=True,
        backstory=(
            "You write persuasive, factually precise cover letters. You never say "
            "'At Company X I did Y' unless the JSON shows Y under Company X."
        )
    )

    quality_assurance_agent = Agent(
        role="Quality Assurance Specialist",
        goal=(
            "Compare the revised CV and cover letter against the structured CV JSON. "
            "Flag any misattribution or missing positions/bullets."
        ),
        tools=[],
        verbose=True,
        backstory=(
            "You are the final gatekeeper. You cross-check everything against the JSON "
            "and describe any discrepancies."
        )
    )

    # Output file paths
    temp_dir = tempfile.gettempdir()
    cv_output_path = os.path.join(temp_dir, "revised_cv.md")
    cl_output_path = os.path.join(temp_dir, "cover_letter.md")

    # Tasks
    job_analysis_task = Task(
        description=(
            "Job description:\n\n"
            f"{job_description_text}\n\n"
            "Extract and present:\n"
            "1. Core responsibilities (bullet list)\n"
            "2. Required technical skills\n"
            "3. Required soft skills\n"
            "4. Qualifications (education, experience level)\n"
        ),
        expected_output="Markdown with clearly labeled sections for responsibilities and requirements.",
        agent=job_description_analyzer,
        async_execution=False
    )

    fit_assessment_task = Task(
        description=(
            "You are given:\n\n"
            "JOB DESCRIPTION:\n"
            f"{job_description_text}\n\n"
            "STRUCTURED CV JSON:\n"
            f"{cv_json}\n\n"
            "Use ONLY this JSON for facts about the candidate. "
            "Do not invent or move achievements between employers.\n\n"
            "Calculate a fit score (0–100%), list strengths and gaps, and write a narrative.\n"
            "Format:\n"
            "## Fit Score: XX%\n"
            "**Category:** HIGH/MEDIUM/LOW FIT\n\n"
            "### Key Strengths:\n- ...\n\n"
            "### Gaps and Areas for Growth:\n- ...\n\n"
            "### Overall Assessment:\n...\n\n"
            "### Recommendation:\n..."
        ),
        expected_output="Fit score + category + strengths + gaps + narrative + recommendation.",
        agent=recruitment_expert,
        context=[job_analysis_task],
        async_execution=False
    )

    cv_revision_task = Task(
        description=(
            "You are given STRUCTURED CV JSON:\n"
            f"{cv_json}\n\n"
            "This JSON has an array 'positions', where each element has:\n"
            "  employer, title, dates, location, responsibilities[]\n\n"
            "CRITICAL RULES:\n"
            "- You MUST keep each responsibility under the same employer as in the JSON.\n"
            "- You may reorder responsibilities WITHIN a position, but never move them to another.\n"
            "- You may lightly rephrase bullets, but must not change their meaning.\n"
            "- Do NOT drop positions or responsibilities.\n\n"
            "Create a full CV in Markdown with sections:\n"
            "1) [Your Name] + contact (placeholders OK)\n"
            "2) Professional Summary (2–3 sentences)\n"
            "3) Professional Experience (use positions array)\n"
            "4) Education (summarized from JSON.education)\n"
            "5) Skills (from JSON.skills)\n"
        ),
        expected_output="Complete Markdown CV with correct employer→achievement mapping preserved.",
        output_file=cv_output_path,
        agent=cv_strategist,
        context=[fit_assessment_task],
        async_execution=True
    )

    cover_letter_task = Task(
        description=(
            "You are given:\n\n"
            "JOB DESCRIPTION:\n"
            f"{job_description_text}\n\n"
            "STRUCTURED CV JSON:\n"
            f"{cv_json}\n\n"
            "FIT ASSESSMENT:\n"
            "{fit_assessment}\n\n"
            "First, read the JSON and build an internal map of achievements -> employer.\n"
            "When you mention a specific achievement, you MUST check which employer it "
            "belongs to in the JSON and attribute it correctly.\n\n"
            "Write a 250–400 word cover letter with:\n"
            "- Intro paragraph
