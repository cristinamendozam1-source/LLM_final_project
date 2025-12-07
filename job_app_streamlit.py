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
    Dalberg Advisors – Social impact consulting firm
    Postgraduate Intern, Mumbai, India
    June 2025 – Aug. 2025
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

    # Simple month list to help detect date-only lines
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul",
              "aug", "sep", "oct", "nov", "dec"]

    for line in lines:
        upper = line.upper()

        # -------- Section detection --------
        if upper.startswith("EXPERIENCE") or "PROFESSIONAL EXPERIENCE" in upper:
            current_section = "experience"
            continue
        if upper.startswith("EDUCATION"):
            current_section = "education"
            continue
        if "SPECIFIC SKILLS" in upper or upper.startswith("SKILLS"):
            current_section = "skills"
            continue

        # -------- EXPERIENCE SECTION --------
        if current_section == "experience":
            # 1) New employer line heuristic:
            #    - not a bullet
            #    - either contains an en dash / hyphen OR has no digits and looks like a heading
            if not line.startswith("•"):
                is_heading_with_dash = ("–" in line or " - " in line)
                looks_like_org = (
                    not any(ch.isdigit() for ch in line) and
                    len(line.split()) <= 10 and
                    line[0].isupper()
                )
                if is_heading_with_dash or looks_like_org:
                    # Finish previous position
                    if current_pos is not None:
                        structured["positions"].append(current_pos)
                    # Start new employer
                    current_pos = {
                        "employer": line,
                        "title": "",
                        "dates": "",
                        "location": "",
                        "responsibilities": []
                    }
                    continue

            # 2) Date-only line (e.g., "June 2025 – Aug. 2025")
            if current_pos is not None and not line.startswith("•"):
                lower_line = line.lower()
                has_year = re.search(r"\d{4}", lower_line) is not None
                has_month = any(m in lower_line for m in months)
                if current_pos["dates"] == "" and has_year and has_month:
                    current_pos["dates"] = line
                    continue

            # 3) Title line (first non-bullet, non-date line after employer)
            if current_pos is not None and not line.startswith("•") and current_pos["title"] == "":
                current_pos["title"] = line
                # sometimes dates are on same line as title (we try to capture them too)
                m = re.search(r"(\d{4}.*)$", line)
                if m and current_pos["dates"] == "":
                    current_pos["dates"] = m.group(1).strip()
                continue

            # 4) Bullet line: real responsibilities
            if current_pos is not None and line.startswith("•"):
                bullet = line.lstrip("•").strip()

                # Ignore placeholder bullets like "Responsibilities to be detailed."
                if not bullet:
                    continue
                if "responsibilities to be detailed" in bullet.lower():
                    continue

                current_pos["responsibilities"].append(bullet)
                continue

            # 5) Continuation lines for last bullet (but avoid swallowing dates)
            if current_pos is not None and current_pos["responsibilities"]:
                lower_line = line.lower()
                has_year = re.search(r"\d{4}", lower_line) is not None
                has_month = any(m in lower_line for m in months)
                # If it looks like a date line, don't append it to the bullet
                if not (has_year and has_month):
                    current_pos["responsibilities"][-1] += " " + line
                    continue

        # -------- EDUCATION SECTION --------
        elif current_section == "education":
            structured["education"].append(line)

        # -------- SKILLS SECTION --------
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
            "- Intro paragraph (interest + short background)\n"
            "- 1–2 body paragraphs highlighting specific achievements with correct employers\n"
            "- Connection to the job and organization\n"
            "- Closing with enthusiasm and invitation to talk\n"
        ),
        expected_output="A 250–400 word factually correct cover letter in Markdown.",
        output_file=cl_output_path,
        agent=cover_letter_writer,
        context=[fit_assessment_task],
        async_execution=True
    )

    qa_task = Task(
        description=(
            "Use the STRUCTURED CV JSON as ground truth:\n"
            f"{cv_json}\n\n"
            "You will review:\n"
            "1) The revised CV (Markdown):\n"
            "{revised_cv}\n\n"
            "2) The cover letter (Markdown):\n"
            "{cover_letter}\n\n"
            "Check:\n"
            "- Are all positions present compared to JSON.positions?\n"
            "- Are all responsibilities present somewhere under the same employer?\n"
            "- Are there any achievements attributed to the wrong employer?\n"
            "- Did the cover letter correctly attribute each specific achievement?\n\n"
            "Return a QA report with:\n"
            "Approval Status: Approved / Approved with Revisions / Major Errors Found\n\n"
            "Then list specific issues if any."
        ),
        expected_output="QA report describing any misattribution or missing content.",
        agent=quality_assurance_agent,
        context=[cv_revision_task, cover_letter_task],
        async_execution=False
    )

    crew = Crew(
        agents=[
            job_description_analyzer,
            recruitment_expert,
            cv_strategist,
            cover_letter_writer,
            quality_assurance_agent
        ],
        tasks=[
            job_analysis_task,
            fit_assessment_task,
            cv_revision_task,
            cover_letter_task,
            qa_task
        ],
        verbose=True
    )

    handles = {
        "fit": fit_assessment_task,
        "cv": cv_revision_task,
        "cl": cover_letter_task,
        "qa": qa_task,
        "jd": job_analysis_task
    }

    return crew, handles, {
        "cv_output_path": cv_output_path,
        "cl_output_path": cl_output_path
    }

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
def main():
    st.markdown('<div class="main-header">💼 AI-Powered Job Application Assistant</div>',
                unsafe_allow_html=True)
    st.markdown("""
    This tool uses a multi-agent AI system to:
    - Analyze the job description
    - Evaluate your fit
    - Generate a tailored CV (without mixing experiences between employers)
    - Draft a factually accurate cover letter
    - Run a QA check against your original CV
    """)

    if not setup_openai_api():
        st.stop()
    st.success("✅ API configured successfully")

    # Inputs
    st.markdown('<div class="section-header">📄 Upload Your Documents</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Resume/CV")
        resume_file = st.file_uploader(
            "Upload your CV (PDF, Word, or text)",
            type=["pdf", "docx", "doc", "txt", "md"],
            help="Upload your current resume"
        )

    with col2:
        st.subheader("Job Description")
        job_input_method = st.radio(
            "How would you like to provide the job description?",
            ["Paste Text", "Upload File"]
        )
        job_description_text = ""
        job_file = None

        if job_input_method == "Paste Text":
            job_description_text = st.text_area(
                "Paste job description here",
                height=300,
                help="Copy and paste the full job posting"
            )
        else:
            job_file = st.file_uploader(
                "Upload job description file",
                type=["txt", "md", "pdf", "docx", "doc"],
                help="Upload job description"
            )

    # Process
    if st.button("🚀 Analyze and Generate Application Materials", type="primary"):
        if not resume_file:
            st.error("❌ Please upload your resume")
            return

        if job_input_method == "Paste Text" and not job_description_text.strip():
            st.error("❌ Please provide a job description")
            return
        if job_input_method == "Upload File" and job_file is None:
            st.error("❌ Please upload a job description file")
            return

        with st.spinner("🔄 Processing your application... This may take a few minutes."):
            try:
                # --- CV text extraction ---
                status_text = st.empty()
                status_text.text("Extracting text from your CV...")

                resume_path = save_uploaded_file(resume_file)
                cv_ext = resume_file.name.split(".")[-1].lower()
                cv_text = extract_text_from_file(resume_path, cv_ext)

                if not cv_text or len(cv_text.strip()) < 50:
                    st.error("❌ Failed to extract meaningful text from the CV. "
                             "Please check the file or try another format.")
                    return

                with st.expander("📄 Preview extracted CV text (first 500 characters)"):
                    st.text(cv_text[:500] + "..." if len(cv_text) > 500 else cv_text)

                # --- Deterministic parsing into structured JSON ---
                status_text.text("Parsing CV into structured JSON (no LLM)...")
                cv_structured = parse_cv_text_to_structured_json(cv_text)

                with st.expander("🔎 Structured CV JSON (debug)"):
                    st.json(cv_structured)

                # --- Job description text extraction ---
                if job_input_method == "Paste Text":
                    jd_text = job_description_text.strip()
                else:
                    jd_path = save_uploaded_file(job_file)
                    jd_ext = job_file.name.split(".")[-1].lower()
                    jd_text = extract_text_from_file(jd_path, jd_ext)
                    jd_text = jd_text.strip()

                if not jd_text or len(jd_text) < 50:
                    st.error("❌ Job description seems too short or empty. "
                             "Please provide a full job posting.")
                    return

                with st.expander("📋 Preview job description (first 300 characters)"):
                    st.text(jd_text[:300] + "..." if len(jd_text) > 300 else jd_text)

                # --- Create crew & run ---
                from datetime import datetime

                progress_bar = st.progress(0)
                progress_bar.progress(20)
                status_text.text("Step 1/3: Running analysis and fit assessment...")

                crew, handles, paths = create_agents_and_tasks(cv_structured, jd_text)

                # kickoff without extra inputs (all in task descriptions)
                result = crew.kickoff()

                progress_bar.progress(70)
                status_text.text("Step 2/3: Generating CV and cover letter...")

                # Read outputs
                revised_cv = ""
                cover_letter = ""
                qa_report = ""
                assessment = ""

                # Fit assessment
                if handles["fit"].output is not None:
                    assessment = str(handles["fit"].output)
                else:
                    assessment = str(result)

                # CV
                if os.path.exists(paths["cv_output_path"]):
                    with open(paths["cv_output_path"], "r", encoding="utf-8") as f:
                        revised_cv = f.read()
                elif handles["cv"].output is not None:
                    revised_cv = str(handles["cv"].output)

                # Cover letter
                if os.path.exists(paths["cl_output_path"]):
                    with open(paths["cl_output_path"], "r", encoding="utf-8") as f:
                        cover_letter = f.read()
                elif handles["cl"].output is not None:
                    cover_letter = str(handles["cl"].output)

                # QA report
                if handles["qa"].output is not None:
                    qa_report = str(handles["qa"].output)
                else:
                    qa_report = str(result)

                progress_bar.progress(100)
                status_text.text("Step 3/3: Finalizing outputs...")

                st.session_state.results = {
                    "assessment": assessment,
                    "revised_cv": revised_cv,
                    "cover_letter": cover_letter,
                    "qa_report": qa_report,
                    "cv_structured": cv_structured
                }
                st.session_state.generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.processing_complete = True

                status_text.text("✅ Processing complete!")
                progress_bar.empty()

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                import traceback
                with st.expander("🔍 Error details"):
                    st.code(traceback.format_exc())
                return

    # -----------------------------------------------------
    # Display Results
    # -----------------------------------------------------
    if st.session_state.processing_complete and st.session_state.results:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)

        assessment = st.session_state.results["assessment"]
        revised_cv = st.session_state.results["revised_cv"]
        cover_letter = st.session_state.results["cover_letter"]
        qa_report = st.session_state.results["qa_report"]

        fit_score, fit_category = extract_fit_score(assessment)
        fit_class = (
            "high-fit" if fit_category == "HIGH"
            else "medium-fit" if fit_category == "MEDIUM"
            else "low-fit"
        )

        st.markdown(f"""
            <div class="fit-score {fit_class}">
                Fit Score: {fit_score}%<br>
                <span style="font-size: 1.5rem;">{fit_category} FIT</span>
            </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Fit Assessment",
            "📄 Revised CV",
            "✉️ Cover Letter",
            "✅ QA Report",
            "💾 Download All"
        ])

        with tab1:
            st.markdown("### Detailed Fit Assessment")
            st.markdown(assessment)

        with tab2:
            st.markdown("### Your Tailored CV")
            st.info(
                "⚠️ **Important:** Review carefully to ensure all information is accurate. "
                "This version should preserve employer boundaries and bullets, but "
                "you are the final judge of correctness."
            )
            st.markdown(revised_cv if revised_cv else "_No CV output found._")

            if revised_cv:
                st.download_button(
                    label="⬇️ Download CV (Markdown)",
                    data=revised_cv,
                    file_name="revised_cv.md",
                    mime="text/markdown"
                )
                cv_docx_path = markdown_to_docx(revised_cv, "revised_cv.docx")
                with open(cv_docx_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download CV (Word)",
                        data=f,
                        file_name="revised_cv.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

        with tab3:
            st.markdown("### Your Cover Letter")
            st.info(
                "⚠️ **Important:** Verify that all specific achievements and numbers "
                "are attributed to the correct employer. Personalize placeholders like "
                "[Your Name] and [Hiring Manager's Name]."
            )
            st.markdown(cover_letter if cover_letter else "_No cover letter output found._")

            if cover_letter:
                st.download_button(
                    label="⬇️ Download Cover Letter (Markdown)",
                    data=cover_letter,
                    file_name="cover_letter.md",
                    mime="text/markdown"
                )
                cl_docx_path = markdown_to_docx(cover_letter, "cover_letter.docx")
                with open(cl_docx_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Cover Letter (Word)",
                        data=f,
                        file_name="cover_letter.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

        with tab4:
            st.markdown("### Quality Assurance Report")
            st.markdown(qa_report if qa_report else "_No QA report available._")

        with tab5:
            st.markdown("### Download All Documents")
            combined = f"""# Job Application Package - Generated by AI Assistant

## Fit Assessment
{assessment}

---

## Quality Assurance Report
{qa_report}

---

## Revised CV
{revised_cv}

---

## Cover Letter
{cover_letter}

---

*Generated on {st.session_state.get('generation_date', 'N/A')}*
"""
            st.download_button(
                label="⬇️ Download Complete Package (Markdown)",
                data=combined,
                file_name="job_application_package.md",
                mime="text/markdown"
            )

    # Sidebar
    with st.sidebar:
        st.markdown("### 📖 How It Works")
        st.markdown("""
        **Step 1: CV Parsing (Deterministic)**
        - PDF/Word parsed into a structured JSON (no LLM).
        - Employer–bullet relationships are preserved.

        **Step 2: Multi-Agent Pipeline (crewAI)**
        - Job description analysis
        - Fit assessment
        - CV rewriting (strict employer boundaries)
        - Cover letter drafting

        **Step 3: QA**
        - QA agent checks outputs against the structured CV JSON.
        """)

        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.info("Your documents are processed temporarily and not stored permanently.")

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use a reasonably structured CV (clear EXPERIENCE/EDUCATION headings).
        - Provide the full job description.
        - Always review the generated CV and cover letter before sending.
        """)


if __name__ == "__main__":
    main()
