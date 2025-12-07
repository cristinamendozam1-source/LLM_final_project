# app.py
# ---------------------------------------------------------
# AI Job Application Assistant (Final Project - Cristina Mendoza Mora)
# ---------------------------------------------------------
# Requirements (install in your environment):
#   pip install streamlit crewai PyPDF2 python-docx
#
# This app:
#  - Parses a PDF CV into structured JSON (no LLM here)
#  - Uses crewAI agents to:
#      1) Analyze the job description
#      2) Assess fit
#      3) Generate a tailored CV (without mixing employers)
#      4) Generate a cover letter
#      5) Run QA for factual accuracy and completeness
#  - Exports CV and cover letter as both Markdown and DOCX
# ---------------------------------------------------------

import os
import json
import tempfile
from typing import Dict, Tuple

import streamlit as st
import PyPDF2
from docx import Document

from crewai import Agent, Task, Crew

import warnings
warnings.filterwarnings('ignore')

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
# API Setup
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
# Save uploaded files & text
# ---------------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return path."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def save_text_as_markdown(text: str, filename: str) -> str:
    """Save text content as markdown file and return path."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(text)
    return temp_path


def read_job_description(job_input_method: str, job_description: str, job_file) -> Tuple[str, str]:
    """
    Return (job_description_text, path_to_markdown_copy).
    If file is PDF or text/md, read and normalize to plain text and md file.
    """
    if job_input_method == "Paste Text":
        jd_text = job_description.strip()
        jd_md_path = save_text_as_markdown(jd_text, "job_description.md")
        return jd_text, jd_md_path

    # Upload File method
    if job_file is None:
        return "", ""

    temp_path = save_uploaded_file(job_file)
    ext = os.path.splitext(job_file.name)[1].lower()

    if ext == ".pdf":
        reader = PyPDF2.PdfReader(temp_path)
        jd_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:  # .txt, .md, etc.
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read()

    jd_text = jd_text.strip()
    jd_md_path = save_text_as_markdown(jd_text, "job_description.md")
    return jd_text, jd_md_path

# ---------------------------------------------------------
#Parse CV PDF deterministically into structured JSON
# ---------------------------------------------------------
def parse_cv_pdf_to_structured_json(pdf_path: str) -> Dict:
    """
    Deterministic CV parser.
    - No LLMs involved.
    - Goal: preserve employer–bullet relationships, titles, and dates.
    - Designed to work reasonably well for structured CVs (like yours) and
      general enough for other users.

    Output structure:
    {
      "experience": [
        {
          "organization": str,
          "title": str,
          "location": str,
          "dates": str,
          "bullets": [str, ...]
        }, ...
      ],
      "education": str,
      "skills": [str, ...]
    }
    """
    import re

    reader = PyPDF2.PdfReader(pdf_path)
    full_text = "\n".join((page.extract_text() or "") for page in reader.pages)

    # Normalize whitespace
    text = re.sub(r"\r", "\n", full_text)
    text = re.sub(r"\n+", "\n", text)

    # Split into lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    structured = {
        "experience": [],
        "education": "",
        "skills": []
    }

    # Identify indices of main headings
    def find_index(keyword):
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                return i
        return None

    exp_idx = find_index("EXPERIENCE")
    edu_idx = find_index("EDUCATION")
    skills_idx = find_index("SPECIFIC SKILLS")

    # Extract EDUCATION block as raw text
    if edu_idx is not None:
        end = exp_idx if exp_idx is not None else len(lines)
        edu_lines = lines[edu_idx + 1:end]
        structured["education"] = "\n".join(edu_lines).strip()

    # Extract SKILLS block as simple list
    if skills_idx is not None:
        end = len(lines)
        skills_lines = lines[skills_idx + 1:end]
        # Often of form: "Languages: ...", "Software: ..."
        skills_text = " ".join(skills_lines)
        # Split by | or commas as a simple heuristic
        possible_skills = re.split(r"[|,]", skills_text)
        structured["skills"] = [s.strip() for s in possible_skills if s.strip()]

    # Extract EXPERIENCE blocks
    if exp_idx is not None:
        end = skills_idx if skills_idx is not None else len(lines)
        exp_lines = lines[exp_idx + 1:end]

        current_job = None

        for line in exp_lines:
            # Bullet
            if line.startswith("•"):
                if current_job is not None:
                    bullet_text = line.lstrip("•").strip()
                    if bullet_text:
                        current_job["bullets"].append(bullet_text)
                continue

            # Candidate organization line: often has "–" or "-" separating name and descriptor
            if " – " in line or "– " in line or " - " in line:
                # When we encounter a new org, store previous job first
                if current_job is not None:
                    structured["experience"].append(current_job)
                current_job = {
                    "organization": line,
                    "title": "",
                    "location": "",
                    "dates": "",
                    "bullets": []
                }
                continue

            # If we have a current job and not a bullet: likely title/location/dates
            if current_job is not None and not current_job["title"]:
                # Heuristic: titles often contain a comma and dates often have digits
                # We'll just store full line in title and try to extract dates as trailing part.
                current_job["title"] = line
                # Extract dates as last chunk with a digit
                date_match = re.search(r"(\d{4}.*)$", line)
                if date_match:
                    current_job["dates"] = date_match.group(1).strip()
                continue

            # Otherwise, if it's another line while current_job exists and no bullet, treat as continuation
            if current_job is not None:
                # Append to the last bullet if exists, else to title
                if current_job["bullets"]:
                    current_job["bullets"][-1] += " " + line
                else:
                    current_job["title"] += " " + line

        # Add last job if present
        if current_job is not None:
            structured["experience"].append(current_job)

    return structured

# ---------------------------------------------------------
# DOCX export
# ---------------------------------------------------------
def markdown_to_docx(md_text: str, filename: str) -> str:
    """
    Convert simple Markdown/plain text to a .docx file.
    We keep it simple: each line becomes a paragraph.
    """
    doc = Document()
    for raw_line in md_text.split("\n"):
        line = raw_line.lstrip("#").strip()  # remove markdown headers for cleaner Word doc
        doc.add_paragraph(line)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    doc.save(temp_path)
    return temp_path

# ---------------------------------------------------------
# Helper: Extract fit score from assessment text
# ---------------------------------------------------------
def extract_fit_score(assessment_text: str) -> Tuple[int, str]:
    import re
    match = re.search(r"(\d+)%", assessment_text)
    if match:
        score = int(match.group(1))
    else:
        score = 50  # default

    if score >= 75:
        category = "HIGH"
    elif score >= 50:
        category = "MEDIUM"
    else:
        category = "LOW"

    return score, category

# ---------------------------------------------------------
# Agents & Tasks (crewAI)
# ---------------------------------------------------------
def create_agents_and_tasks(job_description_text: str, cv_structured: Dict):
    """
    Create crewAI agents and tasks for the end-to-end pipeline.
    Uses:
      - job_description_text (plain string)
      - cv_structured (Python dict, will pass as JSON string)
    """

    cv_structured_json = json.dumps(cv_structured, indent=2)

    # 1) Job Description Analyzer
    job_description_analyzer = Agent(
        role="Job Description Analyzer",
        goal=(
            "Extract and structure key information from job descriptions: "
            "responsibilities, required skills, qualifications and culture signals."
        ),
        backstory=(
            "You are an expert at reading job descriptions and summarizing "
            "what really matters for a candidate."
        ),
        tools=[],
        verbose=True
    )

    # 2) Recruitment Expert
    recruitment_expert = Agent(
        role="Recruitment Assessment Expert",
        goal=(
            "Assess how well a candidate's structured CV matches the job description. "
            "Provide a transparent fit score and narrative."
        ),
        backstory=(
            "You are an experienced recruiter who has reviewed thousands of CVs. "
            "You are honest and concrete about strengths and weaknesses."
        ),
        tools=[],
        verbose=True
    )

    # 3) CV Strategist (hard constraints)
    cv_strategist = Agent(
        role="CV Strategist",
        goal=(
            "Rewrite the CV ONLY by reordering and lightly rephrasing bullets within "
            "each job, preserving 100% factual accuracy and employer-bullet mapping."
        ),
        backstory=(
            "You are obsessive about factual accuracy. You would rather omit content "
            "than misattribute an achievement to the wrong employer."
        ),
        tools=[],
        verbose=True
    )

    # 4) Cover Letter Writer
    cover_letter_writer = Agent(
        role="Cover Letter Writer",
        goal=(
            "Write a 250-400 word cover letter that is persuasive and strictly accurate, "
            "using only achievements whose employer is clear in the structured CV."
        ),
        backstory=(
            "You know that credibility is everything in a cover letter. You never claim "
            "an achievement happened at the wrong organization."
        ),
        tools=[],
        verbose=True
    )

    # 5) Quality Assurance Agent
    quality_assurance_agent = Agent(
        role="Quality Assurance Specialist",
        goal=(
            "Verify that the revised CV and cover letter are accurate, complete, "
            "and consistent with the original structured CV."
        ),
        backstory=(
            "You are the final gatekeeper. You cross-check everything against the structured CV "
            "and flag any misattribution or dropped content."
        ),
        tools=[],
        verbose=True
    )

    # Paths for outputs
    output_dir = tempfile.gettempdir()
    cv_output_path = os.path.join(output_dir, "revised_cv.md")
    cl_output_path = os.path.join(output_dir, "cover_letter.md")
    qa_output_path = os.path.join(output_dir, "qa_report.md")

    # --- Tasks ---

    # Task 1: Job description analysis
    job_analysis_task = Task(
        description=(
            "You are given a job description:\n\n"
            f"{job_description_text}\n\n"
            "Extract and present:\n"
            "1. Core responsibilities (bullet list)\n"
            "2. Required technical skills\n"
            "3. Required soft skills\n"
            "4. Qualifications (education, years of experience)\n"
            "5. Any hints about culture, mission, or values.\n\n"
            "Return a well-structured Markdown summary."
        ),
        expected_output="Markdown with clear sections for responsibilities, skills, qualifications, and culture.",
        agent=job_description_analyzer,
        async_execution=False
    )

    # Task 2: CV “insights” (not parsing; parsing done in Python)
    cv_insights_task = Task(
        description=(
            "You are given a structured JSON representation of a CV:\n\n"
            f"{cv_structured_json}\n\n"
            "Summarize the candidate's profile WITHOUT changing any facts:\n"
            "- Main thematic areas of experience\n"
            "- Types of organizations\n"
            "- Key technical skills\n"
            "- Key soft skills\n"
            "- Geographies worked in\n"
            "- Notable quantified achievements (with employer names).\n\n"
            "Do NOT infer or invent new achievements. Only summarize what is in the JSON."
        ),
        expected_output="Objective summary of the CV's content and main strengths.",
        agent=recruitment_expert,  # reuse knowledge here
        async_execution=False
    )

    # Task 3: Fit assessment
    fit_assessment_task = Task(
        description=(
            "You are given:\n\n"
            "JOB DESCRIPTION:\n"
            f"{job_description_text}\n\n"
            "STRUCTURED CV JSON:\n"
            f"{cv_structured_json}\n\n"
            "Based on these, produce a fit assessment:\n"
            "1. Fit score (0-100%).\n"
            "2. Fit category: HIGH (75+), MEDIUM (50-74), LOW (<50).\n"
            "3. 3-5 key strengths with specific evidence.\n"
            "4. 2-4 main gaps with specific missing requirements.\n"
            "5. 2-3 paragraph narrative explaining the score.\n"
            "6. Application recommendation.\n\n"
            "Format your answer as:\n"
            '## Fit Score: XX%\n'
            '**Category:** [HIGH/MEDIUM/LOW] FIT\n\n'
            '### Key Strengths:\n- ...\n\n'
            '### Gaps and Areas for Growth:\n- ...\n\n'
            '### Overall Assessment:\n...\n\n'
            '### Recommendation:\n...'
        ),
        expected_output="Comprehensive fit assessment in the specified Markdown format.",
        agent=recruitment_expert,
        context=[job_analysis_task, cv_insights_task],
        async_execution=False
    )

    # Task 4: CV revision (uses structured CV)
    cv_revision_task = Task(
        description=(
            "You are given:\n\n"
            "STRUCTURED CV JSON (the ground truth):\n"
            f"{cv_structured_json}\n\n"
            "JOB DESCRIPTION:\n"
            f"{job_description_text}\n\n"
            "CRITICAL RULES (DO NOT BREAK):\n"
            "1. You may NOT move bullets between employers.\n"
            "2. You may NOT merge bullets from different jobs.\n"
            "3. You may NOT invent any new experience or numbers.\n"
            "4. You MUST keep all original bullets (you may reorder within a job).\n\n"
            "Your tasks:\n"
            "- Keep the same jobs and bullets, but reorder bullets within each job to match the role better.\n"
            "- Lightly rephrase bullets to emphasize relevant skills, without changing meaning.\n"
            "- Add a brief professional summary at top.\n"
            "- Include education and skills based on the structured JSON.\n\n"
            "Output a complete CV in Markdown with sections:\n"
            "1. Name & Contact (use placeholders like [Your Name], [Email]).\n"
            "2. Professional Summary.\n"
            "3. Professional Experience (one subsection per job).\n"
            "4. Education.\n"
            "5. Skills.\n\n"
            "Before finalizing, mentally check that each bullet is still under the same employer as in the JSON."
        ),
        expected_output=(
            "A full CV in Markdown, preserving all jobs and bullets but reordered and rephrased "
            "within jobs for clarity and relevance."
        ),
        output_file=cv_output_path,
        agent=cv_strategist,
        context=[fit_assessment_task],
        async_execution=True
    )

    # Task 5: Cover letter
    cover_letter_task = Task(
        description=(
            "You are given:\n\n"
            "JOB DESCRIPTION:\n"
            f"{job_description_text}\n\n"
            "STRUCTURED CV JSON:\n"
            f"{cv_structured_json}\n\n"
            "FIT ASSESSMENT:\n"
            "{fit_assessment}\n\n"
            "First, build an internal mapping of achievements to employers "
            "by carefully reading the JSON.\n"
            "Then write a 250-400 word cover letter:\n"
            "- Opening paragraph: interest + background (no specific achievements needed).\n"
            "- 1-2 body paragraphs: highlight 1-2 achievements. For each, use this pattern:\n"
            "  'In my role as [Title] at [Correct Employer from JSON], I ...'\n"
            "- Connect skills to the job's responsibilities.\n"
            "- Show alignment with the organization's mission.\n"
            "- Closing paragraph: enthusiasm + next steps.\n\n"
            "CRITICAL ACCURACY RULES:\n"
            "- Never assign an achievement to an employer not shown in the JSON.\n"
            "- If you're unsure which employer did a particular achievement, DO NOT mention it.\n"
            "- Do not change numbers or impact metrics.\n"
        ),
        expected_output="A polished, factually accurate cover letter (250-400 words) in Markdown.",
        output_file=cl_output_path,
        agent=cover_letter_writer,
        context=[fit_assessment_task],
        async_execution=True
    )

    # Task 6: QA
    qa_task = Task(
        description=(
            "You are given:\n\n"
            "STRUCTURED CV JSON (ground truth):\n"
            f"{cv_structured_json}\n\n"
            "REVISED CV (Markdown):\n"
            "{revised_cv}\n\n"
            "COVER LETTER (Markdown):\n"
            "{cover_letter}\n\n"
            "Carefully verify:\n"
            "1. All jobs in the JSON appear in the revised CV.\n"
            "2. No bullets are missing.\n"
            "3. No bullet appears under the wrong employer.\n"
            "4. No new experience or numbers were invented.\n"
            "5. Any specific achievements mentioned in the cover letter match the right employer.\n\n"
            "Decide an approval status:\n"
            "- Approved\n"
            "- Approved with Minor Revisions\n"
            "- Major Revisions Required\n\n"
            "Return a QA report in Markdown:\n"
            'Approval Status: [status]\n\n'
            '1. Factual Accuracy:\n'
            '- ...\n\n'
            '2. Completeness:\n'
            '- ...\n\n'
            '3. Employer-Achievement Mapping:\n'
            '- ...\n\n'
            '4. Cover Letter Accuracy:\n'
            '- ...\n\n'
            '5. Suggestions:\n'
            '- ...'
        ),
        expected_output="Detailed QA report with an Approval Status line and concrete checks.",
        output_file=qa_output_path,
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
            cv_insights_task,
            fit_assessment_task,
            cv_revision_task,
            cover_letter_task,
            qa_task
        ],
        verbose=True
    )

    return crew, {
        "cv_output_path": cv_output_path,
        "cl_output_path": cl_output_path,
        "qa_output_path": qa_output_path
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
    - Generate a tailored CV without mixing experiences between employers
    - Draft a factually accurate cover letter
    - Run a QA check against your original CV
    """)

    # API setup
    if not setup_openai_api():
        st.stop()
    st.success("✅ API configured successfully")

    # Upload section
    st.markdown('<div class="section-header">📄 Upload Your Documents</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Resume/CV")
        resume_file = st.file_uploader(
            "Upload your CV (PDF format)",
            type=["pdf"],
            help="Upload your current resume in PDF format"
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
                type=["txt", "md", "pdf"],
                help="Upload job description in text, markdown, or PDF format"
            )

    # Process button
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
                # 1) Parse job description into text + md file
                jd_text, jd_md_path = read_job_description(
                    job_input_method,
                    job_description_text,
                    job_file
                )

                # 2) Save and parse CV into structured JSON
                resume_path = save_uploaded_file(resume_file)
                cv_structured = parse_cv_pdf_to_structured_json(resume_path)

                # 3) Create crew & run
                crew, paths = create_agents_and_tasks(jd_text, cv_structured)

                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Step 1/3: Running analysis agents...")
                progress_bar.progress(33)

                # Inputs passed to tasks (we refer only to constants in descriptions)
                inputs = {
                    "job_description": jd_text,
                    "cv_structured_json": json.dumps(cv_structured, indent=2)
                }

                status_text.text("Step 2/3: Generating CV and cover letter...")
                progress_bar.progress(66)

                result = crew.kickoff(inputs=inputs)

                status_text.text("Step 3/3: Final QA checks...")
                progress_bar.progress(100)

                # 4) Read outputs
                revised_cv = ""
                cover_letter = ""
                qa_report = ""

                if os.path.exists(paths["cv_output_path"]):
                    with open(paths["cv_output_path"], "r", encoding="utf-8") as f:
                        revised_cv = f.read()
                if os.path.exists(paths["cl_output_path"]):
                    with open(paths["cl_output_path"], "r", encoding="utf-8") as f:
                        cover_letter = f.read()
                if os.path.exists(paths["qa_output_path"]):
                    with open(paths["qa_output_path"], "r", encoding="utf-8") as f:
                        qa_report = f.read()

                # Extract fit assessment from crew result (fallback: search in result string)
                assessment_text = ""
                try:
                    assessment_text = str(result.tasks_output[2].output)  # assuming order: analysis, cv_insights, fit...
                except Exception:
                    assessment_text = str(result)

                from datetime import datetime
                st.session_state.results = {
                    "assessment": assessment_text,
                    "revised_cv": revised_cv,
                    "cover_letter": cover_letter,
                    "qa_report": qa_report or str(result),
                    "cv_structured": cv_structured
                }
                st.session_state.generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.processing_complete = True

                status_text.text("✅ Processing complete!")
                progress_bar.empty()

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
                return

    # -----------------------------------------------------
    # Display results
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
                # Markdown download
                st.download_button(
                    label="⬇️ Download CV (Markdown)",
                    data=revised_cv,
                    file_name="revised_cv.md",
                    mime="text/markdown"
                )
                # DOCX download
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

    # Sidebar info
    with st.sidebar:
        st.markdown("### 📖 How It Works")
        st.markdown("""
        **Step 1: CV Parsing (Deterministic)**
        - PDF parsed into a structured JSON (no LLM).
        - Employer–bullet relationships are preserved.

        **Step 2: Multi-Agent Pipeline (crewAI)**
        - Job description analysis
        - CV insight extraction
        - Fit assessment
        - CV rewriting (within strict constraints)
        - Cover letter drafting

        **Step 3: QA**
        - QA agent checks outputs against the structured CV.
        """)

        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.info("Your documents are processed temporarily and not stored permanently.")

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use a reasonably structured PDF CV.
        - Provide the full job description text.
        - Always review the generated CV and cover letter before sending.
        """)


if __name__ == "__main__":
    main()
