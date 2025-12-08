import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import Tuple
from datetime import datetime

from crewai import Agent, Task, Crew
from docx import Document  # pip install python-docx

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Job Application Assistant",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for better styling
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

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None


def setup_openai_api():
    """Configure OpenAI API from Streamlit secrets or user input"""
    if 'OPENAI_API_KEY' in st.secrets:
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
        os.environ['OPENAI_MODEL_NAME'] = st.secrets.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
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
                os.environ['OPENAI_API_KEY'] = api_key
                os.environ['OPENAI_MODEL_NAME'] = model_name
                return True
            else:
                st.warning("⚠️ Please enter your OpenAI API key to continue")
                return False


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return path"""
    temp_dir = tempfile.gettempdir()
    name = Path(uploaded_file.name)
    temp_path = os.path.join(temp_dir, f"temp_{name.stem}{name.suffix}")
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def read_file_to_text(file_path: str) -> str:
    """Read .txt, .md, or .docx into a text string."""
    ext = Path(file_path).suffix.lower()
    if ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == '.docx':
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return ""


def create_agents_and_tasks(resume_text: str, job_desc_text: str) -> Crew:
    """Create the multi-agent system with step-by-step workflow using raw text."""

    # ---------- AGENTS (no tools, rely on context text) ----------

    job_description_analyzer = Agent(
        role="Job Description Analyzer",
        goal=(
            "Extract and structure key information from job descriptions: "
            "1) List of responsibilities, 2) Required skills and qualifications, "
            "3) Company culture indicators, 4) Experience level requirements."
        ),
        verbose=False,
        backstory=(
            "You are an expert in parsing job postings with precision. You identify and categorize "
            "job requirements accurately and structure information for downstream analysis."
        )
    )

    cv_parser = Agent(
        role="CV Parser and Analyzer",
        goal=(
            "Extract complete, structured information from the candidate's CV:\n"
            "1. For EACH position:\n"
            "   - Exact employer/organization name\n"
            "   - Exact position title\n"
            "   - Exact dates\n"
            "   - EVERY bullet point and achievement\n"
            "   - Quantified results (numbers, percentages, impact)\n"
            "2. Skills inventory (technical, soft, domain knowledge)\n"
            "3. Education and certifications\n"
            "4. Career insights: progression, leadership, cross-functional experience.\n"
            "CRITICAL: Maintain a complete, detailed inventory. Do not summarize or condense."
        ),
        verbose=False,
        backstory=(
            "You are an expert at comprehensive CV analysis. You preserve every detail and create a "
            "structured representation that captures competencies and achievements."
        )
    )

    recruitment_expert = Agent(
        role="Recruitment Assessment Expert",
        goal=(
            "Perform quantitative and qualitative fit assessment: "
            "1) Calculate a numerical fit score (0-100%), "
            "2) Identify strengths and alignment areas, "
            "3) Recognize gaps and weaknesses, "
            "4) Determine fit category (High: 75%+, Medium: 50-74%, Low: <50%)."
        ),
        verbose=False,
        backstory=(
            "As an experienced recruiter, you assess fit holistically and provide honest, "
            "data-driven assessments with specific percentage scores."
        )
    )

    cv_strategist = Agent(
        role="Adaptive CV Strategist",
        goal=(
            "Provide detailed, actionable feedback on how to tweak and enhance the candidate's "
            "existing CV so it better highlights their unique value proposition for the target job.\n"
            "Do NOT rewrite the entire CV; instead, give concrete suggestions the candidate can apply."
        ),
        verbose=False,
        backstory=(
            "You are a meticulous CV coach. You give precise, targeted feedback that helps candidates "
            "revise their own CVs, explaining what to change and why, while respecting factual accuracy."
        )
    )

    cover_letter_writer = Agent(
        role="Adaptive Cover Letter Writer",
        goal=(
            "Write compelling, factually accurate cover letters (250-400 words) tailored to the fit level.\n"
            "ONLY reference experiences that are actually in the CV and never mis-attribute achievements."
        ),
        verbose=False,
        backstory=(
            "You write persuasive yet rigorously honest cover letters, cross-checking each claim with "
            "the CV information and aligning tone with the fit assessment."
        )
    )

    quality_assurance_agent = Agent(
        role="Quality Assurance Specialist",
        goal=(
            "Review outputs for accuracy, consistency, and appropriateness: "
            "1) Verify no fabricated experiences are suggested, "
            "2) Ensure tone matches fit level, "
            "3) Check alignment with the CV and job description, "
            "4) Validate that CV feedback is actionable and respectful of constraints."
        ),
        verbose=False,
        backstory=(
            "You are the final gatekeeper ensuring quality and integrity. You verify that outputs "
            "are truthful, well-crafted, and appropriate to the candidate's actual profile."
        )
    )

    # ---------- TASKS ----------

    job_extraction_task = Task(
        description=(
            "You are given the following job description:\n\n"
            f"{job_desc_text}\n\n"
            "Your task:\n"
            "1. Extract core responsibilities (list format)\n"
            "2. Extract required technical skills\n"
            "3. Extract required soft skills\n"
            "4. Extract qualifications (education, years of experience, etc.)\n"
            "5. Extract any company culture / values indicators\n\n"
            "Provide a structured response in markdown with clear headings and bullet lists."
        ),
        expected_output=(
            "Structured markdown with sections: Responsibilities, Technical Skills, "
            "Soft Skills, Qualifications, Culture/Values."
        ),
        agent=job_description_analyzer,
        async_execution=False
    )

    cv_extraction_task = Task(
        description=(
            "You are given the following CV/resume text:\n\n"
            f"{resume_text}\n\n"
            "Parse this CV and produce a comprehensive, structured breakdown:\n\n"
            "1. PROFESSIONAL EXPERIENCE:\n"
            "   - For each position, list:\n"
            "     * Organization name\n"
            "     * Position title\n"
            "     * Dates\n"
            "     * ALL bullet points and achievements (verbatim where possible)\n"
            "     * Any quantifiable results\n\n"
            "2. SKILLS INVENTORY:\n"
            "   - Technical skills\n"
            "   - Soft skills\n"
            "   - Domain expertise / sector knowledge\n\n"
            "3. EDUCATION & CERTIFICATIONS:\n"
            "   - Degrees, institutions, dates\n"
            "   - Certifications / trainings\n\n"
            "4. CAREER INSIGHTS:\n"
            "   - Years of experience in relevant areas\n"
            "   - Leadership responsibilities\n"
            "   - Geographies worked in\n"
            "   - Sectors / functions\n\n"
            "Output in markdown with clear headings and bullet lists. Preserve as much detail as possible."
        ),
        expected_output=(
            "Detailed markdown summary of the CV including experience, skills, education, and insights."
        ),
        agent=cv_parser,
        async_execution=False
    )

    fit_assessment_task = Task(
        description=(
            "Using the structured job description and CV breakdown provided in context, perform a fit assessment.\n\n"
            "1. Calculate a numerical fit score (0-100%) with justification.\n"
            "2. List 3-5 key strengths and alignment areas with specific evidence from the CV.\n"
            "3. Identify 2-4 gaps or weaknesses with reference to job requirements.\n"
            "4. Categorize fit level: HIGH (75%+), MEDIUM (50-74%), LOW (<50%).\n"
            "5. Provide a short narrative explaining hiring likelihood.\n\n"
            "FORMAT YOUR RESPONSE AS:\n"
            "## Fit Score: [X]%\n"
            "**Category:** [HIGH/MEDIUM/LOW] FIT\n\n"
            "### Key Strengths\n"
            "- ...\n\n"
            "### Gaps and Areas for Growth\n"
            "- ...\n\n"
            "### Overall Assessment\n"
            "Paragraph(s)...\n\n"
            "### Recommendation\n"
            "Brief recommendation on application strategy."
        ),
        expected_output=(
            "Markdown report with fit score, category, strengths, gaps, overall assessment, and recommendation."
        ),
        agent=recruitment_expert,
        context=[job_extraction_task, cv_extraction_task],
        async_execution=False
    )

    cv_feedback_task = Task(
        description=(
            "Provide detailed, actionable feedback on how to tweak and enhance the candidate's existing CV "
            "so it better matches the job description.\n\n"
            "Use the structured CV breakdown, job description analysis, and fit assessment from context.\n\n"
            "STRUCTURE YOUR OUTPUT AS MARKDOWN WITH THESE SECTIONS:\n\n"
            "## 1. Overall Impression\n"
            "- Briefly summarize how well the current CV markets the candidate for this role.\n\n"
            "## 2. Unique Value Proposition\n"
            "- Explain what makes this candidate stand out and how to emphasize that more clearly.\n\n"
            "## 3. Section-by-Section Feedback\n"
            "- For each section (Summary/Profile, Experience, Education, Skills, Other):\n"
            "  - What works well\n"
            "  - What could be improved\n"
            "  - Concrete suggestions (e.g., 'Move X higher', 'Clarify Y', 'Quantify Z').\n\n"
            "## 4. Bullet-Level Suggestions for Key Roles\n"
            "- For the 2-3 most relevant roles:\n"
            "  - Identify 1-3 bullets that could be stronger\n"
            "  - Suggest example improved phrasings (do NOT rewrite the entire CV).\n\n"
            "## 5. Tailoring to This Job\n"
            "- Explain exactly which skills/achievements to emphasize or reframe to better match the job.\n\n"
            "RULES:\n"
            "- DO NOT invent new experiences or achievements.\n"
            "- DO NOT move experiences between employers.\n"
            "- Focus on clarity, impact, quantification, and alignment."
        ),
        expected_output=(
            "Markdown document with structured, actionable feedback for improving the existing CV."
        ),
        agent=cv_strategist,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=False
    )

    cover_letter_task = Task(
        description=(
            "Write a 250-400 word cover letter with STRICT factual accuracy using the CV breakdown, "
            "job description analysis, and fit assessment provided in context.\n\n"
            "GUIDELINES:\n"
            "- Only reference experiences that exist in the CV.\n"
            "- Never attribute an achievement to the wrong employer or role.\n"
            "- For each concrete achievement you mention, clearly tie it to the correct position/company.\n\n"
            "TONE BY FIT LEVEL (use the fit assessment):\n"
            "- HIGH FIT: Confident and specific.\n"
            "- MEDIUM FIT: Balanced, emphasizing transferable skills.\n"
            "- LOW FIT: Honest, highlighting growth potential and relevant strengths.\n\n"
            "STRUCTURE:\n"
            "[Your Name]\n"
            "[Your Contact Info]\n"
            "[Date]\n\n"
            "[Hiring Manager's Name]\n"
            "[Company Name]\n"
            "[Company Address]\n\n"
            "Dear [Hiring Manager's Name],\n\n"
            "Paragraph 1: Interest + brief relevant background.\n"
            "Paragraph 2-3: 2-3 specific experiences/achievements connected to the job.\n"
            "Paragraph 4: Alignment with mission/values and a warm close.\n\n"
            "Sincerely,\n"
            "[Your Name]"
        ),
        expected_output=(
            "A polished, 250-400 word cover letter in markdown, with placeholders for names and addresses."
        ),
        agent=cover_letter_writer,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=False
    )

    qa_task = Task(
        description=(
            "Conduct a quality assurance review of the fit assessment, CV feedback, and cover letter.\n\n"
            "1. FACTUAL ACCURACY:\n"
            "- Ensure no suggested changes imply fabricated experiences.\n"
            "- Check that cover letter achievements match the CV breakdown.\n\n"
            "2. CONSISTENCY & ALIGNMENT:\n"
            "- Ensure the three outputs tell a coherent story aligned with the job.\n\n"
            "3. ACTIONABILITY OF CV FEEDBACK:\n"
            "- Confirm feedback is concrete and practical, not vague.\n\n"
            "4. TONE & PROFESSIONALISM:\n"
            "- Tone should be supportive, realistic, and professional.\n\n"
            "FORMAT:\n"
            "Approval Status: [Approved / Approved with Minor Revisions / Major Revisions Required]\n\n"
            "1. Factual Accuracy:\n"
            "   ...\n\n"
            "2. Consistency Across Outputs:\n"
            "   ...\n\n"
            "3. CV Feedback Quality:\n"
            "   ...\n\n"
            "4. Cover Letter Quality:\n"
            "   ...\n\n"
            "Revision Notes:\n"
            "- ..."
        ),
        expected_output=(
            "Markdown QA report with approval status and notes on accuracy, consistency, feedback, and tone."
        ),
        agent=quality_assurance_agent,
        context=[fit_assessment_task, cv_feedback_task, cover_letter_task],
        async_execution=False
    )

    crew = Crew(
        agents=[
            job_description_analyzer,
            cv_parser,
            recruitment_expert,
            cv_strategist,
            cover_letter_writer,
            quality_assurance_agent
        ],
        tasks=[
            job_extraction_task,
            cv_extraction_task,
            fit_assessment_task,
            cv_feedback_task,
            cover_letter_task,
            qa_task
        ],
        verbose=False  # suppress Thought/Action logs in the Streamlit console
    )

    return crew


def extract_fit_score(assessment_text: str) -> Tuple[int, str]:
    """Extract fit score and category from assessment text"""
    import re
    score_match = re.search(r'(\d+)%', assessment_text)
    score = int(score_match.group(1)) if score_match else 50

    if score >= 75:
        category = "HIGH"
    elif score >= 50:
        category = "MEDIUM"
    else:
        category = "LOW"

    return score, category


def main():
    st.markdown('<div class="main-header">💼 AI-Powered Job Application Assistant</div>',
                unsafe_allow_html=True)
    st.markdown("""
    This tool uses a multi-agent AI system with retrieval-augmented reasoning to:
    - Extract and analyze job requirements
    - Analyze your CV in depth
    - Assess candidate–role fit
    - Provide targeted feedback to strengthen your CV
    - Generate a tailored, factually accurate cover letter
    """)

    # API Setup
    if not setup_openai_api():
        st.stop()

    st.success("✅ API configured successfully")

    # Main input section
    st.markdown('<div class="section-header">📄 Upload Your Documents</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Resume/CV")
        resume_file = st.file_uploader(
            "Upload your CV (Word or text format)",
            type=['docx', 'txt'],
            help="Upload your current resume in .docx or .txt format"
        )

    with col2:
        st.subheader("Job Description")
        job_input_method = st.radio(
            "How would you like to provide the job description?",
            ["Paste Text", "Upload File"]
        )

        if job_input_method == "Paste Text":
            job_description_text = st.text_area(
                "Paste job description here",
                height=300,
                help="Copy and paste the full job posting"
            )
            job_file = None
        else:
            job_file = st.file_uploader(
                "Upload job description",
                type=['txt', 'md', 'docx'],
                help="Upload job description in text, markdown, or Word (.docx) format"
            )
            job_description_text = None

    if st.button("🚀 Analyze and Generate Feedback & Materials", type="primary"):
        if not resume_file:
            st.error("❌ Please upload your resume")
            return

        if job_input_method == "Paste Text" and not job_description_text:
            st.error("❌ Please provide a job description")
            return
        elif job_input_method == "Upload File" and not job_file:
            st.error("❌ Please upload a job description file")
            return

        with st.spinner("🔄 Processing your application..."):
            try:
                # Save files
                resume_path = save_uploaded_file(resume_file)
                if job_input_method == "Paste Text":
                    job_desc_path = save_text_as_markdown(job_description_text, "job_description.md")
                else:
                    job_desc_path = save_uploaded_file(job_file)

                # Read text from files
                resume_text = read_file_to_text(resume_path)
                job_text = read_file_to_text(job_desc_path)

                if not resume_text.strip():
                    st.error("❌ Could not read text from the CV file. Please upload a .docx or .txt file.")
                    return
                if not job_text.strip():
                    st.error("❌ Could not read text from the job description file.")
                    return

                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Step 1/3: Setting up agents and tasks...")
                progress_bar.progress(20)

                crew = create_agents_and_tasks(resume_text, job_text)

                status_text.text("Step 2/3: Running multi-agent analysis...")
                progress_bar.progress(60)

                # We no longer need to pass file paths as inputs; content is baked into descriptions
                result = crew.kickoff()

                status_text.text("Step 3/3: Collecting outputs...")
                progress_bar.progress(100)

                # Collect task outputs
                task_outputs = {}
                for task in crew.tasks:
                    if hasattr(task, 'output') and task.output:
                        task_outputs[task.agent.role] = str(task.output)

                fit_assessment_output = task_outputs.get('Recruitment Assessment Expert', str(result))
                cv_feedback_output = task_outputs.get('Adaptive CV Strategist', 'No CV feedback generated.')
                cover_letter_output = task_outputs.get('Adaptive Cover Letter Writer', 'No cover letter generated.')
                qa_output = task_outputs.get('Quality Assurance Specialist', str(result))

                st.session_state.results = {
                    'assessment': fit_assessment_output,
                    'cv_feedback': cv_feedback_output,
                    'cover_letter': cover_letter_output,
                    'qa_report': qa_output,
                    'all_outputs': task_outputs
                }
                st.session_state.generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.processing_complete = True

                status_text.text("✅ Processing complete!")
                progress_bar.empty()

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
                return

    # Display results
    if st.session_state.processing_complete and st.session_state.results:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)

        assessment = st.session_state.results['assessment']
        fit_score, fit_category = extract_fit_score(assessment)

        fit_class = "high-fit" if fit_category == "HIGH" else "medium-fit" if fit_category == "MEDIUM" else "low-fit"
        st.markdown(f'''
            <div class="fit-score {fit_class}">
                Fit Score: {fit_score}%<br>
                <span style="font-size: 1.5rem;">{fit_category} FIT</span>
            </div>
        ''', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Fit Assessment",
            "🛠️ CV Feedback",
            "✉️ Cover Letter",
            "✅ QA Report",
            "💾 Download All"
        ])

        with tab1:
            st.markdown("### Detailed Fit Assessment")
            st.markdown(assessment)

            if 'all_outputs' in st.session_state.results:
                with st.expander("📊 View Detailed Analysis"):
                    job_analysis = st.session_state.results['all_outputs'].get('Job Description Analyzer', 'N/A')
                    cv_analysis = st.session_state.results['all_outputs'].get('CV Parser and Analyzer', 'N/A')

                    st.markdown("#### Job Requirements Extracted:")
                    st.markdown(job_analysis)

                    st.markdown("---")

                    st.markdown("#### Candidate Profile Summary:")
                    st.info("💡 **Use this to verify accuracy:** Check that the feedback and cover letter match the experiences and employers listed here.")
                    st.markdown(cv_analysis)

        with tab2:
            st.markdown("### Feedback on Your CV")
            st.info(
                "🔧 **How to use this:** Return to your original CV file and apply the suggestions "
                "manually—especially around ordering, bullet phrasing, and quantification."
            )
            st.markdown(st.session_state.results['cv_feedback'])
            st.download_button(
                label="⬇️ Download CV Feedback",
                data=st.session_state.results['cv_feedback'],
                file_name="cv_feedback.md",
                mime="text/markdown"
            )

        with tab3:
            st.markdown("### Your Cover Letter")
            st.info("⚠️ **Important:** Verify that all achievements mentioned are attributed to the correct employer, and personalize placeholders like [Your Name] and [Hiring Manager's Name].")
            st.markdown(st.session_state.results['cover_letter'])
            st.download_button(
                label="⬇️ Download Cover Letter",
                data=st.session_state.results['cover_letter'],
                file_name="cover_letter.md",
                mime="text/markdown"
            )

        with tab4:
            st.markdown("### Quality Assurance Report")
            st.markdown(st.session_state.results.get('qa_report', 'No QA report available'))

        with tab5:
            st.markdown("### Download All Documents")

            combined = f"""# Job Application Package - Generated by AI Assistant

## Fit Assessment
{assessment}

---

## Quality Assurance Report
{st.session_state.results.get('qa_report', 'No QA report available')}

---

## CV Feedback
{st.session_state.results['cv_feedback']}

---

## Cover Letter
{st.session_state.results['cover_letter']}

---

*Generated on {st.session_state.get('generation_date', 'N/A')}*
"""
            st.download_button(
                label="⬇️ Download Complete Package",
                data=combined,
                file_name="job_application_package.md",
                mime="text/markdown"
            )

    # Sidebar information
    with st.sidebar:
        st.markdown("### 📖 How It Works")
        st.markdown("""
        **Step 1: Information Extraction**
        - Parses the job description
        - Parses your CV text
        
        **Step 2: Fit Assessment**
        - Quantitative scoring (0-100%)
        - Strength/gap identification
        - Fit category
        
        **Step 3: Tailored Support**
        - Targeted feedback to improve your existing CV
        - Factually accurate cover letter for this role
        """)

        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.info("Your documents are processed temporarily and not stored permanently.")

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use a clear, well-formatted CV in .docx or .txt
        - Include the full job description
        - Apply the CV feedback directly in your original document
        - Review and tweak the cover letter to sound like you
        """)


if __name__ == "__main__":
    main()
