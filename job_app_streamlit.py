import streamlit as st 
import os
import tempfile
from pathlib import Path
import json
from typing import Dict, List, Tuple
import PyPDF2
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool, MDXSearchTool
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

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return path"""
    temp_dir = tempfile.gettempdir()
    name = Path(uploaded_file.name)
    temp_path = os.path.join(temp_dir, f"temp_{name.stem}{name.suffix}")
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def save_text_as_markdown(text: str, filename: str):
    """Save text content as markdown file"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return temp_path

def create_agents_and_tasks(resume_path: str, job_desc_path: str):
    """Create the multi-agent system with step-by-step workflow"""
    
    # Initialize tools (still used for extraction/assessment if configured for your env)
    semantic_pdf = PDFSearchTool()
    semantic_mdx = MDXSearchTool()
    
    # STEP 1: Information Extraction Agents
    job_description_analyzer = Agent(
        role="Job Description Analyzer",
        goal=(
            "Extract and structure key information from job descriptions: "
            "1) List of responsibilities, 2) Required skills and qualifications, "
            "3) Company culture indicators, 4) Experience level requirements."
        ),
        tools=[semantic_mdx],
        verbose=True,
        backstory=(
            "You are an expert in parsing job postings with precision. You use "
            "semantic search and embeddings to identify and categorize job requirements "
            "accurately. You structure information for downstream analysis."
        )
    )
    
    cv_parser = Agent(
        role="CV Parser and Analyzer",
        goal=(
            "Extract complete, structured information from the candidate's CV:\n"
            "EXTRACT WITH PRECISION:\n"
            "1. For EACH position:\n"
            "   - Exact employer/organization name\n"
            "   - Exact position title\n"
            "   - Exact dates\n"
            "   - EVERY bullet point and achievement\n"
            "   - Quantified results (numbers, percentages, impact)\n"
            "2. Skills inventory:\n"
            "   - Technical skills with proficiency levels\n"
            "   - Soft skills demonstrated\n"
            "   - Domain expertise\n"
            "3. Education:\n"
            "   - Degrees, institutions, dates\n"
            "   - Certifications and training\n"
            "4. Career insights:\n"
            "   - Progression trajectory\n"
            "   - Leadership roles\n"
            "   - Cross-functional experience\n\n"
            "CRITICAL: Maintain a complete, detailed inventory. Do not summarize or condense."
        ),
        tools=[semantic_pdf],
        verbose=True,
        backstory=(
            "You are an expert at comprehensive CV analysis using embedding-based retrieval. "
            "You understand that preserving every detail is crucial for downstream agents. "
            "You create a complete, structured representation that captures not just what's "
            "explicitly stated, but also the underlying competencies and achievements demonstrated. "
            "You never lose information through over-summarization."
        )
    )
    
    # STEP 2: Assessment Agent
    recruitment_expert = Agent(
        role="Recruitment Assessment Expert",
        goal=(
            "Perform quantitative and qualitative fit assessment: "
            "1) Calculate a numerical fit score (0-100%), "
            "2) Identify strengths and alignment areas, "
            "3) Recognize gaps and weaknesses, "
            "4) Determine fit category (High: 75%+, Medium: 50-74%, Low: <50%)."
        ),
        tools=[semantic_pdf, semantic_mdx],
        verbose=True,
        backstory=(
            "As an experienced recruiter with expertise in candidate evaluation, you "
            "assess fit holistically. You use retrieved context from both documents to "
            "provide honest, data-driven assessments with specific percentage scores."
        )
    )
    
    # STEP 3: Content / Feedback Generation Agents (Adaptive)
    cv_strategist = Agent(
        role="Adaptive CV Strategist",
        goal=(
            "Provide detailed, actionable feedback on how to tweak and enhance the candidate's "
            "existing CV so it better highlights their unique value proposition for the target job.\n\n"
            "WHAT TO DELIVER:\n"
            "1. High-level feedback on overall structure, clarity, and narrative.\n"
            "2. Section-by-section suggestions (Summary, Experience, Education, Skills, etc.).\n"
            "3. Bullet-level suggestions for key roles (what to emphasize, what to rephrase, where to quantify).\n"
            "4. Suggestions on ordering, grouping, and emphasis to better match the job description.\n"
            "5. Concrete examples of improved bullet points (but do NOT rewrite the entire CV).\n\n"
            "CRITICAL RULES:\n"
            "- DO NOT fabricate or invent new experiences or achievements.\n"
            "- DO NOT move experiences between employers.\n"
            "- Focus on sharpening language, highlighting relevant achievements, and clarifying impact.\n"
            "- Emphasize the candidate's unique value proposition for this specific role.\n"
        ),
        # No tools: rely on context from previous tasks
        verbose=True,
        backstory=(
            "You are a meticulous CV coach. Rather than rewriting documents, you give precise, "
            "targeted feedback that helps candidates revise their own CVs. You explain what to change, "
            "why it matters for the role, and provide example phrasings while respecting factual accuracy."
        )
    )
    
    cover_letter_writer = Agent(
        role="Adaptive Cover Letter Writer",
        goal=(
            "Write compelling, factually accurate cover letters (250-400 words) tailored to fit level.\n\n"
            "CRITICAL ACCURACY RULES:\n"
            "1. ONLY reference experiences that are actually in the CV\n"
            "2. NEVER attribute an achievement to the wrong employer/organization\n"
            "3. VERIFY each claim against the CV parser's detailed extraction\n"
            "4. If mentioning a specific achievement, confirm which employer it was with\n"
            "5. Keep employer-achievement pairings exactly as they appear in original CV\n\n"
            "CONTENT APPROACH BY FIT LEVEL:\n"
            "- HIGH FIT (75%+): Confident narrative with direct capability matches\n"
            "- MEDIUM FIT (50-74%): Balanced letter acknowledging gaps while emphasizing \n"
            "  transferable skills and genuine interest\n"
            "- LOW FIT (<50%): Professional letter transparently addressing fit limitations \n"
            "  while highlighting growth mindset and relevant competencies\n\n"
            "STRUCTURE:\n"
            "1. Opening: Express interest and briefly state relevant background\n"
            "2. Body (2-3 paragraphs): Connect specific experiences to job requirements\n"
            "   - Reference actual positions and achievements from CV\n"
            "   - Ensure employer names are correct for each achievement mentioned\n"
            "3. Closing: Express enthusiasm and request for next steps"
        ),
        # No tools: rely on context
        verbose=True,
        backstory=(
            "You write persuasive yet rigorously honest cover letters. You understand that "
            "misattributing achievements is a critical error that could harm the candidate's "
            "credibility. Before mentioning any achievement, you verify it against the CV parser's "
            "extraction to ensure you're attributing it to the correct employer. You adapt tone "
            "based on fit assessment while maintaining complete factual integrity."
        )
    )
    
    # Quality Assurance
    quality_assurance_agent = Agent(
        role="Quality Assurance Specialist",
        goal=(
            "Review outputs for accuracy, consistency, and appropriateness: "
            "1) Verify no fabricated experiences are suggested, "
            "2) Ensure tone matches fit level, "
            "3) Check alignment with original documents, "
            "4) Validate that CV feedback is actionable and respectful of candidate constraints."
        ),
        # No tools: rely on context
        verbose=True,
        backstory=(
            "You are the final gatekeeper ensuring quality and integrity. You verify "
            "that outputs are truthful, well-crafted, and appropriately calibrated to "
            "the candidate's actual fit level."
        )
    )
    
    # STEP 1 TASKS: Extraction
    job_extraction_task = Task(
        description=(
            f"Analyze the job description at {job_desc_path}. "
            "Use semantic search to extract:\n"
            "1. Core responsibilities (list format)\n"
            "2. Required technical skills\n"
            "3. Required soft skills\n"
            "4. Qualifications (education, experience level)\n"
            "5. Company culture indicators\n"
            "Provide structured output with clear categorization."
        ),
        expected_output="Structured JSON-like output with categorized job requirements",
        agent=job_description_analyzer,
        async_execution=False
    )
    
    cv_extraction_task = Task(
        description=(
            f"Parse the resume at {resume_path} with complete detail preservation. "
            "Use embedding-based retrieval to extract:\n\n"
            "FOR EACH PROFESSIONAL EXPERIENCE:\n"
            "Position: [Exact title]\n"
            "Organization: [Exact name]\n"
            "Dates: [Exact dates]\n"
            "Responsibilities & Achievements:\n"
            "- [List EVERY bullet point verbatim]\n"
            "- [Include ALL quantified results]\n"
            "- [Capture ALL mentioned projects]\n\n"
            "SKILLS INVENTORY:\n"
            "Technical Skills: [Comprehensive list]\n"
            "Soft Skills: [All demonstrated skills]\n"
            "Domain Expertise: [Areas of specialization]\n\n"
            "EDUCATION & CERTIFICATIONS:\n"
            "[Complete list with institutions and dates]\n\n"
            "CAREER INSIGHTS:\n"
            "- Years of experience in relevant areas\n"
            "- Leadership scope and team sizes\n"
            "- Geographic experience\n"
            "- Industry exposure\n\n"
            "OUTPUT: Comprehensive structured breakdown preserving ALL details from CV."
        ),
        expected_output=(
            "Detailed structured breakdown with:\n"
            "- Complete list of positions with ALL achievements preserved\n"
            "- Full skills inventory\n"
            "- Complete education/certification history\n"
            "- Career progression analysis\n"
            "NO INFORMATION LOSS - every detail matters for downstream agents"
        ),
        agent=cv_parser,
        async_execution=False
    )
    
    # STEP 2 TASK: Assessment
    fit_assessment_task = Task(
        description=(
            "Based on extracted job requirements and CV information:\n"
            "1. Calculate a numerical fit score (0-100%) with detailed justification\n"
            "2. List 3-5 key strengths and alignments with specific examples\n"
            "3. Identify 2-4 gaps or weaknesses with specific missing qualifications\n"
            "4. Categorize fit level: HIGH (75%+), MEDIUM (50-74%), LOW (<50%)\n"
            "5. Provide hiring likelihood assessment with reasoning\n\n"
            "Use retrieved context from both documents. Be honest and data-driven.\n\n"
            "FORMAT YOUR RESPONSE AS:\n"
            "## Fit Score: [X]%\n"
            "**Category:** [HIGH/MEDIUM/LOW] FIT\n\n"
            "### Key Strengths:\n"
            "- [List each strength with specific evidence from CV]\n\n"
            "### Gaps and Areas for Growth:\n"
            "- [List each gap with specific missing requirement]\n\n"
            "### Overall Assessment:\n"
            "[Provide 2-3 paragraph narrative explaining the score, addressing:\n"
            "- Why this specific percentage?\n"
            "- What are the strongest alignment points?\n"
            "- What are the main concerns?\n"
            "- What is the hiring likelihood and why?]\n\n"
            "### Recommendation:\n"
            "[Brief recommendation on application strategy]"
        ),
        expected_output=(
            "Comprehensive assessment report with: numerical score with justification, "
            "fit category, detailed strengths list with examples, detailed gaps list, "
            "narrative assessment explaining the scoring, and hiring likelihood with reasoning"
        ),
        agent=recruitment_expert,
        context=[job_extraction_task, cv_extraction_task],
        async_execution=False
    )
    
    # STEP 3 TASKS: CV Feedback and Cover Letter
    cv_feedback_task = Task(
        description=(
            "Provide detailed, actionable feedback on how to tweak and enhance the candidate's "
            "existing CV so it better matches the job description.\n\n"
            "USE AS INPUT:\n"
            "- Parsed CV information\n"
            "- Job requirements\n"
            "- Fit assessment\n\n"
            "STRUCTURE YOUR OUTPUT AS MARKDOWN WITH THESE SECTIONS:\n"
            "## 1. Overall Impression\n"
            "- Briefly summarize how well the current CV markets the candidate for this role.\n\n"
            "## 2. Unique Value Proposition\n"
            "- Explain what makes this candidate stand out for this role and how to emphasize that.\n\n"
            "## 3. Section-by-Section Feedback\n"
            "- For each section (Summary/Profile, Experience, Education, Skills, Other):\n"
            "  - What works well\n"
            "  - What could be improved\n"
            "  - Concrete suggestions (e.g., 'Move X higher', 'Group Y with Z', 'Clarify impact').\n\n"
            "## 4. Bullet-Level Suggestions for Key Roles\n"
            "- For the 2-3 most relevant roles:\n"
            "  - Identify 1-3 bullets that could be stronger\n"
            "  - Suggest improved, example bullet phrasings (do NOT rewrite the entire CV).\n\n"
            "## 5. Tailoring to This Job\n"
            "- Explain exactly what to tweak to align better with the job description "
            "(e.g., emphasize certain skills, add metrics, reframe certain projects).\n\n"
            "RULES:\n"
            "- DO NOT invent new experiences or achievements.\n"
            "- DO NOT move experiences between employers.\n"
            "- Focus on clarity, impact, quantification, and alignment."
        ),
        expected_output=(
            "Markdown document with structured, actionable feedback that the candidate can use "
            "to manually edit and improve their existing CV for this specific job."
        ),
        agent=cv_strategist,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=False
    )
    
    cover_letter_task = Task(
        description=(
            "Write a 250-400 word cover letter with STRICT factual accuracy.\n\n"
            "STEP 1 - REVIEW CV EXTRACTION:\n"
            "Carefully review the CV Parser's complete extraction to understand:\n"
            "- Which achievements belong to which employer\n"
            "- Exact position titles and organizations\n"
            "- Timeline of candidate's career\n\n"
            "STEP 2 - SELECT RELEVANT EXPERIENCES:\n"
            "Based on job requirements and fit assessment, identify 2-3 key experiences "
            "to highlight in the letter. NOTE THE EXACT EMPLOYER for each.\n\n"
            "STEP 3 - WRITE WITH VERIFICATION:\n"
            "For each experience or achievement you mention:\n"
            "- VERIFY it's from the CV extraction\n"
            "- CONFIRM which employer/organization it's associated with\n"
            "- STATE the employer name correctly when referencing the achievement\n\n"
            "TONE BY FIT LEVEL:\n\n"
            "IF FIT SCORE >= 75% (HIGH FIT):\n"
            "- Open with confidence and specific alignment to role\n"
            "- Detail 2-3 direct experience matches with correct employer attribution\n"
            "- Express enthusiasm backed by qualifications\n"
            "- Close with strong call to action\n"
            "- Language: Confident, achievement-oriented, direct\n\n"
            "IF FIT SCORE 50-74% (MEDIUM FIT):\n"
            "- Acknowledge partial fit professionally\n"
            "- Emphasize 2-3 transferable skills/experiences with correct attribution\n"
            "- Show genuine interest and quick learning ability\n"
            "- Express enthusiasm for growth opportunity\n"
            "- Language: Balanced, honest about gaps, emphasizes potential\n\n"
            "IF FIT SCORE < 50% (LOW FIT):\n"
            "- Be transparent about experience differences\n"
            "- Focus on 1-2 most relevant competencies with correct attribution\n"
            "- Demonstrate growth mindset and adaptability\n"
            "- Express authentic interest while realistic about fit\n"
            "- Language: Honest, humble, emphasizes learning agility\n\n"
            "STRUCTURE:\n"
            "Paragraph 1: Opening with interest and brief relevant background summary\n"
            "Paragraph 2-3: Connect specific experiences to job (with correct employers!)\n"
            "Paragraph 4: Express alignment with mission/values and request next steps\n\n"
            "FORMAT:\n"
            "Use standard business letter format with placeholders:\n"
            "[Your Name]\n"
            "[Your Contact Info]\n"
            "[Date]\n\n"
            "[Hiring Manager's Name]\n"
            "[Company Name]\n"
            "[Company Address]\n\n"
            "Dear [Hiring Manager's Name],\n\n"
            "[Letter content]\n\n"
            "Sincerely,\n"
            "[Your Name]"
        ),
        expected_output=(
            "Professional cover letter 250-400 words that:\n"
            "1. Uses correct employer names for every achievement mentioned\n"
            "2. References only experiences that exist in the CV\n"
            "3. Tone-matches the fit level appropriately\n"
            "4. Maintains complete factual accuracy\n"
            "5. Is compelling within honest bounds"
        ),
        agent=cover_letter_writer,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=False
    )
    
    qa_task = Task(
        description=(
            "Conduct thorough quality assurance review of the CV feedback, fit assessment, "
            "and cover letter:\n\n"
            "1. FACTUAL VERIFICATION:\n"
            "- Ensure no suggested changes imply fabricated or non-existent experiences.\n"
            "- Confirm examples of improved bullets stay within the facts of the CV.\n\n"
            "2. CONSISTENCY & ALIGNMENT:\n"
            "- Check that CV feedback, fit assessment, and cover letter are consistent with each other.\n"
            "- Verify that all three align with the job description.\n\n"
            "3. ACTIONABILITY OF CV FEEDBACK:\n"
            "- Ensure feedback is concrete and specific (not vague).\n"
            "- Confirm the user could realistically implement the suggestions.\n\n"
            "4. TONE & PROFESSIONALISM:\n"
            "- Confirm tone is supportive, professional, and appropriate to the fit level.\n\n"
            "OUTPUT FORMAT:\n"
            "Approval Status: [Approved / Approved with Minor Revisions / Major Revisions Required]\n\n"
            "1. Factual Accuracy:\n"
            "   [Summary]\n\n"
            "2. Consistency Across Outputs:\n"
            "   [Summary]\n\n"
            "3. CV Feedback Quality:\n"
            "   [Summary]\n\n"
            "4. Cover Letter Quality:\n"
            "   [Summary]\n\n"
            "Revision Notes:\n"
            "[If needed, specific actionable revisions]"
        ),
        expected_output=(
            "Detailed QA report with approval status and verification that:\n"
            "- No information was fabricated or misattributed\n"
            "- CV feedback is actionable and respectful of constraints\n"
            "- Tone is appropriate\n"
            "- Quality standards are met"
        ),
        agent=quality_assurance_agent,
        context=[cv_feedback_task, cover_letter_task, fit_assessment_task],
        async_execution=False
    )
    
    # Create and return crew
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
        verbose=True
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
    This tool uses a multi-agent AI system with retrieval-augmented generation to:
    - Extract and analyze job requirements using embeddings
    - Assess candidate fit quantitatively and qualitatively
    - Provide targeted feedback to strengthen your CV for a specific job
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
        # Only Word or text
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
            job_description = st.text_area(
                "Paste job description here",
                height=300,
                help="Copy and paste the full job posting"
            )
        else:
            job_file = st.file_uploader(
                "Upload job description",
                type=['txt', 'md', 'docx'],
                help="Upload job description in text, markdown, or Word (.docx) format"
            )
    
    # Process button
    if st.button("🚀 Analyze and Generate Feedback & Materials", type="primary"):
        if not resume_file:
            st.error("❌ Please upload your resume")
            return
        
        if job_input_method == "Paste Text" and not job_description:
            st.error("❌ Please provide a job description")
            return
        elif job_input_method == "Upload File" and not job_file:
            st.error("❌ Please upload a job description file")
            return
        
        with st.spinner("🔄 Processing your application... This may take a few minutes."):
            try:
                # Save resume
                resume_path = save_uploaded_file(resume_file)
                
                # Save job description
                if job_input_method == "Paste Text":
                    job_desc_path = save_text_as_markdown(job_description, "job_description.md")
                else:
                    job_desc_path = save_uploaded_file(job_file)
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Step 1/3: Extracting information from documents...")
                progress_bar.progress(33)
                
                # Create and run crew
                crew = create_agents_and_tasks(resume_path, job_desc_path)
                
                status_text.text("Step 2/3: Assessing candidate fit and drafting outputs...")
                progress_bar.progress(66)
                
                inputs = {
                    'job_posting': job_desc_path,
                    'resume': resume_path
                }
                
                result = crew.kickoff(inputs=inputs)
                
                status_text.text("Step 3/3: Finalizing feedback and materials...")
                progress_bar.progress(100)
                
                # Extract individual task outputs
                task_outputs = {}
                for task in crew.tasks:
                    if hasattr(task, 'output') and task.output:
                        task_outputs[task.agent.role] = str(task.output)
                
                # Get specific outputs
                fit_assessment_output = task_outputs.get('Recruitment Assessment Expert', str(result))
                cv_feedback_output = task_outputs.get('Adaptive CV Strategist', 'No CV feedback generated.')
                cover_letter_output = task_outputs.get('Adaptive Cover Letter Writer', 'No cover letter generated.')
                qa_output = task_outputs.get('Quality Assurance Specialist', str(result))
                
                # Store results with timestamp
                from datetime import datetime
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
        
        # Extract fit score
        assessment = st.session_state.results['assessment']
        fit_score, fit_category = extract_fit_score(assessment)
        
        # Display fit score with color coding
        fit_class = "high-fit" if fit_category == "HIGH" else "medium-fit" if fit_category == "MEDIUM" else "low-fit"
        st.markdown(f'''
            <div class="fit-score {fit_class}">
                Fit Score: {fit_score}%<br>
                <span style="font-size: 1.5rem;">{fit_category} FIT</span>
            </div>
        ''', unsafe_allow_html=True)
        
        # Tabs for different outputs
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
            
            # Show breakdown if available
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
                "🔧 **How to use this:** Go back to your original CV file and apply the suggestions "
                "manually—especially around ordering, bullet phrasing, and quantification. "
                "The tool does not overwrite your CV; it guides you on how to improve it."
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
            st.info("⚠️ **Important:** Verify that all achievements mentioned are attributed to the correct employer. Personalize placeholders like [Your Name] and [Hiring Manager's Name].")
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
        - Semantic parsing of job description
        - CV analysis using embeddings
        
        **Step 2: Fit Assessment**
        - Quantitative scoring (0-100%)
        - Strength/gap identification
        - Category assignment
        
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
        - Include complete job descriptions
        - Apply the CV feedback directly in your original document
        - Review generated materials before sending
        - Adjust language to match your authentic voice
        """)

if __name__ == "__main__":
    main()
