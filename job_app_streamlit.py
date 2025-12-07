import streamlit as st
import os
import tempfile
from pathlib import Path
import json
from typing import Dict, List, Tuple
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

def save_uploaded_file(uploaded_file, suffix='.pdf'):
    """Save uploaded file to temporary directory and return path"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
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
    
    # Initialize tools
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
            "Parse the candidate's CV into structured components: "
            "1) Professional experiences with quantifiable achievements, "
            "2) Technical and soft skills, 3) Education and certifications, "
            "4) Career trajectory and growth indicators."
        ),
        tools=[semantic_pdf],
        verbose=True,
        backstory=(
            "You specialize in extracting meaningful information from resumes using "
            "embedding-based retrieval. You identify not just what's written, but the "
            "underlying competencies and potential in a candidate's background."
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
    
    # STEP 3: Content Generation Agents (Adaptive)
    cv_strategist = Agent(
        role="Adaptive CV Strategist",
        goal=(
            "Create role-specific CV revisions based on fit level:\n"
            "- HIGH FIT (75%+): Confident, achievement-focused revision emphasizing "
            "direct alignment and quantifiable impact.\n"
            "- MEDIUM/LOW FIT (<75%): Honest revision highlighting transferable skills, "
            "relevant experiences, and growth potential without overstating qualifications."
        ),
        tools=[semantic_pdf, semantic_mdx],
        verbose=True,
        backstory=(
            "You are a strategic CV editor who adapts your approach based on candidate fit. "
            "For strong fits, you confidently showcase alignment. For partial fits, you "
            "remain truthful while emphasizing transferable value and learning agility."
        )
    )
    
    cover_letter_writer = Agent(
        role="Adaptive Cover Letter Writer",
        goal=(
            "Write compelling cover letters (250-400 words) tailored to fit level:\n"
            "- HIGH FIT: Confident narrative demonstrating direct capability match.\n"
            "- MEDIUM FIT: Balanced letter acknowledging gaps while emphasizing transferable "
            "skills and genuine interest.\n"
            "- LOW FIT: Professional letter transparently addressing fit limitations while "
            "highlighting growth mindset and relevant competencies."
        ),
        tools=[semantic_pdf, semantic_mdx],
        verbose=True,
        backstory=(
            "You write persuasive yet honest cover letters. You adapt tone and content "
            "based on fit assessment, always maintaining professionalism and authenticity. "
            "You help candidates position themselves realistically while showing their best qualities."
        )
    )
    
    # Quality Assurance
    quality_assurance_agent = Agent(
        role="Quality Assurance Specialist",
        goal=(
            "Review outputs for accuracy, consistency, and appropriateness: "
            "1) Verify no fabricated experiences, "
            "2) Ensure tone matches fit level, "
            "3) Check alignment with original documents, "
            "4) Validate persuasiveness within honest bounds."
        ),
        tools=[semantic_pdf, semantic_mdx],
        verbose=True,
        backstory=(
            "You are the final gatekeeper ensuring quality and integrity. You verify "
            "that outputs are truthful, well-crafted, and appropriately calibrated to "
            "the candidate's actual fit level."
        )
    )
    
    # Create Tasks with clear step separation
    
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
            f"Parse the resume at {resume_path}. "
            "Use embedding-based retrieval to extract:\n"
            "1. Professional experiences with achievements\n"
            "2. Technical skills inventory\n"
            "3. Soft skills demonstrated\n"
            "4. Education and certifications\n"
            "5. Career progression indicators\n"
            "Provide structured output for analysis."
        ),
        expected_output="Structured breakdown of candidate's qualifications and experiences",
        agent=cv_parser,
        async_execution=False
    )
    
    # STEP 2 TASK: Assessment
    fit_assessment_task = Task(
        description=(
            "Based on extracted job requirements and CV information:\n"
            "1. Calculate a numerical fit score (0-100%) with justification\n"
            "2. List 3-5 key strengths and alignments\n"
            "3. Identify 2-4 gaps or weaknesses\n"
            "4. Categorize fit level: HIGH (75%+), MEDIUM (50-74%), LOW (<50%)\n"
            "5. Provide hiring likelihood assessment\n\n"
            "Use retrieved context from both documents. Be honest and data-driven."
        ),
        expected_output=(
            "Assessment report with: numerical score, fit category, strengths list, "
            "gaps list, and narrative justification"
        ),
        agent=recruitment_expert,
        context=[job_extraction_task, cv_extraction_task],
        async_execution=False
    )
    
    # STEP 3 TASKS: Adaptive Content Generation
    cv_revision_task = Task(
        description=(
            "Create a revised CV based on the fit assessment:\n\n"
            "IF FIT SCORE >= 75% (HIGH FIT):\n"
            "- Write confidently, emphasizing direct qualifications\n"
            "- Lead with strongest alignments\n"
            "- Use powerful action verbs\n"
            "- Quantify achievements prominently\n\n"
            "IF FIT SCORE < 75% (MEDIUM/LOW FIT):\n"
            "- Remain honest about experience level\n"
            "- Emphasize transferable skills\n"
            "- Highlight learning agility and adaptability\n"
            "- Show genuine interest in growth\n\n"
            "Always maintain truthfulness. Never fabricate experiences."
        ),
        expected_output="Revised CV in Markdown format, tone-appropriate for fit level",
        output_file="revised_cv.md",
        agent=cv_strategist,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=True
    )
    
    cover_letter_task = Task(
        description=(
            "Write a 250-400 word cover letter based on fit assessment:\n\n"
            "IF FIT SCORE >= 75% (HIGH FIT):\n"
            "- Open with confidence and specific role alignment\n"
            "- Detail direct experience matches\n"
            "- Express enthusiasm backed by qualifications\n"
            "- Close with strong call to action\n\n"
            "IF FIT SCORE 50-74% (MEDIUM FIT):\n"
            "- Acknowledge gaps professionally\n"
            "- Emphasize transferable skills strongly\n"
            "- Show genuine interest and quick learning ability\n"
            "- Express enthusiasm for growth opportunity\n\n"
            "IF FIT SCORE < 50% (LOW FIT):\n"
            "- Be transparent about experience differences\n"
            "- Focus on relevant competencies\n"
            "- Demonstrate growth mindset\n"
            "- Express authentic interest while realistic about fit\n\n"
            "Maintain professionalism throughout. Reference company mission when possible."
        ),
        expected_output="Professional cover letter 250-400 words, tone-matched to fit level",
        output_file="cover_letter.md",
        agent=cover_letter_writer,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=True
    )
    
    qa_task = Task(
        description=(
            "Review the revised CV and cover letter:\n"
            "1. Verify no fabricated information\n"
            "2. Check tone appropriateness for fit level\n"
            "3. Ensure consistency with original documents\n"
            "4. Validate grammar and professionalism\n"
            "5. Confirm persuasiveness within honest bounds\n\n"
            "Provide approval or specific revision requests."
        ),
        expected_output="QA report with approval status and any revision notes",
        agent=quality_assurance_agent,
        context=[cv_revision_task, cover_letter_task, fit_assessment_task],
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
            cv_revision_task,
            cover_letter_task,
            qa_task
        ],
        verbose=True
    )
    
    return crew

def extract_fit_score(assessment_text: str) -> Tuple[int, str]:
    """Extract fit score and category from assessment text"""
    # Look for percentage patterns
    import re
    score_match = re.search(r'(\d+)%', assessment_text)
    score = int(score_match.group(1)) if score_match else 50
    
    # Determine category
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
    - Generate adaptive, honest application materials based on fit level
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
            "Upload your CV (PDF format)",
            type=['pdf'],
            help="Upload your current resume in PDF format"
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
                type=['txt', 'md', 'pdf'],
                help="Upload job description in text, markdown, or PDF format"
            )
    
    # Process button
    if st.button("🚀 Analyze and Generate Application Materials", type="primary"):
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
                    job_desc_path = save_uploaded_file(job_file, suffix='.md')
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Step 1/3: Extracting information from documents...")
                progress_bar.progress(33)
                
                # Create and run crew
                crew = create_agents_and_tasks(resume_path, job_desc_path)
                
                status_text.text("Step 2/3: Assessing candidate fit...")
                progress_bar.progress(66)
                
                inputs = {
                    'job_posting': job_desc_path,
                    'resume': resume_path
                }
                
                result = crew.kickoff(inputs=inputs)
                
                status_text.text("Step 3/3: Generating application materials...")
                progress_bar.progress(100)
                
                # Read generated files
                temp_dir = tempfile.gettempdir()
                
                with open(os.path.join(temp_dir, 'revised_cv.md'), 'r', encoding='utf-8') as f:
                    revised_cv = f.read()
                
                with open(os.path.join(temp_dir, 'cover_letter.md'), 'r', encoding='utf-8') as f:
                    cover_letter = f.read()
                
                # Store results
                st.session_state.results = {
                    'assessment': str(result),
                    'revised_cv': revised_cv,
                    'cover_letter': cover_letter
                }
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Fit Assessment",
            "📄 Revised CV",
            "✉️ Cover Letter",
            "💾 Download All"
        ])
        
        with tab1:
            st.markdown("### Detailed Assessment")
            st.markdown(assessment)
        
        with tab2:
            st.markdown("### Your Tailored CV")
            st.markdown(st.session_state.results['revised_cv'])
            st.download_button(
                label="⬇️ Download CV",
                data=st.session_state.results['revised_cv'],
                file_name="revised_cv.md",
                mime="text/markdown"
            )
        
        with tab3:
            st.markdown("### Your Cover Letter")
            st.markdown(st.session_state.results['cover_letter'])
            st.download_button(
                label="⬇️ Download Cover Letter",
                data=st.session_state.results['cover_letter'],
                file_name="cover_letter.md",
                mime="text/markdown"
            )
        
        with tab4:
            st.markdown("### Download All Documents")
            
            # Create combined document
            combined = f"""# Job Application Package
            
## Fit Assessment
{assessment}

---

## Revised CV
{st.session_state.results['revised_cv']}

---

## Cover Letter
{st.session_state.results['cover_letter']}
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
        
        **Step 3: Adaptive Generation**
        - High Fit (75%+): Confident materials
        - Medium Fit (50-74%): Balanced approach
        - Low Fit (<50%): Honest, growth-focused
        """)
        
        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.info("Your documents are processed temporarily and not stored permanently.")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use a well-formatted PDF resume
        - Include complete job descriptions
        - Review generated materials before sending
        - Adjust based on your authentic voice
        """)

if __name__ == "__main__":
    main()
