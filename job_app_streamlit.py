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
    
    # STEP 3: Content Generation Agents (Adaptive)
    cv_strategist = Agent(
        role="Adaptive CV Strategist",
        goal=(
            "Optimize CV presentation while maintaining 100% factual accuracy:\n"
            "CRITICAL RULES:\n"
            "1. NEVER move experiences between different employers or organizations\n"
            "2. NEVER combine or merge information from different positions\n"
            "3. NEVER add achievements that aren't in the original CV\n"
            "4. PRESERVE all original experiences, dates, and organizational details\n"
            "5. Keep ALL bullet points and details from the original CV\n\n"
            "WHAT TO DO:\n"
            "- Reorder bullet points to lead with most relevant achievements\n"
            "- Adjust phrasing to emphasize alignment (without changing facts)\n"
            "- Add context that connects experiences to job requirements\n"
            "- Strengthen action verbs while keeping original meaning\n"
            "- Highlight transferable skills in descriptions\n\n"
            "APPROACH BY FIT LEVEL:\n"
            "- HIGH FIT (75%+): Lead with strongest direct alignments, confident tone\n"
            "- MEDIUM/LOW FIT (<75%): Emphasize transferable skills, growth potential"
        ),
        tools=[semantic_pdf, semantic_mdx],
        verbose=True,
        backstory=(
            "You are a meticulous CV editor who treats candidate information as sacred. "
            "You understand that fabrication or mixing up experiences is unethical and harmful. "
            "Your expertise is in strategic presentation - reframing and reordering existing "
            "content to maximize impact while maintaining complete factual accuracy. You NEVER "
            "move achievements between different employers or positions."
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
        tools=[semantic_pdf, semantic_mdx],
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
    
    # Get output directory
    output_dir = tempfile.gettempdir()
    cv_output = os.path.join(output_dir, 'revised_cv.md')
    cl_output = os.path.join(output_dir, 'cover_letter.md')
    
    # STEP 3 TASKS: Adaptive Content Generation
    cv_revision_task = Task(
        description=(
            "Create an optimized CV that maintains complete factual accuracy.\n\n"
            "STEP 1 - EXTRACT ORIGINAL STRUCTURE:\n"
            "Carefully read the original CV and note:\n"
            "- Every employer/organization name with exact dates\n"
            "- Every position title\n"
            "- Every single bullet point and achievement under each position\n"
            "- Education, skills, and other sections\n\n"
            "STEP 2 - ANALYZE JOB ALIGNMENT:\n"
            "Identify which experiences are most relevant to the job requirements.\n\n"
            "STEP 3 - STRATEGIC REFRAMING (NOT REWRITING):\n"
            "For each position in the ORIGINAL CV:\n"
            "- Keep the EXACT employer name, position title, and dates\n"
            "- Include ALL original bullet points (don't remove content)\n"
            "- Reorder bullets to lead with most job-relevant items\n"
            "- Adjust phrasing to highlight alignment WITHOUT changing facts\n"
            "- Add brief context phrases that connect to job requirements\n\n"
            "CRITICAL RULES:\n"
            "❌ NEVER move an achievement from Company A to Company B\n"
            "❌ NEVER combine experiences from different positions\n"
            "❌ NEVER add achievements not in the original CV\n"
            "❌ NEVER remove substantial content\n"
            "✅ DO reorder bullets within each position\n"
            "✅ DO adjust phrasing to emphasize relevance\n"
            "✅ DO add brief connective phrases\n"
            "✅ DO strengthen action verbs while keeping meaning\n\n"
            "TONE BY FIT LEVEL:\n"
            "- HIGH FIT (75%+): Confident, achievement-focused language\n"
            "- MEDIUM/LOW FIT (<75%): Emphasize transferable skills and growth potential\n\n"
            "OUTPUT FORMAT:\n"
            "Use clear markdown with sections for:\n"
            "- Professional Summary (brief, tailored to role)\n"
            "- Professional Experience (maintain chronological order from original)\n"
            "- Education\n"
            "- Skills\n"
            "- Any other sections from original CV"
        ),
        expected_output=(
            "Complete CV in Markdown format that:\n"
            "1. Preserves ALL original employers, positions, dates, and achievements\n"
            "2. Reorders content strategically without fabrication\n"
            "3. Uses language that emphasizes job alignment\n"
            "4. Maintains 100% factual accuracy\n"
            "5. Includes all details from original CV"
        ),
        output_file=cv_output,
        agent=cv_strategist,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=True
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
            "Example CORRECT format:\n"
            "'In my role as [Position] at [Correct Employer], I [achievement]...'\n\n"
            "Example INCORRECT (DO NOT DO THIS):\n"
            "'At [Company A], I achieved [something that actually happened at Company B]'\n\n"
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
        output_file=cl_output,
        agent=cover_letter_writer,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=True
    )
    
    qa_task = Task(
        description=(
            "Conduct thorough quality assurance review:\n\n"
            "STEP 1 - FACTUAL VERIFICATION:\n"
            "Cross-reference the revised CV with the original CV line by line:\n"
            "- Check EVERY employer name is exactly correct\n"
            "- Verify EVERY achievement is under the correct employer\n"
            "- Confirm NO experiences were moved between different positions/companies\n"
            "- Validate ALL dates match the original\n\n"
            "STEP 2 - COMPLETENESS CHECK:\n"
            "- Verify ALL positions from original CV are included\n"
            "- Confirm ALL major achievements are preserved\n"
            "- Check that detail level is maintained\n\n"
            "STEP 3 - QUALITY ASSESSMENT:\n"
            "- Tone appropriateness for fit level\n"
            "- Grammar and professionalism\n"
            "- Formatting consistency\n"
            "- Overall persuasiveness\n\n"
            "STEP 4 - COVER LETTER ACCURACY CHECK:\n"
            "For EACH achievement or experience mentioned in the cover letter:\n"
            "- Cross-reference with CV extraction\n"
            "- Verify the employer name is correct\n"
            "- Confirm no achievements were misattributed\n"
            "- Check that all claims are factually accurate\n\n"
            "STEP 5 - COVER LETTER QUALITY:\n"
            "- Appropriate tone for fit level\n"
            "- Professional quality and grammar\n"
            "- Persuasive within honest bounds\n"
            "- Proper business letter format\n\n"
            "OUTPUT FORMAT:\n"
            "Approval Status: [Approved / Approved with Minor Revisions / Major Revisions Required]\n\n"
            "1. Verification of Fabricated Information:\n"
            "   [Confirm each employer's achievements are correct or list discrepancies]\n\n"
            "2. Tone Appropriateness for Fit Level:\n"
            "   [Assessment]\n\n"
            "3. Consistency with Original Documents:\n"
            "   [Verification results]\n\n"
            "4. Cover Letter Accuracy:\n"
            "   [Verify each achievement is attributed to correct employer]\n\n"
            "5. Grammar and Professionalism:\n"
            "   [Notes and suggestions]\n\n"
            "6. Persuasiveness within Honest Bounds:\n"
            "   [Assessment]\n\n"
            "Revision Notes:\n"
            "[If needed, specific actionable revisions]"
        ),
        expected_output=(
            "Detailed QA report with approval status and verification that:\n"
            "- No information was fabricated or misattributed\n"
            "- All content is factually accurate\n"
            "- Tone is appropriate\n"
            "- Quality standards are met"
        ),
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
                
                # Extract individual task outputs
                task_outputs = {}
                for task in crew.tasks:
                    if hasattr(task, 'output') and task.output:
                        task_outputs[task.agent.role] = str(task.output)
                
                # Get the fit assessment specifically
                fit_assessment_output = task_outputs.get('Recruitment Assessment Expert', str(result))
                
                # Read generated files - try multiple possible locations
                temp_dir = tempfile.gettempdir()
                possible_locations = [
                    temp_dir,
                    os.getcwd(),
                    '/mount/src/llm_final_project',
                    '.'
                ]
                
                revised_cv = None
                cover_letter = None
                
                # Try to find the files
                for location in possible_locations:
                    cv_path = os.path.join(location, 'revised_cv.md')
                    cl_path = os.path.join(location, 'cover_letter.md')
                    
                    if os.path.exists(cv_path) and os.path.exists(cl_path):
                        with open(cv_path, 'r', encoding='utf-8') as f:
                            revised_cv = f.read()
                        with open(cl_path, 'r', encoding='utf-8') as f:
                            cover_letter = f.read()
                        break
                
                # If files not found, extract from crew result
                if not revised_cv or not cover_letter:
                    st.warning("⚠️ Output files not found in expected locations. Extracting from agent outputs...")
                    result_str = str(result)
                    
                    # For now, use the full result as a placeholder
                    revised_cv = "# Revised CV\n\n" + result_str
                    cover_letter = "# Cover Letter\n\n" + result_str
                
                # Store results with timestamp
                from datetime import datetime
                st.session_state.results = {
                    'assessment': fit_assessment_output,
                    'revised_cv': revised_cv,
                    'cover_letter': cover_letter,
                    'qa_report': str(result),  # Store QA separately
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
            "📄 Revised CV",
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
                    st.info("💡 **Use this to verify accuracy:** Check that all achievements in your revised CV and cover letter match the organizations listed here.")
                    st.markdown(cv_analysis)
        
        with tab2:
            st.markdown("### Your Tailored CV")
            st.info("⚠️ **Important:** Always review the CV carefully to ensure all information is accurate and no experiences were misattributed between employers.")
            st.markdown(st.session_state.results['revised_cv'])
            st.download_button(
                label="⬇️ Download CV",
                data=st.session_state.results['revised_cv'],
                file_name="revised_cv.md",
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
            
            # Create combined document with proper assessment
            combined = f"""# Job Application Package - Generated by AI Assistant

## Fit Assessment
{assessment}

---

## Quality Assurance Report
{st.session_state.results.get('qa_report', 'No QA report available')}

---

## Revised CV
{st.session_state.results['revised_cv']}

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
