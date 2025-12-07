import streamlit as st
import os
import tempfile
from pathlib import Path
import json
from typing import Dict, List, Tuple
import PyPDF2
import docx  # For Word documents
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool
import warnings
import re
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

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract complete text from PDF preserving structure.
    This maintains the original order and formatting.
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract complete text from Word document preserving structure.
    """
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return ""

def extract_text_from_file(file_path: str, file_type: str) -> str:
    """
    Extract text from various file formats.
    """
    if file_type == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_type in ['docx', 'doc']:
        return extract_text_from_docx(file_path)
    elif file_type in ['txt', 'md']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return ""
    else:
        st.error(f"Unsupported file type: {file_type}")
        return ""

def save_text_as_file(text: str, filename: str) -> str:
    """Save text content to a file and return path"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return temp_path

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

def create_agents_and_tasks(cv_text_path: str, job_desc_path: str):
    """Create the multi-agent system with step-by-step workflow"""
    
    # Initialize tools - use FileReadTool for structured text
    cv_read_tool = FileReadTool(file_path=cv_text_path)
    job_read_tool = FileReadTool(file_path=job_desc_path)
    
    # STEP 1: Information Extraction Agents
    job_description_analyzer = Agent(
        role="Job Description Analyzer",
        goal=(
            "Carefully read the job description to extract two things: 1) a structured list of job "
            "responsibilities and 2) a structured list of required skills and qualifications."
        ),
        tools=[job_read_tool],
        verbose=True,
        backstory=(
            "You are an expert in deconstructing job postings. Your keen eye for detail lets you "
            "separate responsibilities from requirements so that other agents can act on your insights. "
            "Always organize information clearly and concisely."
        )
    )
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
            "Parse CV text into a STRICTLY STRUCTURED format that preserves employer boundaries.\n\n"
            "OUTPUT MUST BE VALID JSON with this exact structure:\n"
            "{\n"
            '  "positions": [\n'
            "    {\n"
            '      "employer": "Exact company/org name",\n'
            '      "title": "Exact position title",\n'
            '      "dates": "Exact date range",\n'
            '      "location": "City, Country",\n'
            '      "responsibilities": [\n'
            '        "Bullet point 1 exactly as written",\n'
            '        "Bullet point 2 exactly as written"\n'
            "      ]\n"
            "    }\n"
            "  ],\n"
            '  "education": [...],\n'
            '  "skills": [...]\n'
            "}\n\n"
            "CRITICAL: Each position object contains ONLY content from that specific employer. "
            "Never mix content across positions. Preserve exact wording and order."
        ),
        tools=[cv_read_tool],
        verbose=True,
        backstory=(
            "You are a meticulous data parser who outputs ONLY valid JSON. You understand that "
            "mixing content between employers is catastrophic. You read the CV text sequentially "
            "and group all content under the correct employer heading. Your JSON output is the "
            "single source of truth for downstream agents."
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
            "Receive STRUCTURED JSON with employer-separated positions. Generate optimized CV "
            "maintaining 100% factual accuracy.\n\n"
            "INPUT: JSON with positions array where each object = one employer\n"
            "OUTPUT: Markdown CV where each position section uses ONLY content from its JSON object\n\n"
            "YOU MUST:\n"
            "1. Process each JSON position object independently\n"
            "2. For each position, use ONLY the responsibilities from that object\n"
            "3. Never pull content from a different position's object\n"
            "4. Reorder bullets within each position by relevance\n"
            "5. Adjust phrasing while keeping facts identical"
        ),
        tools=[cv_read_tool, job_read_tool],
        verbose=True,
        backstory=(
            "You are a CV editor who works with structured data. You receive clean JSON showing "
            "exactly which bullets belong to which employer. You would never dream of moving a "
            "bullet from position[0] to position[1]. You optimize presentation within strict "
            "boundaries defined by the JSON structure."
        )
    )
    
    cover_letter_writer = Agent(
        role="Adaptive Cover Letter Writer",
        goal=(
            "Write 250-400 word cover letters using STRUCTURED JSON CV data.\n\n"
            "INPUT: JSON showing which achievements belong to which employer\n"
            "PROCESS: Before mentioning any achievement, check the JSON to verify employer\n"
            "OUTPUT: Letter where every achievement is correctly attributed\n\n"
            "When you say 'At Company X, I did Y', verify in the JSON that Y appears "
            "under Company X's position object."
        ),
        tools=[cv_read_tool, job_read_tool],
        verbose=True,
        backstory=(
            "You write cover letters using structured CV data. You check the JSON to see which "
            "employer each achievement belongs to before mentioning it. You understand that "
            "saying 'At Company A, I did X' when X was actually done at Company B destroys "
            "credibility instantly."
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
            "Extract and structure:\n"
            "1. Core responsibilities (list format)\n"
            "2. Required technical skills\n"
            "3. Required soft skills\n"
            "4. Qualifications (education, experience level)\n"
            "5. Company culture indicators\n"
            "Provide structured output with clear categorization."
        ),
        expected_output="Structured output with categorized job requirements",
        agent=job_description_analyzer,
        async_execution=False
    )
    
    cv_extraction_task = Task(
        description=(
            f"Read the CV text from {cv_text_path}. Parse it into VALID JSON format.\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the COMPLETE CV text sequentially from start to finish\n"
            "2. Identify section boundaries (Professional Experience, Education, Skills, etc.)\n"
            "3. For each position in Professional Experience:\n"
            "   - Find the employer/organization name\n"
            "   - Find the position title\n"
            "   - Find the date range\n"
            "   - Collect ALL bullet points that follow BEFORE the next employer heading\n"
            "4. Output ONLY valid JSON - no other text\n\n"
            "JSON STRUCTURE (MUST MATCH EXACTLY):\n"
            "```json\n"
            "{\n"
            '  "positions": [\n'
            "    {\n"
            '      "employer": "Company Name",\n'
            '      "title": "Job Title",\n'
            '      "dates": "Jan 2020 - Present",\n'
            '      "location": "City, Country",\n'
            '      "responsibilities": [\n'
            '        "First bullet point",\n'
            '        "Second bullet point"\n'
            "      ]\n"
            "    }\n"
            "  ],\n"
            '  "education": [\n'
            "    {\n"
            '      "degree": "Degree name",\n'
            '      "institution": "University name",\n'
            '      "year": "Year",\n'
            '      "location": "City, Country"\n'
            "    }\n"
            "  ],\n"
            '  "skills": {\n'
            '    "technical": ["skill1", "skill2"],\n'
            '    "soft": ["skill1", "skill2"]\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "VALIDATION RULES:\n"
            "- Each position object contains ONLY content from that specific employer\n"
            "- Preserve ALL bullet points in their original order\n"
            "- Use exact wording from CV\n"
            "- Output must be parseable JSON (use proper escaping for quotes)\n"
        ),
        expected_output=(
            "Valid JSON object containing:\n"
            "- Complete list of positions with all details preserved\n"
            "- Each position strictly separated by employer\n"
            "- Education and skills sections\n"
            "MUST be valid, parseable JSON"
        ),
        agent=cv_parser,
        async_execution=False
    )
    
    # STEP 2 TASK: Assessment
    fit_assessment_task = Task(
        description=(
            "You will receive a JSON object from the CV Parser showing the candidate's structured CV data.\n\n"
            "ANALYZE THE JSON:\n"
            "- Review positions array (each object = one employer with their responsibilities)\n"
            "- Review education and skills\n"
            "- Compare against job requirements\n\n"
            "Calculate fit score (0-100%) and provide detailed assessment.\n\n"
            "FORMAT YOUR RESPONSE AS:\n"
            "## Fit Score: [X]%\n"
            "**Category:** [HIGH/MEDIUM/LOW] FIT\n\n"
            "### Key Strengths:\n"
            "- [List each strength with specific evidence]\n\n"
            "### Gaps and Areas for Growth:\n"
            "- [List each gap]\n\n"
            "### Overall Assessment:\n"
            "[2-3 paragraphs explaining score]\n\n"
            "### Recommendation:\n"
            "[Application strategy]"
        ),
        expected_output=(
            "Comprehensive assessment with numerical score, category, strengths, gaps, "
            "narrative assessment, and recommendation"
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
            "You will receive JSON from CV Parser with this structure:\n"
            '{"positions": [{"employer": "...", "title": "...", "responsibilities": [...]}]}\n\n'
            "CREATE OPTIMIZED CV:\n\n"
            "STEP 1 - PARSE THE JSON:\n"
            "- Load the JSON positions array\n"
            "- Each array element = one employer with their responsibilities\n\n"
            "STEP 2 - FOR EACH POSITION IN THE ARRAY:\n"
            "- Extract: employer, title, dates, responsibilities\n"
            "- Reorder responsibilities by relevance to job (within this position only)\n"
            "- Adjust phrasing for impact (without changing facts)\n"
            "- Write this position's section\n\n"
            "STEP 3 - ASSEMBLE CV:\n"
            "Format:\n"
            "# [Name]\n"
            "[Contact info]\n\n"
            "## Professional Summary\n"
            "[2-3 sentences]\n\n"
            "## Professional Experience\n\n"
            "### [Position Title]\n"
            "**[Employer Name]** | [Location] | [Dates]\n"
            "- [Reordered responsibility 1]\n"
            "- [Reordered responsibility 2]\n"
            "[ALL responsibilities from THIS position's JSON object]\n\n"
            "[Repeat for each position in JSON array]\n\n"
            "## Education\n"
            "[From JSON]\n\n"
            "## Skills\n"
            "[From JSON]\n\n"
            "CRITICAL RULES:\n"
            "- Process each JSON position object independently\n"
            "- NEVER take a responsibility from positions[0] and put it under positions[1]\n"
            "- Each CV section uses ONLY content from its corresponding JSON object\n\n"
            "TONE: Adjust based on fit level (confident for high fit, balanced for medium/low)"
        ),
        expected_output=(
            "Complete CV in Markdown with:\n"
            "- Each position containing ONLY content from its JSON object\n"
            "- Strategic ordering within each position\n"
            "- Enhanced phrasing\n"
            "- ZERO cross-contamination between employers"
        ),
        output_file=cv_output,
        agent=cv_strategist,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=True
    )
    
    cover_letter_task = Task(
        description=(
            "You will receive JSON from CV Parser showing which achievements belong to which employer.\n\n"
            "WRITE 250-400 WORD COVER LETTER:\n\n"
            "STEP 1 - REVIEW JSON STRUCTURE:\n"
            "The CV JSON has positions array like:\n"
            '[{"employer": "Company A", "responsibilities": [...]}, {"employer": "Company B", ...}]\n\n'
            "STEP 2 - SELECT ACHIEVEMENTS TO HIGHLIGHT:\n"
            "Choose 1-2 most relevant achievements for the letter.\n"
            "For each, note: which position index it's in → which employer that is\n\n"
            "STEP 3 - WRITE WITH VERIFIED ATTRIBUTION:\n"
            "When mentioning an achievement, use this pattern:\n"
            '"In my role as [title from that JSON object] at [employer from that JSON object], I [achievement]"\n\n'
            "WRONG (causes misattribution):\n"
            "- Taking achievement from positions[0] and saying it was at positions[1].employer\n"
            "- Mentioning your current role but describing achievement from previous role\n\n"
            "CORRECT:\n"
            "- Check JSON: positions[2] has employer='Acme Corp' and responsibility='Led $5M project'\n"
            "- Write: 'In my role as Senior Manager at Acme Corp, I led a $5M project...'\n\n"
            "STRUCTURE:\n"
            "Paragraph 1: Interest + brief background (general, no specific achievements)\n"
            "Paragraph 2: Highlight 1-2 achievements with CORRECT employer attribution\n"
            "Paragraph 3: Connect to company mission and role requirements\n"
            "Paragraph 4: Enthusiasm and call to action\n\n"
            "TONE: Match fit level (confident for high, balanced for medium, humble for low)\n\n"
            "FORMAT: Standard business letter with [placeholders] for personalization"
        ),
        expected_output=(
            "250-400 word cover letter with:\n"
            "- Correct employer attribution for any specific achievements\n"
            "- Professional business format\n"
            "- Appropriate tone for fit level"
        ),
        output_file=cl_output,
        agent=cover_letter_writer,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=True
    )
    
    qa_task = Task(
        description=(
            "VERIFY OUTPUTS AGAINST SOURCE JSON:\n\n"
            "You have access to the structured JSON showing which responsibilities belong to which employer.\n\n"
            "VERIFICATION PROCESS:\n\n"
            "STEP 1 - LOAD JSON AS REFERENCE:\n"
            "Parse the CV JSON to understand the ground truth employer-responsibility mapping.\n\n"
            "STEP 2 - CHECK REVISED CV:\n"
            "For each position section in the revised CV:\n"
            "- Identify the employer name\n"
            "- List all achievements/responsibilities mentioned\n"
            "- Cross-reference with JSON: are these from the correct position object?\n"
            "- Flag any achievement that appears under wrong employer\n\n"
            "STEP 3 - CHECK COVER LETTER:\n"
            "For each specific achievement mentioned:\n"
            "- Extract the claimed employer ('At Company X, I did Y')\n"
            "- Verify in JSON that Y actually appears under Company X\n"
            "- Flag any misattribution\n\n"
            "STEP 4 - OVERALL QUALITY:\n"
            "- Tone appropriate for fit level?\n"
            "- Grammar and professionalism?\n"
            "- Completeness?\n\n"
            "OUTPUT FORMAT:\n"
            "Approval Status: [Approved / Approved with Revisions / Major Errors Found]\n\n"
            "1. CV Accuracy Check:\n"
            "   ✓ All achievements correctly attributed\n"
            "   OR\n"
            "   ✗ Found misattributions: [list specific errors]\n\n"
            "2. Cover Letter Accuracy Check:\n"
            "   [Similar format]\n\n"
            "3. Quality Assessment:\n"
            "   [Tone, grammar, persuasiveness]\n\n"
            "4. Revision Notes:\n"
            "   [If needed, specific corrections required]"
        ),
        expected_output=(
            "QA report with approval status and verification that all content is "
            "accurately attributed using JSON as ground truth"
        ),
        agent=quality_assurance_agent,
        context=[cv_revision_task, cover_letter_task, fit_assessment_task, cv_extraction_task],
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

def validate_outputs(revised_cv: str, cover_letter: str, cv_extraction: str) -> dict:
    """
    Validate CV and cover letter for potential misattribution errors.
    Uses general heuristics applicable to any CV.
    Returns dict with warnings if issues found.
    """
    warnings = {
        'cv_warnings': [],
        'cover_letter_warnings': []
    }
    
    if not cv_extraction:
        return warnings
    
    # Generic approach: Look for quantified achievements (numbers, $, %) 
    # and check if they might be under unexpected employers
    
    import re
    
    # Extract all employer/organization names from CV extraction
    employer_pattern = r'(?:Organization|Employer|Company):\s*([^\n]+)'
    employers = re.findall(employer_pattern, cv_extraction, re.IGNORECASE)
    
    # Look for quantified achievements in revised CV
    # Pattern: dollar amounts, percentages, large numbers with M/K/B
    achievement_patterns = [
        r'\$[\d,]+\s*(?:million|M|billion|B|thousand|K)',
        r'\d+(?:,\d{3})*\s*(?:million|M|billion|B|thousand|K)',
        r'\d+%',
        r'\d+\+?\s*(?:people|users|customers|clients|employees|team members)'
    ]
    
    # Check if major quantified achievements appear in unexpected places
    cv_sections = revised_cv.split('###')
    for section in cv_sections:
        lines = section.split('\n')
        section_header = lines[0] if lines else ''
        
        # Look for achievements in this section
        for pattern in achievement_patterns:
            matches = re.findall(pattern, section, re.IGNORECASE)
            if matches and len(matches) >= 2:
                # Multiple significant achievements in one section
                # This might indicate consolidation from multiple roles
                if len(section.split('-')) > 8:  # More than 8 bullet points
                    warnings['cv_warnings'].append(
                        f"⚠️ Section contains many achievements. Please verify all items "
                        f"belong to the employer listed in this section."
                    )
                    break
    
    # Check cover letter for specific achievement claims
    # Look for patterns like "At [Company], I [quantified achievement]"
    cl_lines = cover_letter.split('.')
    for line in cl_lines:
        # Check if line contains both a company reference and quantified achievement
        has_company_ref = any(word in line.lower() for word in ['at ', 'with ', 'for '])
        has_quantified = any(re.search(pattern, line, re.IGNORECASE) for pattern in achievement_patterns)
        
        if has_company_ref and has_quantified:
            warnings['cover_letter_warnings'].append(
                f"⚠️ Achievement with specific numbers mentioned. Please verify it's "
                f"attributed to the correct employer from your original CV."
            )
            break
    
    # General reminder if any quantified claims exist
    if re.search(r'\$[\d,]+|[\d,]+\s*(?:million|M)', cover_letter):
        if not warnings['cover_letter_warnings']:
            warnings['cover_letter_warnings'].append(
                "💡 Tip: Double-check that any specific achievements or numbers mentioned "
                "are attributed to the correct employer."
            )
    
    return warnings

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
    - Generate adaptive, honest application materials (CV and cover letter) based on fit level
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
            "Upload your CV",
            type=['pdf', 'docx', 'doc', 'txt', 'md'],
            help="Upload your CV in PDF, Word, or text format"
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
                # Extract text from CV (supports multiple formats)
                status_text = st.empty()
                status_text.text("Extracting text from your CV...")
                
                # Save file temporarily first
                resume_path = save_uploaded_file(resume_file)
                
                # Determine file type
                file_extension = resume_file.name.split('.')[-1].lower()
                
                # Extract text based on file type
                resume_text = extract_text_from_file(resume_path, file_extension)
                
                if not resume_text:
                    st.error(f"❌ Failed to extract text from {file_extension.upper()}. Please try a different file or format.")
                    return
                
                # Save extracted text as a file
                cv_text_path = save_text_as_file(resume_text, "cv_text.txt")
                
                # Save job description
                if job_input_method == "Paste Text":
                    job_desc_path = save_text_as_markdown(job_description, "job_description.md")
                else:
                    job_desc_path = save_uploaded_file(job_file, suffix='.md')
                
                # Create progress indicators
                progress_bar = st.progress(0)
                
                status_text.text("Step 1/3: Parsing CV into structured format...")
                progress_bar.progress(33)
                
                # Create and run crew
                crew = create_agents_and_tasks(cv_text_path, job_desc_path)
                
                status_text.text("Step 2/3: Assessing candidate fit...")
                progress_bar.progress(66)
                
                inputs = {
                    'job_posting': job_desc_path,
                    'cv_text': cv_text_path
                }
                
                result = crew.kickoff(inputs=inputs)
                
                status_text.text("Step 3/3: Generating application materials...")
                progress_bar.progress(100)
                
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
                
                # Extract individual task outputs
                task_outputs = {}
                for task in crew.tasks:
                    if hasattr(task, 'output') and task.output:
                        task_outputs[task.agent.role] = str(task.output)
                
                # Get the fit assessment specifically
                fit_assessment_output = task_outputs.get('Recruitment Assessment Expert', str(result))
                cv_extraction_output = task_outputs.get('CV Parser and Analyzer', '')
                
                # Validate outputs for misattributions (now that revised_cv and cover_letter are defined)
                validation_warnings = validate_outputs(revised_cv, cover_letter, cv_extraction_output)
                
                # Store results with timestamp
                from datetime import datetime
                st.session_state.results = {
                    'assessment': fit_assessment_output,
                    'revised_cv': revised_cv,
                    'cover_letter': cover_letter,
                    'qa_report': str(result),  # Store QA separately
                    'all_outputs': task_outputs,
                    'validation_warnings': validation_warnings
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
            
            # Show validation warnings if any
            if st.session_state.results.get('validation_warnings', {}).get('cv_warnings'):
                for warning in st.session_state.results['validation_warnings']['cv_warnings']:
                    st.warning(warning)
            
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
            
            # Show validation warnings if any
            if st.session_state.results.get('validation_warnings', {}).get('cover_letter_warnings'):
                for warning in st.session_state.results['validation_warnings']['cover_letter_warnings']:
                    st.warning(warning)
            
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
