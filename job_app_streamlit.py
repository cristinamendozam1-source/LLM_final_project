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
            "Optimize CV presentation while maintaining 100% factual accuracy:\n\n"
            "⚠️ CRITICAL RULES - VIOLATIONS ARE UNACCEPTABLE:\n"
            "1. NEVER EVER move an achievement from one employer to another\n"
            "2. NEVER EVER combine information from different positions\n"
            "3. NEVER EVER add achievements that aren't in the original CV\n"
            "4. Each bullet point MUST stay under the EXACT employer where it originally appeared\n"
            "5. If unsure which employer an achievement belongs to, DO NOT include it\n\n"
            "YOUR ONLY ALLOWED CHANGES:\n"
            "- Reorder bullet points WITHIN each position (not across positions)\n"
            "- Adjust verb choice and phrasing WITHOUT changing facts\n"
            "- Add brief contextual phrases that don't change the core statement\n"
            "- Strengthen professional summary to emphasize relevant skills\n\n"
            "WORKFLOW:\n"
            "1. Read the COMPLETE CV extraction - note which bullets are under which employer\n"
            "2. For EACH employer in the CV:\n"
            "   - Copy the employer name, position title, and dates EXACTLY\n"
            "   - Include ALL bullet points that were under that employer\n"
            "   - Reorder those bullets by relevance to job\n"
            "   - Adjust phrasing if needed WITHOUT changing facts\n"
            "3. Do NOT move any content between employers\n\n"
            "If an achievement mentions 'Living Goods' or '$4 million', CHECK which employer "
            "it was actually done at in the CV extraction before including it."
        ),
        tools=[semantic_pdf, semantic_mdx],
        verbose=True,
        backstory=(
            "You are a meticulous CV editor with ZERO tolerance for factual errors. You treat "
            "misattributing an achievement to the wrong employer as a career-ending mistake. "
            "Before making ANY change, you verify it against the CV extraction. Your reputation "
            "depends on perfect accuracy. You would rather omit content than risk misattribution."
        )
    )
    
    cover_letter_writer = Agent(
        role="Adaptive Cover Letter Writer",
        goal=(
            "Write compelling, factually accurate cover letters (250-400 words).\n\n"
            "⚠️ CRITICAL ACCURACY RULES - VIOLATIONS ARE UNACCEPTABLE:\n"
            "1. BEFORE mentioning ANY achievement, verify which employer it's from\n"
            "2. NEVER say 'At Company X, I did [thing that was actually done at Company Y]'\n"
            "3. For any quantified achievement or major project, state the CORRECT employer\n"
            "4. When in doubt about attribution, don't mention the specific achievement\n\n"
            "VERIFICATION PROCESS:\n"
            "Before writing, review the CV extraction and make a reference list:\n"
            "- Achievement A (with quantified results) was done at Employer X\n"
            "- Achievement B (major project) was done at Employer Y\n"
            "Then reference this list as you write.\n\n"
            "WRITING APPROACH:\n"
            "- Open with relevant background (don't need to mention specific achievements)\n"
            "- Choose 1-2 achievements to highlight - CHECK EMPLOYER FIRST\n"
            "- State: 'In my role as [Position] at [CORRECT Employer], I [achievement]'\n"
            "- Connect skills to job requirements\n"
            "- Express enthusiasm and close professionally"
        ),
        tools=[semantic_pdf, semantic_mdx],
        verbose=True,
        backstory=(
            "You write persuasive cover letters with ZERO tolerance for errors. You know that "
            "misattributing even one achievement destroys credibility. Before mentioning any "
            "specific accomplishment, you verify its employer in the CV extraction. You would "
            "rather write a more general letter than risk stating 'I did X at Company A' when "
            "X was actually done at Company B. Accuracy is non-negotiable."
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
            "Create an optimized CV maintaining 100% factual accuracy.\n\n"
            "⚠️ BEFORE YOU START - READ THIS CAREFULLY:\n"
            "The CV extraction shows which achievements belong to which employer.\n"
            "Your job is to present this information strategically WITHOUT moving content.\n"
            "Moving an achievement from Employer A to Employer B is FORBIDDEN.\n\n"
            "MANDATORY PROCESS:\n\n"
            "STEP 1 - EXTRACT THE STRUCTURE:\n"
            "From the CV Parser's output, create a list:\n"
            "- Employer 1: [Name], [Position], [Dates]\n"
            "  • Bullet 1\n"
            "  • Bullet 2\n"
            "  • Bullet 3\n"
            "- Employer 2: [Name], [Position], [Dates]\n"
            "  • Bullet 1\n"
            "  • Bullet 2\n\n"
            "STEP 2 - VERIFY BOUNDARIES:\n"
            "Review the CV extraction carefully and identify key achievements.\n"
            "For each significant achievement (quantified results, major projects, etc.):\n"
            "- Note which employer/organization it belongs to\n"
            "- Create a mental map: Achievement X belongs to Employer Y\n\n"
            "CRITICAL RULE:\n"
            "Each achievement must stay under its original employer.\n"
            "If you're unsure which employer an achievement belongs to, refer back to "
            "the CV extraction to verify.\n\n"
            "STEP 3 - OPTIMIZE WITHIN BOUNDARIES:\n"
            "For EACH employer separately:\n"
            "- Keep employer name, title, dates exactly as given\n"
            "- List ALL bullets that belong to that employer\n"
            "- Reorder bullets (most job-relevant first)\n"
            "- Adjust phrasing to emphasize skills (without changing facts)\n\n"
            "STEP 4 - WRITE THE CV:\n"
            "Format:\n"
            "# [Name]\n"
            "[Contact info]\n\n"
            "## Professional Summary\n"
            "[2-3 sentences highlighting relevant experience]\n\n"
            "## Professional Experience\n\n"
            "### [Position Title]\n"
            "[Exact Employer Name] – [Location] | [Exact Dates]\n"
            "- [Bullet 1 - most relevant]\n"
            "- [Bullet 2]\n"
            "- [All other bullets from THIS employer only]\n\n"
            "[Repeat for each employer]\n\n"
            "## Education\n"
            "[From original CV]\n\n"
            "## Skills\n"
            "[From original CV]\n\n"
            "VERIFICATION BEFORE SUBMITTING:\n"
            "- Did I keep each bullet under its original employer? YES/NO\n"
            "- Did I move any achievements between employers? YES/NO (must be NO)\n"
            "- Did I include all original content? YES/NO (must be YES)\n\n"
            "If you moved ANY content between employers, START OVER."
        ),
        expected_output=(
            "Complete CV in Markdown with:\n"
            "- All bullets under CORRECT original employer\n"
            "- Strategic ordering within each position\n"
            "- Enhanced phrasing without factual changes\n"
            "- All original content preserved\n"
            "ZERO employer-achievement misattributions"
        ),
        output_file=cv_output,
        agent=cv_strategist,
        context=[job_extraction_task, cv_extraction_task, fit_assessment_task],
        async_execution=True
    )
    
    cover_letter_task = Task(
        description=(
            "Write a 250-400 word cover letter with STRICT factual accuracy.\n\n"
            "⚠️ CRITICAL: VERIFY BEFORE WRITING\n\n"
            "STEP 1 - CREATE REFERENCE LIST:\n"
            "Review CV Parser output and list key achievements with their ACTUAL employers:\n"
            "Example format:\n"
            "- '[Quantified achievement]' → Done at: [Employer name from CV extraction]\n"
            "- '[Major project]' → Done at: [Employer name from CV extraction]\n"
            "- '[Leadership role]' → Done at: [Employer name from CV extraction]\n\n"
            "STEP 2 - SELECT WHAT TO HIGHLIGHT:\n"
            "Choose 1-2 achievements that are most relevant to the job.\n"
            "Write down which employer each one is from using your reference list.\n\n"
            "STEP 3 - WRITE WITH CORRECT ATTRIBUTION:\n"
            "Use this pattern ONLY:\n"
            "'In my role as [Position] at [VERIFIED CORRECT Employer from CV], I [achievement].'\n\n"
            "WRONG pattern (NEVER DO THIS):\n"
            "❌ 'At [Current/Most Recent Company], I [achievement that was done at previous company]'\n"
            "❌ 'In my current role, I [achievement from a different position]'\n\n"
            "CORRECT pattern:\n"
            "✅ 'In my role as [Position Title] at [Actual Employer Name], I [achievement]...'\n"
            "✅ 'As [Position] at [Correct Organization], I led [specific project]...'\n\n"
            "LETTER STRUCTURE:\n\n"
            "Paragraph 1 (Opening):\n"
            "- Express interest in the role\n"
            "- Brief background (no specific achievements here)\n"
            "- Mention total years of experience and general areas\n\n"
            "Paragraph 2 (Main content):\n"
            "- Pick 1-2 most relevant experiences from your reference list\n"
            "- State: 'In my role as [Position] at [CORRECT Employer], I [achievement]'\n"
            "- Explain how this relates to job requirements\n\n"
            "Paragraph 3 (Connection):\n"
            "- Connect skills/values to company mission\n"
            "- Show understanding of role requirements\n"
            "- Can mention additional relevant skills WITHOUT specific attributions\n\n"
            "Paragraph 4 (Close):\n"
            "- Express enthusiasm\n"
            "- Request interview/next steps\n\n"
            "TONE BY FIT LEVEL:\n"
            "- HIGH (75%+): Confident, achievement-focused\n"
            "- MEDIUM (50-74%): Balanced, honest about gaps, emphasize transferable skills\n"
            "- LOW (<50%): Humble, focus on learning potential\n\n"
            "FINAL VERIFICATION:\n"
            "Before submitting, ask yourself:\n"
            "- Did I mention any specific achievements with quantified results?\n"
            "- If yes, did I verify the correct employer for EACH one against CV extraction?\n"
            "- Am I 100% certain each achievement is attributed to its actual employer?\n"
            "If not certain, rewrite to be more general or omit the detail."
        ),
        expected_output=(
            "250-400 word cover letter with:\n"
            "- ZERO employer-achievement misattributions\n"
            "- Correct employer names for any achievements mentioned\n"
            "- Professional business letter format\n"
            "- Appropriate tone for fit level\n"
            "- Persuasive within bounds of complete accuracy"
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
                cv_extraction_output = task_outputs.get('CV Parser and Analyzer', '')
                
                # Validate outputs for misattributions
                validation_warnings = validate_outputs(revised_cv, cover_letter, cv_extraction_output)
                
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
