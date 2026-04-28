"""
Streamlit UI for Sinhala Answer Evaluator
Main application interface
"""
import streamlit as st
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PAGE_TITLE, PAGE_ICON, STREAMLIT_THEME,
    QUESTIONS_FILE, MARKING_GUIDES_FILE
)
from utils.rag import initialize_rag
from utils.ontology_handler import initialize_ontology
from utils.ollama_wrapper import initialize_ollama
from models.agents import AnswerEvaluationOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm, .st-emotion-cache-16idsys p {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main Container layout */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Primary Streamlit Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.2) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.4) !important;
    }

    /* Input text area premium feel */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #cbd5e0 !important;
        padding: 1.2rem !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2) !important;
    }

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Custom CSS Classes */
    .score-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(56, 239, 125, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    .criterion-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .criterion-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    .criterion-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .criterion-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: #2d3748;
    }
    .criterion-marks {
        background: #f0f4f8;
        color: #2b6cb0;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .progress-container {
        background-color: #edf2f7;
        border-radius: 10px;
        height: 6px;
        width: 100%;
        margin: 8px 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
    }
    .justification-text {
        font-size: 0.85rem;
        color: #4a5568;
        line-height: 1.5;
        margin-top: 8px;
    }

    .strength {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 14px 18px;
        border-left: 5px solid #28a745;
        margin: 12px 0;
        border-radius: 8px;
        color: #155724;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .strength:hover { 
        transform: scale(1.01); 
        box-shadow: 0 6px 15px rgba(40, 167, 69, 0.2);
    }

    .gap {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 14px 18px;
        border-left: 5px solid #dc3545;
        margin: 12px 0;
        border-radius: 8px;
        color: #721c24;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .gap:hover { 
        transform: scale(1.01); 
        box-shadow: 0 6px 15px rgba(220, 53, 69, 0.2);
    }

    .info-box {
        background: rgba(231, 243, 255, 0.85);
        backdrop-filter: blur(12px);
        padding: 25px;
        border-left: 6px solid #2196F3;
        margin: 25px 0;
        border-radius: 12px;
        color: #0c5460;
        box-shadow: 0 8px 20px rgba(33, 150, 243, 0.15);
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .terminal-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
        font-size: 0.9rem;
        line-height: 1.4;
        margin: 10px 0;
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)


class StreamlitLogHandler(logging.Handler):
    """Custom logging handler to display logs in Streamlit"""
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.log_text = ""

    def emit(self, record):
        msg = self.format(record)
        self.log_text += msg + "\n"
        # Display as a code block for terminal feel
        self.container.code(self.log_text, language="bash")


@st.cache_resource
def load_questions() -> List[Dict]:
    """Load questions from JSON file"""
    try:
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('questions', [])
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        st.error(f"Error loading questions: {e}")
        return []


@st.cache_resource
def load_marking_guides() -> Dict:
    """Load marking guides from JSON file"""
    try:
        with open(MARKING_GUIDES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert to dict keyed by question_id
        guides = {}
        for guide in data.get('marking_guides', []):
            guides[guide['question_id']] = guide
        return guides
    except Exception as e:
        logger.error(f"Error loading marking guides: {e}")
        st.error(f"Error loading marking guides: {e}")
        return {}


@st.cache_resource
def initialize_system(cache_buster=1):
    """Initialize all system components"""
    try:
        with st.spinner("Initializing system components..."):
            rag = initialize_rag(force_reload=True)
            ontology = initialize_ontology()
            ollama = initialize_ollama()
            orchestrator = AnswerEvaluationOrchestrator(rag, ontology, ollama)
        st.success("✅ System initialized successfully!")
        return orchestrator, rag, ontology, ollama
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        st.error(f"❌ Failed to initialize system: {e}")
        st.info("Make sure OLLAMA is running and all dependencies are installed.")
        return None, None, None, None


def render_results(evaluation_result: Dict[str, Any]):
    """Render evaluation results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="අවසන් ලකුණු",
            value=f"{evaluation_result.get('final_score', 0)}/20",
            delta=f"{evaluation_result.get('percentage', 0):.1f}%"
        )
    
    with col2:
        st.metric(
            label="ශ්‍රේණිය",
            value=get_grade_letter(evaluation_result.get('final_score', 0))
        )
    
    with col3:
        st.metric(
            label="ආවරණය",
            value=evaluation_result.get('validation', {}).get('overall_coverage', 'N/A')
        )
    
        
    # Criterion Breakdown
    st.subheader("📈 නිර්ණායක අනුව ලකුණු විස්තර")

    criterion_breakdown = evaluation_result.get("criterion_breakdown", [])

    if criterion_breakdown:
        # Display in 2 columns for compactness
        cols = st.columns(2)
        for idx, item in enumerate(criterion_breakdown):
            criterion = item.get("criterion", "N/A")
            awarded = item.get("awarded_marks", 0)
            maximum = item.get("max_marks", 0)
            justification = item.get("justification", "විස්තර නොමැත.")
            
            # Calculate percentage for progress bar
            percentage = (awarded / maximum * 100) if maximum > 0 else 0
            # Color based on performance
            bar_color = "#48bb78" if percentage >= 75 else "#ecc94b" if percentage >= 40 else "#f56565"

            with cols[idx % 2]:
                st.markdown(f"""
                    <div class="criterion-card">
                        <div class="criterion-header">
                            <span class="criterion-title">📌 {criterion}</span>
                            <span class="criterion-marks">{awarded}/{maximum}</span>
                        </div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {percentage}%; background-color: {bar_color};"></div>
                        </div>
                        <div class="justification-text">
                            <strong>විග්‍රහය:</strong> {justification}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("නිර්ණායක අනුව ලකුණු විස්තර නොමැත.")
    
    # Validation results
    st.subheader("✅ වලංගුතා විශ්ලේෂණය")
    validation = evaluation_result.get('validation', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### **ශක්තිමත් අංශ:**")
        for strength in validation.get('strengths', [])[:3]:
            st.markdown(f"<div class='strength'>✓ {strength}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### **දියුණු කළ යුතු අංශ:**")
        for gap in validation.get('gaps', [])[:3]:
            st.markdown(f"<div class='gap'>✗ {gap}</div>", unsafe_allow_html=True)
    
    # Detailed explanation
    st.subheader("📝 සවිස්තරාත්මක ප්‍රතිපෝෂණ")
    explanation = evaluation_result.get('explanation', 'ප්‍රතිපෝෂණයක් නොමැත.')
    st.info(explanation)
    
    # Context and concepts used
    st.markdown("#### 🔍 භාවිතා කරන ලද සන්දර්භය (retrieved context from documents)")
    context_list = evaluation_result.get('context_list', [])
    if context_list:
        for item in context_list:
            with st.container():
                st.info(f"[{item['id']}] {item['source_info']} (relevance: {item['score']:.2f})")
                st.write(item['content'])
                st.markdown("---")
    else:
        st.write("අදාළ සන්දර්භයක් හමු නොවීය.")
    
    st.markdown("#### 🔍 භාවිතා කරන ලද සංකල්ප (retrieved concepts from ontology)")
    concepts = evaluation_result.get('concepts', [])
    if concepts:
        st.write(f"**හඳුනාගත් ප්‍රධාන සංකල්ප (concepts from ontology):** {', '.join(concepts)}")


def get_grade_letter(score: int) -> str:
    """Convert numeric score to letter grade"""
    if score >= 18:
        return "A+ (විශිෂ්ටයි)"
    elif score >= 16:
        return "A (ඉතා හොඳයි)"
    elif score >= 14:
        return "B+ (හොඳයි)"
    elif score >= 12:
        return "B (සතුටුදායකයි)"
    elif score >= 10:
        return "C (සාමාන්‍යයි)"
    elif score >= 8:
        return "D (සමත්)"
    else:
        return "F (අසමත්)"

def clear_form():
    """Clear the answer text area and hide results"""
    st.session_state.show_results = False
    st.session_state.answer_text = ""


def main():
    """Main application"""
    # Display logo and title
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(PAGE_ICON, width=80)
    with col2:
        st.title(PAGE_TITLE)
    st.markdown("### බුද්ධිමත් සිංහල පිළිතුරු ඇගයුම්කරු")
    st.markdown("**මාතෘකාව:** යටත්විජිත ශ්‍රී ලංකාව (පෘතුගීසි → ලන්දේසි → බ්‍රිතාන්‍ය)")
    
    # Info box
    st.markdown("""
    <div class='info-box'>
    <strong>📌 උපදෙස් :</strong><br>
    1. යටත්විජිත ශ්‍රී ලංකා ඉතිහාසය පිළිබඳ ප්‍රශ්නයක් තෝරන්න<br>
    2. ඔබගේ පිළිතුර සිංහලෙන් ඇතුළත් කරන්න<br>    
    3. සවිස්තරාත්මක ප්‍රතිපෝෂණ සහ ලකුණු ලබා ගන්න
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    orchestrator, rag, ontology, ollama = initialize_system()
    
    if orchestrator is None:
        st.error("පද්ධතිය ආරම්භ කිරීමට අපොහොසත් විය. OLLAMA සම්බන්ධතාවය පරීක්ෂා කරන්න.")
        return
    
    # Load questions and marking guides
    questions = load_questions()
    marking_guides = load_marking_guides()
    
    if not questions:
        st.error("ප්‍රශ්න පූරණය කළ නොහැක!")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ සැකසුම්")
        
        # Question selection
        question_options = {
            f"Q{q['id']}: {q['focus_area']}": q for q in questions
        }
        selected_q = st.selectbox(
            "ප්‍රශ්නයක් තෝරන්න:",
            options=question_options.keys(),
            help="පිළිතුරු දීමට අවශ්‍ය ඉතිහාස ප්‍රශ්නය තෝරන්න"
        )
        
        current_question = question_options[selected_q]
        question_id = current_question['id']
        
        st.markdown("---")
        st.subheader("📌 ප්‍රශ්නයේ විස්තර")
        st.write(f"**අවධානය යොමු කළ යුතු අංශය:** {current_question['focus_area']}")
        st.write(f"**ලකුණු:** 20")
        st.write(f"**භාෂාව:** සිංහල")      
        
    
    # Main content area
    st.subheader(f"ප්‍රශ්නය - {question_id} ")
    
    # Display question in both Sinhala and English
    col1, col2 = st.columns(2)
    with col1:
        st.write("**සිංහල:**")
        st.write(current_question['question'])
    with col2:
        st.write("**ඉංග්‍රීසි පරිවර්තනය:**")
        st.write(current_question['question_english'])

    st.markdown("---")
    st.subheader("📋 ලකුණු දීමේ නිර්ණායක")
    guide = marking_guides.get(question_id, {})
    for criterion in guide.get('criteria', []):
        st.write(f"• {criterion['criterion']} ({criterion['marks']} marks)")
    
    st.markdown("---")

    st.subheader("📝 ඔබගේ පිළිතුර (සිංහලෙන්):")    
    # Answer input
    answer_input = st.text_area(
        "පිළිතුර",
        height=200,
        placeholder="ඔබගේ පිළිතුර මෙහි සිංහලෙන් ටයිප් කරන්න...",
        help="ප්‍රශ්නය සඳහා ඔබගේ සම්පූර්ණ පිළිතුර ඇතුළත් කරන්න",
        key="answer_text",
        label_visibility="collapsed"
    )
    
    # Evaluation button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("පිළිතුර ඇගයීම", use_container_width=True):
            if not answer_input.strip():
                st.error("කරුණාකර ඇගයීමට පෙර පිළිතුරක් ඇතුළත් කරන්න!")
            else:
                # Get marking criteria for this question
                guide = marking_guides.get(question_id, {})
                criteria = guide.get('criteria', [])
                
                if not criteria:
                    st.error("මෙම ප්‍රශ්නය සඳහා ලකුණු දීමේ නිර්ණායක හමු නොවීය!")
                else:
                    # Create a placeholder for the terminal output
                    log_placeholder = st.empty()
                    
                    # Run evaluation
                    with st.spinner("ඔබගේ පිළිතුර ඇගයීම සිදුකරමින් ... "):
                        with log_placeholder.container():
                            st.markdown("<h4 style='color: #667eea;'>🛠️ පද්ධති ක්‍රියාකාරිත්වය (Real-time Processing)</h4>", unsafe_allow_html=True)
                            terminal_placeholder = st.empty()
                        
                        # Setup log capture for models.agents
                        handler = StreamlitLogHandler(terminal_placeholder)
                        handler.setFormatter(logging.Formatter('%(message)s'))
                        agent_logger = logging.getLogger("models.agents")
                        # Add a flag to prevent duplicate handlers if the button is clicked rapidly
                        if not any(isinstance(h, StreamlitLogHandler) for h in agent_logger.handlers):
                            agent_logger.addHandler(handler)
                        
                        try:
                            evaluation_result = orchestrator.evaluate_answer(
                                question=current_question['question'],
                                answer=answer_input,
                                marking_criteria=criteria
                            )
                            
                            # Store result in session state
                            st.session_state.last_evaluation = evaluation_result
                            st.session_state.show_results = True
                            
                        except Exception as e:
                            st.error(f"ඇගයීමේදී දෝෂයක් මතු විය: {e}")
                            logger.error(f"Evaluation error: {e}", exc_info=True)
                        finally:
                            # Clean up: remove handler and clear the terminal output
                            agent_logger.removeHandler(handler)
                            log_placeholder.empty()
    
    with col2:
        st.button("මකා දමන්න", use_container_width=True, on_click=clear_form)
    
    with col3:
        if st.button("උදව්", use_container_width=True):
            st.info("""
            **පිළිතුරු සැපයීම සඳහා උපදෙස්:**
            - පිළිතුර පැහැදිලි සිංහලෙන් සපයන්න
            - සියලුම නිර්ණායක ආවරණය වන සේ පිළිතුරු දෙන්න
            - ඓතිහාසික කරුණු සහ උදාහරණ භාවිතා කරන්න
            - ඉදිරිපත් කිරීමට පෙර නැවත කියවා බලන්න
            """)
    
    # Display results if available
    if st.session_state.get('show_results', False):
        st.markdown("---")
        st.header("✨ ඇගයුම් ප්‍රතිඵල")
        
        evaluation_result = st.session_state.get('last_evaluation', {})
        
        if 'error' in evaluation_result:
            st.error(f"ඇගයුම් දෝෂය: {evaluation_result['error']}")
        else:
            render_results(evaluation_result)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>සිංහල පිළිතුරු ඇගයුම්කරු v1.0 | OLLAMA (gemma4) මගින් බලගන්වා ඇත | RAG + Ontology Architecture</p>
    <p>© 2026 PHDB Nayanakantha</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize session state
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = {}
    if 'answer_text' not in st.session_state:
        st.session_state.answer_text = ""
    
    main()
