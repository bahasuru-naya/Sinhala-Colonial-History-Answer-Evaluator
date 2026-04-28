"""
Agents for Sinhala Answer Evaluation
Implements specialized agents: Retriever, Validator, Scorer, Explainer
"""
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from utils.rag import RAGPipeline
from utils.ontology_handler import OntologyHandler
from utils.ollama_wrapper import OLLAMAWrapper

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State passed between agents"""
    question: str
    answer: str
    marking_criteria: List[Dict]
    retrieved_context: Optional[str] = None
    ontology_concepts: Optional[List[str]] = None
    validation_results: Optional[Dict] = None
    scoring_results: Optional[Dict] = None
    explanation: Optional[str] = None
    final_score: Optional[int] = None

    criterion_breakdown: Optional[List[Dict[str, Any]]] = None
    retrieved_context_list: Optional[List[Dict[str, Any]]] = None


class RetrieverAgent:
    """Retrieves relevant context from knowledge base"""    

    def __init__(self, rag_pipeline: RAGPipeline, ontology: OntologyHandler):
        self.rag = rag_pipeline
        self.ontology = ontology

    def execute(self, state: AgentState) -> AgentState:
        logger.info("🔍 [Retriever] Starting retrieval...")

        try:
            retrieved_docs = self._retrieve_with_fallback(state.question, k=5)

            if retrieved_docs:
                formatted_list = self.rag.format_context(retrieved_docs)
                state.retrieved_context_list = formatted_list
                
                # Join into a string for the LLM prompt
                state.retrieved_context = "\n---\n".join([
                    f"[{d['id']}] {d['source_info']} (relevance: {d['score']:.2f})\n{d['content']}"
                    for d in formatted_list
                ])
                
                logger.info(
                    f"✓ [Retriever] Retrieved {len(retrieved_docs)} documents "
                    f"(top score: {retrieved_docs[0][1]:.3f})"
                )
            else:
                state.retrieved_context = "No matching context found in knowledge base."
                state.retrieved_context_list = []
                logger.warning(
                    "[Retriever] No documents retrieved even at threshold=0.0 — "
                    "check that load_knowledge_base() ran successfully and "
                    "collection.count() > 0."
                )

            # Extract ontology concepts
            combined_text = f"{state.question} {state.answer or ''}"
            concepts = self.ontology.extract_key_concepts(combined_text)
            state.ontology_concepts = concepts if concepts else []

            logger.info(f"✓ [Retriever] Found {len(state.ontology_concepts)} key concepts")
            return state

        except Exception as e:
            logger.error(f"✗ [Retriever] Error: {e}", exc_info=True)
            state.retrieved_context = f"Error during retrieval: {e}"
            return state

    def _retrieve_with_fallback(self, query: str, k: int):
        """
        Try retrieval at the configured SIMILARITY_THRESHOLD first.
        If nothing comes back, retry with progressively lower thresholds
        so a high config value never silently swallows all results.
        """
        # --- diagnostic: log collection size so we can spot empty-KB issues ---
        if self.rag.using_chroma and self.rag.collection is not None:
            count = self.rag.collection.count()
            logger.info(f"[Retriever] ChromaDB collection has {count} chunks")
            if count == 0:
                logger.warning(
                    "[Retriever] Collection is empty — call initialize_rag(force_reload=True)"
                )
                return []
        elif hasattr(self.rag, 'fallback_docs'):
            logger.info(f"[Retriever] Fallback store has {len(self.rag.fallback_docs)} chunks")

        # First attempt at the configured threshold
        docs = self.rag.retrieve(query=query, k=k)
        if docs:
            return docs

        else:
            logger.warning( f"[Retriever] No results")            
            return []   


class ValidatorAgent:
    """Validates answer coverage against marking criteria"""
    
    def __init__(self, ollama: OLLAMAWrapper, ontology: OntologyHandler):
        """
        Initialize validator agent
        
        Args:
            ollama: OLLAMA wrapper instance
            ontology: Ontology handler instance
        """
        self.ollama = ollama
        self.ontology = ontology
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Validate answer against marking criteria
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with validation results
        """
        logger.info("✓ [Validator] Starting validation...")
        
        try:
            criteria_text = "\n".join([
                f"- {c['criterion']}: {c['marks']} marks"
                for c in state.marking_criteria
            ])
            
            prompt = f"""ඔබ සිංහල ඉතිහාස පිළිතුරු ඇගයීම සඳහා JSON පමණක් ලබාදෙන validator agent කෙනෙකි.

                        ඔබගේ කාර්යය:
                        ශිෂ්‍ය පිළිතුර ලකුණු දීමේ නිර්ණායක අනුව විශ්ලේෂණය කර, 
                        1.strengths
                        2.gaps
                        3.overall_coverage
                        යන ත්‍රිත්වය තුළ පිළිතුරේ ගුණාත්මක බව JSON ආකෘතියෙන් ලබා දීමයි.  
                        
                        Question:
                        {state.question}

                        Answer:
                        {state.answer}

                        Marking Criteria:
                        {criteria_text}

                        Additional Context:
                        {state.retrieved_context}

                        Relevant Historical Concepts:
                        {state.ontology_concepts}

                        අත්‍යවශ්‍ය නීති:
                        - JSON පමණක් ලබාදෙන්න
                        - Markdown නොදෙන්න                        
                        - JSON ට පිටතින් කිසිම වචනයක් නොදෙන්න
                        - strengths, gaps අනිවාර්යයෙන් සිංහලෙන් ලියන්න
                        - overall_coverage අගය අවම (low), මධ්‍යම (medium), ඉහළ (high) තුනෙන් එකක් පමණක් විය යුතුය
                        - strengths සහ gaps arrays විය යුතුය
                        - strengths සඳහා අයිතම 2-4 අතර
                        - gaps සඳහා අයිතම 2-4 අතර

                        නිවැරදි JSON ආකෘතිය:

                        {{
                        "strengths": [
                            "සිංහල වාක්‍ය 1",
                            "සිංහල වාක්‍ය 2"
                        ],
                        "gaps": [
                            "සිංහල වාක්‍ය 1",
                            "සිංහල වාක්‍ය 2"
                        ],
                        "overall_coverage": "අවම (low)"                        
                        }}
                        """
            
            response = self.ollama.generate(
                prompt=prompt,
                system_prompt="You are an expert validator. Provide structured JSON assessment of answer quality.",
                temperature=0.3,
                max_tokens=800,
                format="json"
            )

            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                results = json.loads(json_str)
                # Ensure lists
                if isinstance(results.get('strengths'), str):
                    results['strengths'] = [results['strengths']]
                if isinstance(results.get('gaps'), str):
                    results['gaps'] = [results['gaps']]
                state.validation_results = results
            else:
                state.validation_results = {"raw_response": response}
            
            logger.info(f"✓ [Validator] Validation complete")
            return state
        
        except Exception as e:
            logger.error(f"✗ [Validator] Error: {e}")
            state.validation_results = {"error": str(e)}
            return state


class ScorerAgent:
    """Scores answer per marking criterion"""
    
    def __init__(self, ollama: OLLAMAWrapper):
        """
        Initialize scorer agent
        
        Args:
            ollama: OLLAMA wrapper instance
        """
        self.ollama = ollama
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Score answer per criterion
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with scoring results
        """
        logger.info("🎯 [Scorer] Starting scoring...")
        
        try:                               
            
            response = self.ollama.score_answer(
                question=state.question,
                answer=state.answer,
                marking_criteria=state.marking_criteria,
                retrieved_context=state.retrieved_context,
                ontology_concepts=state.ontology_concepts
            )
            
            state.scoring_results = response
            
            # Build structured criterion breakdown
            breakdown = []
            total_score = 0

            criteria_scores = response.get("criteria_scores", [])

            for item in criteria_scores:
                criterion_name = item.get("criterion", "N/A")
                max_marks = item.get("max_marks", 0)
                awarded_marks = item.get("score_awarded", 0)
                justification = item.get(
                    "justification",
                    "මෙම නිර්ණායකය සඳහා විස්තරාත්මක සාධාරණීකරණයක් ලබා දී නොමැත."
                )

                # Safety clamp
                awarded_marks = max(0, min(awarded_marks, max_marks))

                breakdown.append({
                    "criterion": criterion_name,
                    "awarded_marks": awarded_marks,
                    "max_marks": max_marks,
                    "justification": justification
                })

                total_score += awarded_marks

            # Store normalized breakdown
            state.criterion_breakdown = breakdown            
            
            # Extract total score
            if 'total_score' in response:
                state.final_score = min(response['total_score'], 20)  # Cap at 20
            else:
                state.final_score = min(total_score, 20)
            
            logger.info(f"✓ [Scorer] Scoring complete - Score: {state.final_score}/20")
            return state
        
        except Exception as e:
            logger.error(f"✗ [Scorer] Error: {e}")
            state.scoring_results = {"error": str(e)}
            state.final_score = 0
            return state


class ExplainerAgent:
    """Generates human-readable explanation of scoring"""
    
    def __init__(self, ollama: OLLAMAWrapper):
        """
        Initialize explainer agent
        
        Args:
            ollama: OLLAMA wrapper instance
        """
        self.ollama = ollama
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Generate detailed explanation of scoring
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with explanation
        """
        logger.info("📝 [Explainer] Starting explanation generation...")
        
        try:
            # Build explanation context
            validation_summary = ""
            if state.validation_results:
                if 'gaps' in state.validation_results:
                    validation_summary += f"\nGaps identified: {', '.join(state.validation_results['gaps'][:3])}"
                if 'strengths' in state.validation_results:
                    validation_summary += f"\nStrengths: {', '.join(state.validation_results['strengths'][:3])}"
            
            scoring_summary = ""
            if state.scoring_results:
                scoring_summary = state.scoring_results
            
            context_citations = ""
            if state.retrieved_context and state.retrieved_context != "No relevant context found.":
                context_citations = "\n\n[Based on retrieved historical context]"
            
            if state.ontology_concepts:
                context_citations += f"\n[Based on relevant historical concepts from ontology]"
            
            prompt = f"""Create a clear, encouraging feedback message for a student about their answer.

                        Final Score: {state.final_score}/20

                        Question: {state.question}

                        Answer Quality Summary: {validation_summary}

                        Scoring Details: {scoring_summary}

                        Generate a feedback message that:
                        1. States their final score clearly
                        2. Explains what they did well
                        3. Identifies specific areas for improvement
                        4. Provides encouragement
                        5. Is written in a respectful, educational tone

                        CRITICAL INSTRUCTION: The ENTIRE feedback message MUST be written in SINHALA LANGUAGE. Do not use English.
                        """
            
            explanation = self.ollama.generate(
                prompt=prompt,
                system_prompt="You are an educational feedback specialist. Provide constructive, encouraging feedback entirely in Sinhala language.",
                temperature=0.7,
                max_tokens=600
            )
            
            state.explanation = explanation + context_citations
            
            logger.info("✓ [Explainer] Explanation generated")
            return state
        
        except Exception as e:
            logger.error(f"✗ [Explainer] Error: {e}")
            state.explanation = f"Error generating explanation: {e}"
            return state


class AnswerEvaluationOrchestrator:
    """Orchestrates all agents in sequence"""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        ontology: OntologyHandler,
        ollama: OLLAMAWrapper
    ):
        """
        Initialize orchestrator with all agents
        
        Args:
            rag_pipeline: RAG pipeline instance
            ontology: Ontology handler instance
            ollama: OLLAMA wrapper instance
        """
        self.retriever = RetrieverAgent(rag_pipeline, ontology)
        self.validator = ValidatorAgent(ollama, ontology)
        self.scorer = ScorerAgent(ollama)
        self.explainer = ExplainerAgent(ollama)
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        marking_criteria: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate a student answer end-to-end
        
        Args:
            question: The question asked
            answer: The student's answer
            marking_criteria: List of marking criteria with marks allocation
            
        Returns:
            Complete evaluation result dictionary
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting Answer Evaluation Pipeline")
        logger.info("=" * 60)
        
        # Initialize state
        state = AgentState(
            question=question,
            answer=answer,
            marking_criteria=marking_criteria
        )
        
        # Execute agents in sequence
        try:
            # 1. Retriever
            state = self.retriever.execute(state)
            
            # 2. Validator
            state = self.validator.execute(state)
            
            # 3. Scorer
            state = self.scorer.execute(state)
            
            # 4. Explainer
            state = self.explainer.execute(state)
            
            logger.info("=" * 60)
            logger.info("✓ Evaluation Complete")
            logger.info("=" * 60)
            
            # Format result
            return {
                "final_score": state.final_score,
                "total_marks": 20,
                "percentage": (state.final_score / 20 * 100) if state.final_score else 0,
                "validation": state.validation_results,
                "scoring": state.scoring_results,
                "explanation": state.explanation,
                "context": state.retrieved_context,
                "context_list": state.retrieved_context_list,
                "concepts": state.ontology_concepts,
                "criterion_breakdown": state.criterion_breakdown
            }
        
        except Exception as e:
            logger.error(f"✗ Pipeline execution failed: {e}")
            return {
                "error": str(e),
                "final_score": 0,
                "explanation": f"Error during evaluation: {e}"
            }
