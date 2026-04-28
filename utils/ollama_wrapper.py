"""
OLLAMA Wrapper for Sinhala Answer Evaluation
Handles local model inference with Sinhala support
"""
import json
import requests
import logging
from typing import Optional, Dict, Any
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class OLLAMAWrapper:
    """Wrapper for OLLAMA local model inference"""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        """
        Initialize OLLAMA wrapper
        
        Args:
            base_url: OLLAMA server base URL
            model: Model name to use (e.g., 'gemma4')
        """
        self.base_url = base_url
        self.model = model
        self.endpoint = f"{base_url}/api/generate"
        self.verify_connection()
    
    def verify_connection(self) -> bool:
        """Verify connection to OLLAMA server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ Connected to OLLAMA at {self.base_url}")
                return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to OLLAMA: {e}")
            raise RuntimeError(f"OLLAMA server not reachable at {self.base_url}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        format: str = ""
    ) -> str:
        """
        Generate text using OLLAMA model
        
        Args:
            prompt: Input prompt (can be in Sinhala or English)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt for context
            stream: Whether to stream response
            format: Output format (e.g., 'json')
            
        Returns:
            Generated text response
        """
        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": stream
            }
            if format:
                payload["format"] = format
            
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=OLLAMA_TIMEOUT,
                stream=stream
            )
            
            if response.status_code != 200:
                logger.error(f"OLLAMA error: {response.text}")
                raise RuntimeError(f"OLLAMA returned status {response.status_code}")
            
            if stream:
                result = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            result += data["response"]
                return result.strip()
            else:
                data = response.json()
                return data.get("response", "").strip()
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def score_answer(
        self,
        question: str,
        answer: str,
        marking_criteria: list,
        retrieved_context: Optional[str] = None,
        ontology_concepts: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Use OLLAMA to score an answer based on marking criteria
        
        Args:
            question: The question asked
            answer: The student's answer
            marking_criteria: List of marking criteria with marks allocation
            retrieved_context: Optional context from RAG retrieval
            ontology_concepts: Optional relevant ontology concepts
            
        Returns:
            Dictionary with scores and reasoning per criterion
        """
        criteria_text = "\n".join([
            f"- {c['criterion']}: {c['marks']} marks ({c['description']})"
            for c in marking_criteria
        ])
        
        context_text = ""
        if retrieved_context:
            context_text = f"\n\nRetrieved Context:\n{retrieved_context}"
        
        concepts_text = ""
        if ontology_concepts:
            concepts_text = f"\n\nRelevant Historical Concepts:\n" + ", ".join(ontology_concepts)
        
        prompt = f"""You are an expert Sri Lankan history examiner. Score this student answer.

                    Question: {question}

                    Student Answer: {answer}                   

                    Marking Criteria:
                    {criteria_text}  

                    Additional Context:
                    {context_text}

                    Relevant Historical Concepts:
                    {concepts_text}                                    

                    Provide a JSON response with:
                    1. total_score: (out of {sum(c['marks'] for c in marking_criteria)})
                    2. Return ONLY valid JSON.
                    3. Do not include explanations, markdown, or extra text.                 
                    
                    
                    Response format (JSON only):
                                        {{
                                    "criteria_scores": [
                                        {{
                                        "criterion": "Criterion name exactly as provided",
                                        "max_marks": 0,
                                        "score_awarded": 0,
                                        "justification": "Short simple explanation in sinhala"
                                        }}
                                    ],
                                    "total_score": 0
                                    }}

                    Rules:
                    - "criteria_scores" must contain exactly 4 items.
                    - "criterion" must exactly match each provided criterion name.
                    - "max_marks" must exactly match allocated marks.
                    - "score_awarded" must be between 0 and max_marks.
                    - "justification" must be short and criterion-specific.
                    - "total_score" must equal sum of all score_awarded.
                    """
        
        system_prompt = "You are a history grading expert. Respond only with valid JSON."
        
        try:
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temp for consistent scoring
                max_tokens=1500,
                format="json"
            )
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                logger.warning("Could not extract JSON from OLLAMA response")
                return {
                    "error": "Failed to parse OLLAMA response",
                    "raw_response": response
                }
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"error": f"JSON parsing error: {e}"}
        except Exception as e:
            logger.error(f"Error scoring answer: {e}")
            raise
    
    def generate_explanation(
        self,
        scores_data: Dict[str, Any],
        answer_quality: str
    ) -> str:
        """
        Generate detailed explanation of scoring
        
        Args:
            scores_data: Dictionary with scores and reasoning
            answer_quality: Brief description of answer quality
            
        Returns:
            Formatted explanation string
        """
        prompt = f"""Generate a clear, student-friendly explanation of their score breakdown:

                    Score Data: {json.dumps(scores_data, ensure_ascii=False, indent=2)}
                    Answer Quality: {answer_quality}

                    Provide a concise, encouraging explanation that:
                    1. Summarizes their performance
                    2. Highlights what they did well
                    3. Suggests areas for improvement
                    4. Is appropriate for a student in Sinhala or English context
                    """
        
        system_prompt = "You are an educational feedback specialist. Provide constructive, clear feedback."
        
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800
        )


def initialize_ollama() -> OLLAMAWrapper:
    """Initialize and return OLLAMA wrapper"""
    try:
        return OLLAMAWrapper()
    except RuntimeError as e:
        logger.error(f"Failed to initialize OLLAMA: {e}")
        raise
