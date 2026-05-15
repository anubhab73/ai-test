from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_EVAL_MODEL = os.getenv("GROQ_EVAL_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
FALLBACK_EVAL_MODELS = [
    DEFAULT_EVAL_MODEL,
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]


def get_evaluation_model_candidates():
    """
    Return Groq evaluator models in fallback order without duplicates.
    """
    seen = set()
    models = []

    for model_name in FALLBACK_EVAL_MODELS:
        if model_name and model_name not in seen:
            seen.add(model_name)
            models.append(model_name)

    return models


def create_evaluation_chain(model_name: str | None = None):
    """
    Create an enhanced chain for evaluating student answers using Groq.
    """
    selected_model = model_name or DEFAULT_EVAL_MODEL

    llm = ChatGroq(
        model=selected_model,
        temperature=0.1,
    )
    
    prompt = PromptTemplate(
        input_variables=["model_answer", "student_answer"],
        template="""
        You are an expert educational evaluator. Evaluate the student's answer against the model answer.
        
        Model Answer (Correct Answer):
        {model_answer}
        
        Student Answer:
        {student_answer}
        
        EVALUATION GUIDELINES:
        
        For Multiple Choice Questions (MCQs):
        - If student selects the correct option: 8-10 points
        - If student selects wrong option: 0-2 points
        - If no answer or unclear: 0 points
        
        For Short/Long Answer Questions:
        SCORE 9-10: Excellent - Answer is completely correct, comprehensive, and well-explained
        SCORE 7-8: Good - Mostly correct with minor omissions or slight inaccuracies  
        SCORE 5-6: Average - Partially correct but missing key elements or contains some errors
        SCORE 3-4: Below Average - Shows some understanding but major errors or omissions
        SCORE 0-2: Poor - Incorrect, irrelevant, or no meaningful answer
        
        Special Cases:
        - If student answer is empty, "[No answer provided]", or clearly irrelevant: score = 0
        - If student answer shows effort but is completely wrong: score = 1-2
        - If student answer has some correct elements: minimum score = 3
        
        Provide evaluation with:
        1. Score (0-10) as a number - BE CONSISTENT AND FAIR
        2. Detailed constructive feedback explaining the score
        3. Specific weak areas or concepts the student needs to improve
        
        Output ONLY valid JSON with this exact structure:
        {{
            "score": 7.5,
            "feedback": "Your answer demonstrates good understanding of the basic concept. You correctly identified X and Y. However, you missed discussing Z, which is a key component. The explanation of A could be more detailed. Overall, a solid attempt with room for improvement in completeness.",
            "weak_areas": ["Concept Z not covered", "Insufficient detail on A", "Missing connection between X and Y"]
        }}
        
        Be fair, consistent, and provide actionable feedback that helps the student improve.
        """
    )
    
    base_parser = JsonOutputParser()
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    chain = prompt | llm | fixing_parser
    return chain
