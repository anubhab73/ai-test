from utils.compare_utils import create_evaluation_chain, get_evaluation_model_candidates
from dotenv import load_dotenv
import json
import re

load_dotenv()

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "with"
}


def _normalize_choice(answer: str) -> str:
    """
    Normalize MCQ answers like "A", "A.", or "A) Option text" to the choice letter.
    """
    answer = (answer or "").strip().upper()
    if not answer or answer == "[NO ANSWER PROVIDED]":
        return ""

    match = re.search(r"\b([A-D])\b", answer)
    if match:
        return match.group(1)

    return answer[:1] if answer[:1] in {"A", "B", "C", "D"} else answer


def _build_mcq_evaluation(model_answer: str, student_answer: str) -> dict:
    """
    Evaluate MCQs deterministically to avoid unnecessary LLM calls.
    """
    correct_choice = _normalize_choice(model_answer)
    student_choice = _normalize_choice(student_answer)

    if not student_choice:
        return {
            "score": 0.0,
            "feedback": "No answer was provided for this multiple choice question.",
            "weak_areas": ["Skipped multiple choice question"]
        }

    if student_choice == correct_choice:
        return {
            "score": 10.0,
            "feedback": "Correct choice selected. You identified the right answer.",
            "weak_areas": []
        }

    return {
        "score": 0.0,
        "feedback": f"Incorrect choice selected. The correct option was {correct_choice}.",
        "weak_areas": ["Multiple choice accuracy"]
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _build_text_fallback_evaluation(model_answer: str, student_answer: str, errors: list[str]) -> dict:
    """
    Use a transparent keyword-overlap fallback when all evaluator models fail.
    """
    if not student_answer or student_answer == "[No answer provided]":
        return {
            "score": 0.0,
            "feedback": "No answer was provided, so there was nothing to evaluate.",
            "weak_areas": ["No response submitted"]
        }

    model_tokens = [token for token in _tokenize(model_answer) if token not in STOPWORDS]
    student_tokens = [token for token in _tokenize(student_answer) if token not in STOPWORDS]

    model_keywords = set(model_tokens)
    student_keywords = set(student_tokens)
    overlap = model_keywords & student_keywords
    coverage = len(overlap) / max(len(model_keywords), 1)
    missing_keywords = list(model_keywords - student_keywords)[:3]

    if coverage >= 0.75:
        score = 7.5
    elif coverage >= 0.45:
        score = 5.5
    elif coverage > 0:
        score = 2.5
    else:
        score = 1.0

    feedback = (
        "LLM-based evaluation was unavailable, so this score was estimated using keyword overlap with the model answer. "
        "Re-run once your Groq evaluator model is available for a more reliable score."
    )

    if errors:
        feedback += f" Last error: {errors[-1]}"

    weak_areas = missing_keywords or ["Answer completeness"]

    return {
        "score": score,
        "feedback": feedback,
        "weak_areas": weak_areas
    }


def _evaluate_text_answer(model_answer: str, student_answer: str, chain_cache: dict) -> dict:
    """
    Try supported Groq evaluator models in order, then fall back locally.
    """
    errors = []

    for model_name in get_evaluation_model_candidates():
        try:
            if model_name not in chain_cache:
                chain_cache[model_name] = create_evaluation_chain(model_name)

            resp = chain_cache[model_name].invoke({
                "model_answer": model_answer,
                "student_answer": student_answer
            })

            eval_dict = resp if isinstance(resp, dict) else json.loads(resp)
            score = max(0, min(10, float(eval_dict.get("score", 0))))

            return {
                "score": round(score, 1),
                "feedback": eval_dict.get("feedback", "Evaluation completed."),
                "weak_areas": eval_dict.get("weak_areas", []),
            }
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    return _build_text_fallback_evaluation(model_answer, student_answer, errors)


def evaluate_answers(questions_data, student_answers_data: dict):
    """
    Evaluate answers using individual answer matching with improved scoring.
    """
    chain_cache = {}
    
    results = []
    total_score = 0
    weak_areas = []
    question_count = 0
    
    # Extract questions and answers
    questions = questions_data.get('questions', {})
    answers = questions_data.get('answers', {})
    
    # Extract student answers
    individual_answers = student_answers_data.get('individual_answers', [])
    
    # Flatten all questions in order
    all_questions = []
    all_model_answers = []
    
    # Add MCQs first
    if 'mcqs' in questions and 'mcqs' in answers:
        for q, a in zip(questions['mcqs'], answers['mcqs']):
            all_questions.append({
                'type': 'mcq',
                'question': q.get('question', 'Unknown question'),
                'options': q.get('options', [])
            })
            all_model_answers.append(a.get('model_answer', 'No model answer'))
    
    # Add short answers
    if 'shorts' in questions and 'shorts' in answers:
        for q, a in zip(questions['shorts'], answers['shorts']):
            all_questions.append({
                'type': 'short',
                'question': q.get('question', 'Unknown question')
            })
            all_model_answers.append(a.get('model_answer', 'No model answer'))
    
    # Add long answers
    if 'longs' in questions and 'longs' in answers:
        for q, a in zip(questions['longs'], answers['longs']):
            all_questions.append({
                'type': 'long',
                'question': q.get('question', 'Unknown question')
            })
            all_model_answers.append(a.get('model_answer', 'No model answer'))
    
    if not all_questions:
        return {"error": "No questions to evaluate"}
    
    # Evaluate each question with its corresponding student answer
    for i, (question, model_ans) in enumerate(zip(all_questions, all_model_answers)):
        # Get the corresponding student answer
        if i < len(individual_answers):
            student_ans = individual_answers[i]
        else:
            student_ans = ""
        
        # Handle empty or very short answers
        if not student_ans or len(student_ans.strip()) < 1:
            student_ans = "[No answer provided]"
        
        try:
            if question['type'] == 'mcq':
                eval_dict = _build_mcq_evaluation(model_ans, student_ans)
            else:
                eval_dict = _evaluate_text_answer(model_ans, student_ans, chain_cache)
            
            score = float(eval_dict.get('score', 0))
                
            results.append({
                "question_number": i + 1,
                "question_type": question['type'],
                "question": question['question'][:100] + '...' if len(question['question']) > 100 else question['question'],
                "model_answer": model_ans,
                "student_answer": student_ans,
                "evaluation": eval_dict
            })
            total_score += score
            weak_areas.extend(eval_dict.get('weak_areas', []))
            question_count += 1
            
        except Exception as e:
            print(f"Evaluation error for question {i+1}: {e}")
            default_score = 0.0 if student_ans == "[No answer provided]" else 1.0
            results.append({
                "question_number": i + 1,
                "question_type": question['type'],
                "question": question['question'],
                "model_answer": model_ans,
                "student_answer": student_ans,
                "evaluation": {
                    "score": default_score,
                    "feedback": f"Evaluation failed unexpectedly: {str(e)}",
                    "weak_areas": ["Evaluation system error"]
                }
            })
            total_score += default_score
            question_count += 1
    
    avg_score = total_score / question_count if question_count > 0 else 0
    
    # Generate study plan based on results
    study_plan = generate_study_plan(avg_score, weak_areas, results)
    
    return {
        "avg_score": round(avg_score, 1), 
        "total_questions": question_count,
        "results": results, 
        "weak_areas": list(set(weak_areas)),
        "study_plan": study_plan
    }

def generate_study_plan(avg_score: float, weak_areas: list, results: list):
    """
    Generate a personalized study plan based on evaluation results.
    """
    # Determine performance level
    if avg_score >= 8:
        performance_level = "Excellent"
        overall_assessment = "You have demonstrated strong understanding of the material."
    elif avg_score >= 6:
        performance_level = "Good" 
        overall_assessment = "You have a good grasp of the concepts with some areas for improvement."
    elif avg_score >= 4:
        performance_level = "Average"
        overall_assessment = "You understand the basics but need to focus on key concepts."
    else:
        performance_level = "Needs Improvement"
        overall_assessment = "Focus on fundamental concepts and regular practice."
    
    # Analyze question type performance
    mcq_scores = []
    short_scores = []
    long_scores = []
    
    for result in results:
        if 'evaluation' in result:
            score = result['evaluation'].get('score', 0)
            if result['question_type'] == 'mcq':
                mcq_scores.append(score)
            elif result['question_type'] == 'short':
                short_scores.append(score)
            elif result['question_type'] == 'long':
                long_scores.append(score)
    
    # Calculate average scores by type
    avg_mcq = sum(mcq_scores) / len(mcq_scores) if mcq_scores else 0
    avg_short = sum(short_scores) / len(short_scores) if short_scores else 0
    avg_long = sum(long_scores) / len(long_scores) if long_scores else 0
    
    # Identify priority areas
    priority_topics = []
    if weak_areas:
        for area in weak_areas[:3]:  # Top 3 weak areas
            priority_topics.append({
                "topic": area,
                "urgency": "High" if avg_score < 6 else "Medium",
                "reason": f"Identified as a challenging concept based on your answers"
            })
    
    # Generate study strategies
    study_strategies = []
    
    if avg_mcq < 6:
        study_strategies.append({
            "area": "Multiple Choice Questions",
            "strategy": "Practice identifying key concepts and eliminating wrong options",
            "resources": "Focus on understanding why incorrect options are wrong"
        })
    
    if avg_short < 6:
        study_strategies.append({
            "area": "Short Answer Questions", 
            "strategy": "Work on concise yet complete responses",
            "resources": "Practice writing 2-3 sentence summaries of key concepts"
        })
    
    if avg_long < 6:
        study_strategies.append({
            "area": "Long Answer Questions",
            "strategy": "Develop structured responses with clear explanations",
            "resources": "Practice outlining answers before writing"
        })
    
    # Set goals
    short_term_goals = [
        "Review all incorrect answers",
        "Create flashcards for key concepts",
        "Practice 10 similar questions daily"
    ]
    
    long_term_goals = [
        "Achieve consistent scores above 8/10",
        "Master all identified weak areas", 
        "Develop effective study habits"
    ]
    
    # Time management recommendations
    daily_study_time = "45-60 minutes" if avg_score >= 7 else "60-90 minutes" if avg_score >= 5 else "90-120 minutes"
    
    focus_distribution = []
    if weak_areas:
        focus_distribution.append(f"70% on weak areas: {', '.join(weak_areas[:2])}")
        focus_distribution.append("30% on reinforcement of strong areas")
    else:
        focus_distribution.append("50% on advanced concepts")
        focus_distribution.append("50% on application practice")
    
    return {
        "performance_level": performance_level,
        "overall_assessment": overall_assessment,
        "priority_topics": priority_topics,
        "study_strategies": study_strategies,
        "short_term_goals": short_term_goals,
        "long_term_goals": long_term_goals,
        "time_management": {
            "daily_study_time": daily_study_time,
            "focus_distribution": "; ".join(focus_distribution)
        },
        "motivational_message": "Consistent practice and focused effort will lead to significant improvement. You've got this!"
    }
