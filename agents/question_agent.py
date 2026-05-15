from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from utils.rag_utils import retrieve_context
from dotenv import load_dotenv
import json
import os

load_dotenv()


QUESTION_TYPES = {
    "mcq": {
        "question_key": "mcqs",
        "answer_key": "mcqs",
        "label": "MCQs",
    },
    "short": {
        "question_key": "shorts",
        "answer_key": "shorts",
        "label": "short answer questions",
    },
    "long": {
        "question_key": "longs",
        "answer_key": "longs",
        "label": "long answer questions",
    },
}


def _get_requested_counts(question_config: dict) -> dict:
    counts = {}
    for question_type, meta in QUESTION_TYPES.items():
        enabled = question_config.get(question_type, {}).get("enabled", False)
        count = int(question_config.get(question_type, {}).get("count", 0)) if enabled else 0
        counts[question_type] = count
    return counts


def _build_type_instruction_text(requested_counts: dict) -> str:
    instructions = []

    if requested_counts["mcq"] > 0:
        instructions.append(
            f"- {requested_counts['mcq']} MCQs (each with exactly 4 options A-D and one correct model answer)"
        )
    if requested_counts["short"] > 0:
        instructions.append(
            f"- {requested_counts['short']} short answer questions (each with one model answer)"
        )
    if requested_counts["long"] > 0:
        instructions.append(
            f"- {requested_counts['long']} long answer questions (each with one model answer)"
        )

    return "\n".join(instructions)


def _parse_response(response):
    return response if isinstance(response, dict) else json.loads(response)


def _normalize_question_set(raw_questions: dict, requested_counts: dict) -> tuple[dict, list[str]]:
    """
    Keep only the requested structure and verify exact counts.
    """
    normalized = {
        "questions": {
            "mcqs": [],
            "shorts": [],
            "longs": [],
        },
        "answers": {
            "mcqs": [],
            "shorts": [],
            "longs": [],
        },
    }
    mismatches = []

    source_questions = raw_questions.get("questions", {}) if isinstance(raw_questions, dict) else {}
    source_answers = raw_questions.get("answers", {}) if isinstance(raw_questions, dict) else {}

    for question_type, expected_count in requested_counts.items():
        question_key = QUESTION_TYPES[question_type]["question_key"]
        answer_key = QUESTION_TYPES[question_type]["answer_key"]
        label = QUESTION_TYPES[question_type]["label"]

        question_items = source_questions.get(question_key, []) or []
        answer_items = source_answers.get(answer_key, []) or []

        if not isinstance(question_items, list):
            question_items = []
        if not isinstance(answer_items, list):
            answer_items = []

        paired_items = list(zip(question_items, answer_items))

        cleaned_pairs = []
        for question_item, answer_item in paired_items:
            if question_type == "mcq":
                options = question_item.get("options", [])
                model_answer = str(answer_item.get("model_answer", "")).strip().upper()
                if (
                    question_item.get("question")
                    and isinstance(options, list)
                    and len(options) == 4
                    and model_answer in {"A", "B", "C", "D"}
                ):
                    cleaned_pairs.append((
                        {
                            "question": question_item["question"],
                            "options": options[:4],
                        },
                        {
                            "model_answer": model_answer,
                        },
                    ))
            else:
                if question_item.get("question") and answer_item.get("model_answer"):
                    cleaned_pairs.append((
                        {"question": question_item["question"]},
                        {"model_answer": answer_item["model_answer"]},
                    ))

        if len(cleaned_pairs) < expected_count:
            mismatches.append(
                f"Expected {expected_count} {label}, but only got {len(cleaned_pairs)} valid items."
            )

        trimmed_pairs = cleaned_pairs[:expected_count]
        normalized["questions"][question_key] = [pair[0] for pair in trimmed_pairs]
        normalized["answers"][answer_key] = [pair[1] for pair in trimmed_pairs]

        actual_count = len(trimmed_pairs)
        if actual_count != expected_count:
            mismatches.append(
                f"Expected {expected_count} {label}, but final normalized result has {actual_count}."
            )

    return normalized, mismatches


def generate_questions(topic: str, vectorstore, question_config: dict):
    """
    Generate questions based on selected types and counts using Groq.
    """
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.2,
    )

    if vectorstore is None:
        print("Warning: Vectorstore not loaded. Proceeding with generic question generation.")
        context = "No document provided. Generate general questions about the topic."
    else:
        context = retrieve_context(vectorstore, topic, k=3)

    requested_counts = _get_requested_counts(question_config)
    type_instruction_text = _build_type_instruction_text(requested_counts)

    prompt = PromptTemplate(
        input_variables=["topic", "context", "type_instructions", "retry_feedback"],
        template="""
        Based on the topic '{topic}' and the following document context:
        {context}

        Generate a question paper with EXACTLY these question types and counts:
        {type_instructions}

        STRICT REQUIREMENTS:
        - Follow the requested counts exactly.
        - If a type is not requested, return an empty array for it.
        - Keep question counts and answer counts perfectly aligned.
        - DO NOT include answers in the main questions output.
        - Questions should test understanding of the provided context.
        - For MCQs: provide exactly 4 options and the correct answer must be A, B, C, or D.
        - For short answers: provide one concise model answer per question.
        - For long answers: provide one detailed model answer per question.

        Retry guidance from previous attempt:
        {retry_feedback}

        Output ONLY valid JSON with this exact structure:
        {{
            "questions": {{
                "mcqs": [{{"question": "Question text?", "options": ["Option A", "Option B", "Option C", "Option D"]}}],
                "shorts": [{{"question": "Question text?"}}],
                "longs": [{{"question": "Question text?"}}]
            }},
            "answers": {{
                "mcqs": [{{"model_answer": "A"}}],
                "shorts": [{{"model_answer": "Detailed correct answer..."}}],
                "longs": [{{"model_answer": "Comprehensive correct answer..."}}]
            }}
        }}
        """
    )

    base_parser = JsonOutputParser()
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    chain = prompt | llm | fixing_parser

    retry_feedback = "No previous issues."

    try:
        for attempt in range(3):
            response = chain.invoke({
                "topic": topic,
                "context": context,
                "type_instructions": type_instruction_text,
                "retry_feedback": retry_feedback,
            })
            questions = _parse_response(response)
            normalized_questions, mismatches = _normalize_question_set(questions, requested_counts)

            if not mismatches:
                return normalized_questions

            retry_feedback = (
                "Your previous output did not match the requested counts. "
                "Fix these issues exactly:\n- " + "\n- ".join(mismatches)
            )
            print(f"Question generation retry {attempt + 1}: {' | '.join(mismatches)}")

        return {
            "error": (
                "Generation failed to match the requested question counts after multiple attempts. "
                f"Last issues: {'; '.join(mismatches)}"
            )
        }

    except Exception as e:
        print(f"Generation error: {e}")
        return {"error": f"Generation failed: {str(e)}"}
