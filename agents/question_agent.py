from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from utils.rag_utils import retrieve_context
from dotenv import load_dotenv
import os
import json

load_dotenv()

def generate_questions(topic: str, vectorstore, question_config: dict):
    """
    Generate questions based on selected types and counts using Groq.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.6,
    )
    
    if vectorstore is None:
        print("Warning: Vectorstore not loaded. Proceeding with generic question generation.")
        context = "No document provided. Generate general questions about the topic."
    else:
        context = retrieve_context(vectorstore, topic, k=3)
    
    # Build question type instructions based on configuration
    type_instructions = []
    
    if question_config['mcq']['enabled']:
        mcq_count = question_config['mcq']['count']
        type_instructions.append(f"{mcq_count} MCQs (each with 4 options A-D and model answer)")
    
    if question_config['short']['enabled']:
        short_count = question_config['short']['count']
        type_instructions.append(f"{short_count} Short answer questions (with model answer)")
    
    if question_config['long']['enabled']:
        long_count = question_config['long']['count']
        type_instructions.append(f"{long_count} Long answer question (with model answer)")
    
    type_instruction_text = "\n".join([f"- {instruction}" for instruction in type_instructions])
    
    prompt = PromptTemplate(
        input_variables=["topic", "context", "type_instructions"],
        template="""
        Based on the topic '{topic}' and the following document context:
        {context}
        
        Generate a question paper with EXACTLY these question types and counts:
        {type_instructions}
        
        IMPORTANT: 
        - DO NOT include answers in the main output
        - Store model answers separately for evaluation
        - Questions should be educational and test understanding
        - Generate EXACTLY the number of questions specified for each type
        - For MCQs: provide exactly 4 options (A, B, C, D) and indicate the correct one in model_answer
        - For short answers: expect 2-3 sentence responses
        - For long answers: expect detailed 5-7 sentence responses
        
        Output ONLY valid JSON with this structure:
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
    
    try:
        response = chain.invoke({
            "topic": topic, 
            "context": context, 
            "type_instructions": type_instruction_text
        })
        questions = response
        if isinstance(response, str):
            questions = json.loads(response)
        
        # Validate the generated questions match the requested counts
        if 'questions' in questions:
            questions_data = questions['questions']
            
            # Validate MCQ count
            if question_config['mcq']['enabled']:
                expected_mcq = question_config['mcq']['count']
                actual_mcq = len(questions_data.get('mcqs', []))
                if actual_mcq != expected_mcq:
                    print(f"Warning: Generated {actual_mcq} MCQs but expected {expected_mcq}")
            
            # Validate short answer count
            if question_config['short']['enabled']:
                expected_short = question_config['short']['count']
                actual_short = len(questions_data.get('shorts', []))
                if actual_short != expected_short:
                    print(f"Warning: Generated {actual_short} short answers but expected {expected_short}")
            
            # Validate long answer count
            if question_config['long']['enabled']:
                expected_long = question_config['long']['count']
                actual_long = len(questions_data.get('longs', []))
                if actual_long != expected_long:
                    print(f"Warning: Generated {actual_long} long answers but expected {expected_long}")
        
    except Exception as e:
        print(f"Generation error: {e}")
        questions = {"error": f"Generation failed: {str(e)}"}
    
    return questions