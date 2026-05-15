# app.py
import streamlit as st
from agents.pdf_agent import process_pdf
from agents.question_agent import generate_questions
from agents.evaluation_agent import evaluate_answers
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# --- Constants (directly here) ---
GROQ_MODEL = "llama-3.1-8b-instant"
PERSIST_DIR = "./vector_db"

# --- UI Styling ---
st.markdown("""
<style>
.main .block-container {padding-top: 2rem;}
.stButton > button {background-color:#8B5CF6;color:white;border:none;border-radius:8px;padding:0.5rem 1rem;font-weight:500;}
.stButton > button:hover {background-color:#7C3AED;color:white;}
.step-circle {background:#A78BFA;color:white;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:16px;z-index:2;}
.step-circle-active {background:#8B5CF6;box-shadow:0 0 0 3px #C4B5FD;}
.step-circle-completed {background:#10B981;}
.question-box {background:#F3F4F6;border-left:4px solid #8B5CF6;padding:1rem;margin:0.5rem 0;border-radius:4px;}
.study-plan-box {background:#FEF3C7;border:2px solid #F59E0B;border-radius:8px;padding:1.5rem;margin:1rem 0;}
</style>
""", unsafe_allow_html=True)

st.title("AI Quiz Taker System")
st.markdown("### Intelligent Document-Based Quiz Generator & Evaluator")
st.markdown("*Powered by Groq, Gemini Embeddings, and Pinecone*")

# --- Session State ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'topic' not in st.session_state:
    st.session_state.topic = ""
if 'questions' not in st.session_state:
    st.session_state.questions = None
if 'student_answers' not in st.session_state:
    st.session_state.student_answers = {}
if 'mcq_index' not in st.session_state:
    st.session_state.mcq_index = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = ""
if 'question_config' not in st.session_state:
    st.session_state.question_config = {
        'mcq': {'enabled': True, 'count': 3},
        'short': {'enabled': True, 'count': 2},
        'long': {'enabled': True, 'count': 1}
    }

# --- Progress Steps ---
steps = ["Upload PDF", "Select Topic", "Configure Questions", "Get Questions", "Enter Answers", "Get Feedback"]
cols = st.columns(6)
for i, step in enumerate(steps):
    with cols[i]:
        circle_class = "step-circle"
        if st.session_state.step == i + 1:
            circle_class += " step-circle-active"
        elif st.session_state.step > i + 1:
            circle_class += " step-circle-completed"
        st.markdown(f'<div class="{circle_class}">{i+1}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;margin-top:.5rem;font-size:12px;color:#6B7280;">{step}</div>', unsafe_allow_html=True)
st.divider()

# --- Step 1: Upload PDF ---
if st.session_state.step == 1:
    st.subheader("Upload Your Document")
    pdf_upload = st.file_uploader("Choose PDF file", type="pdf", key="pdf_uploader_step1")
    if pdf_upload:
        st.session_state.pdf_name = pdf_upload.name.replace('.pdf', '')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_upload.read())
            pdf_path = tmp.name
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = process_pdf(pdf_path, st.session_state.pdf_name)
            os.unlink(pdf_path)
            if st.session_state.vectorstore:
                st.success("PDF processed successfully!")
                if st.button("Next Step", key="step1_next"):
                    st.session_state.step = 2
                    st.rerun()
            else:
                st.error("Failed to process PDF.")

# --- Step 2: Select Topic ---
elif st.session_state.step == 2:
    st.subheader("Choose Quiz Topic")
    topic = st.text_input("Topic", value=st.session_state.topic, placeholder="Enter topic here...")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", key="step2_back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if topic.strip() and st.button("Next Step", key="step2_next"):
            st.session_state.topic = topic.strip()
            st.session_state.step = 3
            st.rerun()

# --- Step 3: Configure Questions ---
elif st.session_state.step == 3:
    st.subheader("Configure Quiz")
    cfg = st.session_state.question_config

    c1, c2 = st.columns([3, 1])
    with c1:
        mcq_enabled = st.checkbox("Multiple Choice Questions", value=cfg['mcq']['enabled'])
    with c2:
        mcq_count = st.number_input("MCQ Count", 1, 10, cfg['mcq']['count']) if mcq_enabled else 0

    c1, c2 = st.columns([3, 1])
    with c1:
        short_enabled = st.checkbox("Short Answer Questions", value=cfg['short']['enabled'])
    with c2:
        short_count = st.number_input("Short Count", 1, 10, cfg['short']['count']) if short_enabled else 0

    c1, c2 = st.columns([3, 1])
    with c1:
        long_enabled = st.checkbox("Long Answer Questions", value=cfg['long']['enabled'])
    with c2:
        long_count = st.number_input("Long Count", 1, 5, cfg['long']['count']) if long_enabled else 0

    st.session_state.question_config = {
        'mcq': {'enabled': mcq_enabled, 'count': mcq_count},
        'short': {'enabled': short_enabled, 'count': short_count},
        'long': {'enabled': long_enabled, 'count': long_count}
    }

    total = mcq_count + short_count + long_count
    if total > 0:
        st.info(f"Total Questions: {total}")
    else:
        st.error("Select at least one question type")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="step3_back"):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if total > 0 and st.button("Next Step", key="step3_next"):
            st.session_state.step = 4
            st.rerun()

# --- Step 4: Generate Questions ---
elif st.session_state.step == 4:
    st.subheader("Generate Questions")
    cfg = st.session_state.question_config
    total = sum(cfg[t]['count'] if cfg[t]['enabled'] else 0 for t in ('mcq', 'short', 'long'))
    st.info(f"**Topic:** {st.session_state.topic}\n\n**Total Questions:** {total}")

    if st.button("Generate Questions", type="primary"):
        with st.spinner("Generating..."):
            st.session_state.questions = generate_questions(
                st.session_state.topic,
                st.session_state.vectorstore,
                st.session_state.question_config
            )
        if st.session_state.questions and 'error' not in st.session_state.questions:
            st.success("Questions generated!")
            qdata = st.session_state.questions['questions']
            qno = 1
            for typ in ('mcqs', 'shorts', 'longs'):
                if typ in qdata:
                    title = {"mcqs": "Multiple Choice", "shorts": "Short Answer", "longs": "Long Answer"}[typ]
                    st.markdown(f"#### {title} Questions")
                    for q in qdata[typ]:
                        st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
                        st.markdown(f"**Q{qno}.** {q.get('question', '—')}")
                        if typ == 'mcqs':
                            for j, opt in enumerate(q.get('options', [])):
                                st.markdown(f"   {chr(65+j)}. {opt}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        qno += 1
        else:
            st.error(st.session_state.questions.get('error', 'Generation failed'))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="step4_back"):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.session_state.questions and 'error' not in st.session_state.questions:
            if st.button("Next Step", key="step4_next"):
                st.session_state.step = 5
                st.rerun()

# --- Step 5: Enter Answers (FIXED) ---
elif st.session_state.step == 5:
    st.subheader("Enter Your Answers")

    if not st.session_state.questions or 'questions' not in st.session_state.questions:
        st.error("No questions. Generate them first.")
        if st.button("Back"):
            st.session_state.step = 4
            st.rerun()
        st.stop()

    qdata = st.session_state.questions['questions']
    total_q = sum(len(qdata.get(t, [])) for t in ('mcqs', 'shorts', 'longs'))

    qno = 1

    # MCQs
    if 'mcqs' in qdata:
        st.markdown("#### Multiple Choice Questions")
        for i, mcq in enumerate(qdata['mcqs']):
            key = f"mcq_{i}"
            opts = mcq.get('options', [])
            st.markdown(f"**Q{qno}.** {mcq.get('question', '—')}")
            chosen_idx = st.session_state.mcq_index.get(key)
            radio_options = [f"{chr(65+j)}. {opt}" for j, opt in enumerate(opts)]
            if chosen_idx is not None and not (0 <= chosen_idx < len(radio_options)):
                chosen_idx = None
            selected = st.radio(
                "Choose one:",
                options=radio_options,
                index=chosen_idx,
                key=key
            )
            if selected is None:
                st.session_state.mcq_index[key] = None
                st.session_state.student_answers[key] = ""
            else:
                sel_idx = radio_options.index(selected)
                st.session_state.mcq_index[key] = sel_idx
                st.session_state.student_answers[key] = chr(65 + sel_idx)
            qno += 1

    # Short Answers
    if 'shorts' in qdata:
        st.markdown("#### Short Answer Questions")
        for i, s in enumerate(qdata['shorts']):
            key = f"short_{i}"
            st.markdown(f"**Q{qno}.** {s.get('question', '—')}")
            ans = st.text_area("Your answer:", value=st.session_state.student_answers.get(key, ""), height=100, key=key)
            st.session_state.student_answers[key] = ans.strip()
            qno += 1

    # Long Answers
    if 'longs' in qdata:
        st.markdown("#### Long Answer Questions")
        for i, l in enumerate(qdata['longs']):
            key = f"long_{i}"
            st.markdown(f"**Q{qno}.** {l.get('question', '—')}")
            ans = st.text_area("Your answer:", value=st.session_state.student_answers.get(key, ""), height=150, key=key)
            st.session_state.student_answers[key] = ans.strip()
            qno += 1

    answered = sum(1 for v in st.session_state.student_answers.values() if v)
    st.info(f"Progress: {answered}/{total_q} answered")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="step5_back"):
            st.session_state.step = 4
            st.rerun()
    with c2:
        if answered > 0 and st.button("Submit Answers", type="primary", key="step5_submit"):
            ordered = []
            for t, name in [('mcqs', 'mcq'), ('shorts', 'short'), ('longs', 'long')]:
                for i in range(len(qdata.get(t, []))):
                    ordered.append(st.session_state.student_answers.get(f"{name}_{i}", ""))
            payload = {"individual_answers": ordered}

            with st.spinner("Evaluating..."):
                st.session_state.results = evaluate_answers(st.session_state.questions, payload)

            if st.session_state.results and 'error' not in st.session_state.results:
                st.session_state.step = 6
                st.rerun()
            else:
                st.error(st.session_state.results.get('error', 'Evaluation failed'))

# --- Step 6: Results ---
elif st.session_state.step == 6:
    st.subheader("Evaluation Results")
    res = st.session_state.results
    st.markdown(f"**Average Score:** {res['avg_score']}/10")
    st.markdown(f"**Total Questions:** {res['total_questions']}")

    for r in res['results']:
        st.markdown(f"### Q{r['question_number']} ({r['question_type'].upper()})")
        st.markdown(r['question'])
        st.markdown("**Your Answer:**"); st.write(r['student_answer'] or "[No answer]")
        st.markdown("**Model Answer:**"); st.write(r['model_answer'])
        ev = r['evaluation']
        st.markdown(f"**Score:** {ev['score']}/10")
        st.markdown("**Feedback:**"); st.write(ev['feedback'])
        if ev.get('weak_areas'):
            st.markdown("**Weak Areas:**")
            for a in ev['weak_areas']:
                st.markdown(f"- {a}")

    if res.get('weak_areas'):
        st.markdown("### Overall Weak Areas")
        for a in set(res['weak_areas']):
            st.markdown(f"- {a}")

    plan = res.get('study_plan')
    if plan:
        st.markdown('<div class="study-plan-box">', unsafe_allow_html=True)
        st.markdown("### Personalized Study Plan")
        st.markdown(f"**Performance:** {plan['performance_level']}")
        st.markdown(plan['overall_assessment'])
        if plan.get('priority_topics'):
            st.markdown("#### Priority Topics")
            for t in plan['priority_topics']:
                st.markdown(f"- **{t['topic']}** ({t['urgency']}) – {t['reason']}")
        if plan.get('study_strategies'):
            st.markdown("#### Strategies")
            for s in plan['study_strategies']:
                st.markdown(f"- **{s['area']}**: {s['strategy']} – {s['resources']}")
        st.markdown("#### Short-term Goals")
        for g in plan['short_term_goals']:
            st.markdown(f"- {g}")
        st.markdown("#### Long-term Goals")
        for g in plan['long_term_goals']:
            st.markdown(f"- {g}")
        st.markdown("#### Time Management")
        tm = plan['time_management']
        st.markdown(f"- Daily: {tm['daily_study_time']}")
        st.markdown(f"- Focus: {tm['focus_distribution']}")
        st.markdown(f"**{plan['motivational_message']}**")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Start New Quiz", type="primary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
