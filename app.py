import streamlit as st
import PyPDF2
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from groq import Groq

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="AI HR Recruitment Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI HR Recruitment Agent")
st.write("Upload a Resume and get AI-powered analysis")

# ==========================================
# GROQ API
# ==========================================
import streamlit as st
from groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=GROQ_API_KEY
)

# ==========================================
# STATE
# ==========================================

class HRState(TypedDict):
    resume_text: str
    skills: str
    score: str
    ats_score: str
    missing_skills: str
    job_match: str
    interview_questions: str
    final_report: str

# ==========================================
# AGENT 1 - SKILLS
# ==========================================

def skills_agent(state):
    prompt = f"""
    Extract all technical skills from this resume.

    Resume:
    {state['resume_text']}

    Return only skills in bullet points.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "skills": response.choices[0].message.content
    }

# ==========================================
# AGENT 2 - RESUME SCORE
# ==========================================

def score_agent(state):
    prompt = f"""
    Analyze this resume.

    Give:
    1. Score out of 100
    2. Strengths
    3. Weaknesses

    Resume:
    {state['resume_text']}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "score": response.choices[0].message.content
    }

# ==========================================
# AGENT 3 - ATS SCORE
# ==========================================

def ats_agent(state):
    prompt = f"""
    Analyze this resume for ATS compatibility.

    Give:
    1. ATS Score out of 100
    2. Missing Sections
    3. ATS Improvements

    Resume:
    {state['resume_text']}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "ats_score": response.choices[0].message.content
    }

# ==========================================
# AGENT 4 - MISSING SKILLS
# ==========================================

def missing_skills_agent(state):
    prompt = f"""
    Candidate Skills:

    {state['skills']}

    Compare with modern AI Engineer /
    GenAI Engineer requirements.

    Give:
    1. Missing Skills
    2. Recommended Technologies
    3. Learning Roadmap
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "missing_skills": response.choices[0].message.content
    }

# ==========================================
# AGENT 5 - JOB MATCH
# ==========================================

def job_match_agent(state):
    prompt = f"""
    Based on these skills:

    {state['skills']}

    Suggest:
    - Best Job Roles
    - Suitable Career Paths
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "job_match": response.choices[0].message.content
    }

# ==========================================
# AGENT 6 - INTERVIEW QUESTIONS
# ==========================================

def interview_agent(state):
    prompt = f"""
    Generate 10 interview questions.

    Skills:
    {state['skills']}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "interview_questions":
        response.choices[0].message.content
    }

# ==========================================
# SUPERVISOR
# ==========================================

def supervisor(state):

    report = f"""
# 🤖 AI HR Recruitment Report

---

## 📊 Resume Score

{state['score']}

---

## 🎯 ATS Analysis

{state['ats_score']}

---

## 🛠 Skills Found

{state['skills']}

---

## ❌ Missing Skills

{state['missing_skills']}

---

## 💼 Best Job Matches

{state['job_match']}

---

## 🎤 Interview Questions

{state['interview_questions']}

---

Generated using LangGraph + Groq
"""

    return {
        "final_report": report
    }

# ==========================================
# BUILD GRAPH
# ==========================================

graph = StateGraph(HRState)

graph.add_node("skills_agent", skills_agent)
graph.add_node("score_agent", score_agent)
graph.add_node("ats_agent", ats_agent)
graph.add_node("missing_skills_agent", missing_skills_agent)
graph.add_node("job_match_agent", job_match_agent)
graph.add_node("interview_agent", interview_agent)
graph.add_node("supervisor", supervisor)

graph.add_edge(START, "skills_agent")
graph.add_edge("skills_agent", "score_agent")
graph.add_edge("score_agent", "ats_agent")
graph.add_edge("ats_agent", "missing_skills_agent")
graph.add_edge("missing_skills_agent", "job_match_agent")
graph.add_edge("job_match_agent", "interview_agent")
graph.add_edge("interview_agent", "supervisor")
graph.add_edge("supervisor", END)

app = graph.compile()

# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Analyze Resume"):

        with st.spinner("Analyzing Resume..."):

            reader = PyPDF2.PdfReader(uploaded_file)

            resume_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    resume_text += text

            result = app.invoke({
                "resume_text": resume_text
            })

        st.success("Analysis Completed!")

        st.markdown(result["final_report"])

        st.download_button(
            label="📥 Download Report",
            data=result["final_report"],
            file_name="HR_Report.txt",
            mime="text/plain"
        )
