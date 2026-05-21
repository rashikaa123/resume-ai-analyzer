import streamlit as st
import pdfplumber
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# -----------------------------
# CUSTOM BACKGROUND + UI DESIGN
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .stButton>button {
        background-color: #00c6ff;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }

    .stProgress > div > div > div > div {
        background-color: #00ff99;
    }

    .css-1d391kg {
        background-color: #1e293b;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Resume Analyzer")
# -----------------------------
# SKILLS DATABASE
# -----------------------------
required_skills = [
    "python", "java", "sql", "machine learning", "data analysis",
    "html", "css", "javascript", "react", "flask", "django",
    "power bi", "excel", "communication", "teamwork"
]

# -----------------------------
# RESUME TEXT EXTRACTION
# -----------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# -----------------------------
# EXTRACT NAME
# -----------------------------
def extract_name(text):
    lines = text.split('\n')

    for line in lines[:5]:
        line = line.strip()

        if len(line.split()) <= 3 and line.replace(' ', '').isalpha():
            return line

    return "Candidate"

# -----------------------------
# SKILL EXTRACTION
# -----------------------------
def extract_skills(text):
    text = text.lower()

    found_skills = []

    for skill in required_skills:
        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills

# -----------------------------
# SECTION DETECTION
# -----------------------------
def check_sections(text):
    text = text.lower()

    sections = {
        "Education": "education" in text,
        "Skills": "skills" in text,
        "Projects": "project" in text,
        "Experience": "experience" in text,
        "Certifications": "certification" in text
    }

    return sections

# -----------------------------
# ATS SCORE CALCULATION
# -----------------------------
def calculate_ats_score(skills, sections, resume_text):

    # Skill Score
    skills_score = (len(skills) / len(required_skills)) * 100

    # Section Score
    total_sections = len(sections)
    present_sections = sum(sections.values())
    section_score = (present_sections / total_sections) * 100

    # Resume Length Score
    word_count = len(resume_text.split())

    if word_count > 350:
        resume_score = 100
    elif word_count > 200:
        resume_score = 80
    else:
        resume_score = 50

    # Final Weighted Score
    final_score = (
        skills_score * 0.5 +
        section_score * 0.3 +
        resume_score * 0.2
    )

    return round(final_score, 2)

# -----------------------------
# ATS LEVEL
# -----------------------------
def ats_level(score):

    if score >= 85:
        return "Excellent"

    elif score >= 70:
        return "Good"

    elif score >= 50:
        return "Average"

    else:
        return "Needs Improvement"

# -----------------------------
# SMART FEEDBACK
# -----------------------------
def generate_feedback(score, sections):

    feedback = []

    if score >= 85:
        feedback.append("Your resume is highly ATS optimized")

    elif score >= 70:
        feedback.append("Your resume looks strong but still has improvement opportunities ")

    else:
        feedback.append("Your resume needs better optimization for ATS systems")

    if not sections["Projects"]:
        feedback.append("Add projects section to showcase practical work")

    if not sections["Certifications"]:
        feedback.append("Adding certifications can improve recruiter confidence")

    if not sections["Experience"]:
        feedback.append("Try adding internships or practical experience")

    return feedback

# -----------------------------
# JOB DESCRIPTION MATCHING
# -----------------------------
def jd_match(resume_text, jd_text):

    documents = [resume_text, jd_text]

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])

    return round(similarity[0][0] * 100, 2)

# -----------------------------
# CAREER ROLE PREDICTION
# -----------------------------
def predict_role(skills, resume_text):

    text = resume_text.lower()
    skills = [s.lower() for s in skills]

    if "machine learning" in text or "python" in text:
        return "Data Science / ML Engineer"

    elif "react" in text or "javascript" in text:
        return "Frontend Web Developer"

    elif "sql" in text or "excel" in text:
        return "Data Analyst"

    elif "teaching" in text or "teacher" in text:
        return "Teacher / Educator"

    elif "marketing" in text or "sales" in text:
        return "Marketing & Sales"

    else:
        return "General Technical Role"

# -----------------------------
# SMART CAREER SUGGESTIONS
# -----------------------------
def career_based_suggestions(role, skills, sections):

    suggestions = []

    skills = [s.lower() for s in skills]

    # Teacher Role
    if role == "Teacher / Educator":

        if "communication" not in skills:
            suggestions.append("Add communication skills because teaching roles highly value communication abilities")

        if not sections["Experience"]:
            suggestions.append("Try adding teaching experience, tutoring, or internship details")

        suggestions.append("Mention presentation, classroom handling, or mentoring skills")

    # Marketing Role
    elif role == "Marketing & Sales":

        if "communication" not in skills:
            suggestions.append("Marketing resumes should include communication skills for better recruiter impact")

        if not sections["Experience"]:
            suggestions.append("Add sales achievements or marketing campaign experience")

        suggestions.append("Include negotiation, leadership, and customer handling skills")

    # Data Science Role
    elif role == "Data Science / ML Engineer":

        if "machine learning" not in skills:
            suggestions.append("Add machine learning related projects and skills")

        suggestions.append("Include GitHub links and technical projects for better visibility")

    # Web Developer
    elif role == "Frontend Web Developer":

        suggestions.append("Add deployed project links and frontend technologies")

        if not sections["Projects"]:
            suggestions.append("Frontend resumes should include strong projects section")

    # General Suggestions
    suggestions.append("Add measurable achievements instead of general statements")

    return suggestions

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_description = st.text_area("Paste Job Description (Optional)")

# -----------------------------
# MAIN ANALYSIS
# -----------------------------
if uploaded_file:

    resume_text = extract_text(uploaded_file)

    # Extract Name
    user_name = extract_name(resume_text)

    # Greeting
    st.subheader(f" Hey! {user_name}, let's analyze your resume!")

    # Skills
    skills = extract_skills(resume_text)

    # Sections
    sections = check_sections(resume_text)

    # ATS Score
    score = calculate_ats_score(skills, sections, resume_text)

    # ATS Level
    level = ats_level(score)

    # Role Prediction
    role = predict_role(skills, resume_text)

    # Career Suggestions
    career_tips = career_based_suggestions(role, skills, sections)

    # -----------------------------
    # SCORE DISPLAY
    # -----------------------------
    st.markdown("---")

    st.subheader("ATS Score")

    st.progress(int(score))

    st.success(f"ATS Score: {score}%")

    st.info(f"Resume Quality: {level}")

    # -----------------------------
    # ROLE PREDICTION
    # -----------------------------
    st.subheader(" Predicted Role")
    st.write(role)

    # -----------------------------
    # SKILLS
    # -----------------------------
    st.subheader("Extracted Skills")

    if skills:
        st.write(skills)
    else:
        st.warning("No major skills detected")

    # -----------------------------
    # SECTION CHECK
    # -----------------------------
    st.subheader("Resume Sections")

    for section, status in sections.items():

        if status:
            st.success(f" {section} Found")

        else:
            st.error(f" {section} Missing")

    # -----------------------------
    # AI FEEDBACK
    # -----------------------------
    st.subheader("AI Resume Suggestions")

    feedback = generate_feedback(score, sections)

    for item in feedback:
        st.write("•", item)

    # -----------------------------
    # CAREER BASED IMPROVEMENTS
    # -----------------------------
    st.subheader(" Career-Based Improvement Suggestions")

    for tip in career_tips:
        st.write("✅", tip)

    # -----------------------------
    # JOB DESCRIPTION MATCH
    # -----------------------------
    if job_description:

        st.subheader(" Job Description Match")

        match_score = jd_match(resume_text, job_description)

        st.progress(int(match_score))

        st.success(f"Resume Match Score: {match_score}%")

        if match_score < 60:
            st.warning("Your resume needs more matching keywords for this job ")

        else:
            st.success("Your resume matches the job description well ")

    # -----------------------------
    # FINAL MESSAGE
    # -----------------------------
    motivational_lines = [
        f"{user_name}, your resume has potential ",
        f"Great start {user_name}! A few improvements can make your resume stronger ",
        f"{user_name}, recruiters love resumes with strong projects and skills ",
        f"Keep improving your resume {user_name}! You're getting closer to your dream role "
    ]

    st.markdown("---")

    st.success(random.choice(motivational_lines))