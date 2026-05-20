# ====================== app.py ======================

import streamlit as st
import PyPDF2
import pickle
import plotly.graph_objects as go
import language_tool_python

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide"
)

# ---------------- LOAD CSS ---------------- #

def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ---------------- LOAD ML MODEL ---------------- #

model = pickle.load(open("models/model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

# ---------------- LOAD GRAMMAR TOOL ---------------- #

tool = language_tool_python.LanguageTool('en-US')

# ---------------- SKILLS DATABASE ---------------- #

skills_db = {

    "data scientist": [
        "python",
        "machine learning",
        "sql",
        "tensorflow",
        "nlp",
        "deep learning",
        "statistics",
        "pandas"
    ],

    "web developer": [
        "html",
        "css",
        "javascript",
        "react",
        "node.js",
        "mongodb"
    ],

    "software engineer": [
        "java",
        "python",
        "c++",
        "sql",
        "git",
        "problem solving"
    ],

    "civil engineer": [
        "autocad",
        "construction",
        "site management",
        "structural design",
        "surveying"
    ],

    "mechanical engineer": [
        "autocad",
        "solidworks",
        "manufacturing",
        "thermodynamics",
        "machine design"
    ],

    "electrical engineer": [
        "circuit design",
        "power systems",
        "matlab",
        "electrical maintenance"
    ],

    "doctor": [
        "patient care",
        "diagnosis",
        "medical knowledge",
        "communication",
        "clinical experience"
    ],

    "nurse": [
        "patient care",
        "medical support",
        "communication",
        "critical care"
    ],

    "teacher": [
        "communication",
        "classroom management",
        "presentation",
        "subject knowledge"
    ],

    "accountant": [
        "tally",
        "gst",
        "taxation",
        "excel",
        "financial reporting"
    ],

    "bba": [
        "management",
        "communication",
        "marketing",
        "leadership",
        "business strategy"
    ],

    "marketing": [
        "seo",
        "social media",
        "branding",
        "content marketing",
        "communication"
    ],

    "hr": [
        "recruitment",
        "communication",
        "employee management",
        "leadership"
    ],

    "agriculture": [
        "crop management",
        "soil science",
        "farming",
        "irrigation"
    ],

    "carpenter": [
        "woodworking",
        "furniture design",
        "cutting",
        "measurement"
    ],

    "arts": [
        "creativity",
        "design",
        "communication",
        "presentation"
    ],

    "commerce": [
        "accounting",
        "finance",
        "business studies",
        "economics"
    ]
}

# ---------------- PDF EXTRACTION ---------------- #

def extract_text_from_pdf(uploaded_file):

    reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text

# ---------------- GRAMMAR CHECK ---------------- #

def grammar_check(text):

    matches = tool.check(text)

    return matches[:10]

# ---------------- HEADER ---------------- #

st.title("AI Resume Analyzer")

st.caption(
    "Professional ATS Resume Evaluation Platform"
)

st.markdown("---")
# ---------------- SIDEBAR ---------------- #

st.sidebar.title("AI Dashboard")

st.sidebar.write(
    "Analyze resumes with AI-powered ATS evaluation."
)

st.sidebar.markdown("---")

st.sidebar.metric("AI Model", "Active")
st.sidebar.metric("ATS Engine", "Running")
st.sidebar.metric("Supported Careers", "15+")

st.sidebar.markdown("---")

st.sidebar.info(
    "Supported Fields:\n\n"
    "Technology\n"
    "Medical\n"
    "Engineering\n"
    "Commerce\n"
    "Agriculture\n"
    "Arts\n"
    "Management"
)

# ---------------- INPUTS ---------------- #

job_role = st.text_input(
    "Target Job Role",
    placeholder="Example: Data Scientist"
)

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# ---------------- MAIN ANALYSIS ---------------- #

if uploaded_file is not None:

    resume_text = extract_text_from_pdf(uploaded_file)

    st.success("Resume uploaded successfully.")

    col1, col2 = st.columns([2, 1])

    # ---------------- LEFT SIDE ---------------- #

    with col1:

        st.subheader("Extracted Resume Content")

        st.text_area(
            "Resume Text",
            resume_text,
            height=500
        )

    # ---------------- RIGHT SIDE ---------------- #

    with col2:

        st.subheader("Resume Analysis")

        if st.button("Analyze Resume"):

            # ---------------- ML ATS SCORE ---------------- #

            combined_text = resume_text + " " + job_role

            vector = tfidf.transform([combined_text])

            prediction = model.predict_proba(vector)[0][1]

            ats_score = int(prediction * 100)

            # ---------------- ATS GAUGE ---------------- #

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ats_score,

                title={
                    'text': "ATS Score"
                },

                gauge={
                    'axis': {
                        'range': [0, 100]
                    },

                    'bar': {
                        'color': "#06b6d4"
                    },

                    'steps': [

                        {
                            'range': [0, 50],
                            'color': "#ef4444"
                        },

                        {
                            'range': [50, 75],
                            'color': "#f59e0b"
                        },

                        {
                            'range': [75, 100],
                            'color': "#22c55e"
                        }
                    ]
                }
            ))

            fig.update_layout(
                paper_bgcolor="#111827",
                font={'color': "white"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ---------------- MISSING SKILLS ---------------- #

            st.markdown("---")

            st.subheader("Missing Skills")

            resume_lower = resume_text.lower()

            required_skills = skills_db.get(
                job_role.lower(),
                []
            )

            missing_skills = []

            for skill in required_skills:

                if skill not in resume_lower:

                    missing_skills.append(skill)

            if missing_skills:

                for skill in missing_skills:

                    st.warning(skill)

            else:

                st.success("No major skills missing.")

            # ---------------- GRAMMAR CHECK ---------------- #

            st.markdown("---")

            st.subheader("Grammar Suggestions")

            grammar_errors = grammar_check(resume_text)

            if grammar_errors:

                for error in grammar_errors:

                    st.warning(error.message)

            else:

                st.success(
                    "No major grammar issues detected."
                )

            # ---------------- RESUME STRENGTH ---------------- #

            st.markdown("---")

            st.subheader("Resume Strength")

            strength = 0

            if "project" in resume_lower:
                strength += 20

            if "experience" in resume_lower:
                strength += 20

            if "skills" in resume_lower:
                strength += 20

            if "education" in resume_lower:
                strength += 20

            if ats_score > 70:
                strength += 20

            st.progress(strength / 100)

            st.write(f"Resume Strength: {strength}%")

            # ---------------- SUGGESTIONS ---------------- #

            st.markdown("---")

            st.subheader("Resume Suggestions")

            if ats_score < 50:

                st.error(
                    "Add more relevant skills, certifications, and projects."
                )

                st.info(
                    "Include measurable achievements and role-specific keywords."
                )

            elif ats_score < 75:

                st.warning(
                    "Resume is moderately optimized."
                )

                st.info(
                    "Improve project descriptions and technical skills."
                )

            else:

                st.success(
                    "Resume appears optimized for ATS systems."
                )

                st.info(
                    "Maintain strong formatting and concise descriptions."
                )

            # ---------------- ROLE RECOMMENDATIONS ---------------- #

            st.markdown("---")

            st.subheader("Recommended Resume Additions")

            recommendations = {

                "data scientist": [
                    "Add Machine Learning projects",
                    "Include SQL and Python certifications",
                    "Mention data visualization tools"
                ],

                "web developer": [
                    "Add portfolio links",
                    "Include React or Node.js projects",
                    "Mention frontend frameworks"
                ],

                "doctor": [
                    "Add clinical experience",
                    "Mention certifications",
                    "Include patient care achievements"
                ],

                "civil engineer": [
                    "Mention AutoCAD expertise",
                    "Add construction/site projects",
                    "Include surveying knowledge"
                ]
            }

            role_tips = recommendations.get(
                job_role.lower(),
                ["Add role-specific achievements and certifications."]
            )

            for tip in role_tips:

                st.info(tip)