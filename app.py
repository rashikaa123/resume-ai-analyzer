# ====================== app.py ======================

import streamlit as st
import PyPDF2
import pickle
import plotly.graph_objects as go
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG ================= #

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ================= LOAD CSS ================= #

def load_css():

    with open("assets/style.css") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ================= LOAD MODEL ================= #

model = pickle.load(
    open("models/model.pkl", "rb")
)

tfidf = pickle.load(
    open("models/tfidf.pkl", "rb")
)

# ================= OCCUPATION SKILLS DATABASE ================= #

skills_db = {

    # TECH

    "software engineer": [
        "python", "java", "sql", "git", "api"
    ],

    "web developer": [
        "html", "css", "javascript", "react"
    ],

    "data scientist": [
        "machine learning", "python", "pandas", "sql"
    ],

    "cyber security": [
        "linux", "network security", "firewall"
    ],

    # ENGINEERING

    "civil engineer": [
        "autocad", "construction", "surveying"
    ],

    "mechanical engineer": [
        "machine design", "manufacturing"
    ],

    "electrical engineer": [
        "circuits", "matlab", "power systems"
    ],

    # MEDICAL

    "doctor": [
        "patient care", "diagnosis"
    ],

    "nurse": [
        "patient care", "medical assistance"
    ],

    "pharmacist": [
        "medicine", "drug knowledge"
    ],

    # BUSINESS

    "accountant": [
        "gst", "taxation", "excel"
    ],

    "marketing": [
        "seo", "branding"
    ],

    "hr": [
        "recruitment", "management"
    ],

    "sales": [
        "communication", "negotiation"
    ],

    # CREATIVE

    "graphic designer": [
        "photoshop", "illustrator"
    ],

    "video editor": [
        "editing", "premiere pro"
    ],

    "artist": [
        "creativity", "drawing"
    ],

    "painter": [
        "wall painting", "finishing"
    ],

    # SKILLED WORK

    "carpenter": [
        "woodworking", "furniture"
    ],

    "plumber": [
        "pipe fitting", "maintenance"
    ],

    "electrician": [
        "wiring", "electrical repair"
    ],

    "welder": [
        "metal fabrication", "welding"
    ],

    "driver": [
        "driving", "vehicle maintenance"
    ],

    "mechanic": [
        "repair", "engine"
    ],

    # EDUCATION

    "teacher": [
        "communication", "presentation"
    ],

    "professor": [
        "research", "education"
    ],

    # AGRICULTURE

    "farmer": [
        "farming", "soil", "crop management"
    ],

    "agriculture": [
        "crop production", "soil science"
    ],

    # HOTEL

    "chef": [
        "cooking", "food preparation"
    ],

    "waiter": [
        "customer service"
    ],

    "hotel manager": [
        "management", "hospitality"
    ]
}

# ================= PDF TEXT EXTRACTION ================= #

def extract_text_from_pdf(uploaded_file):

    reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted

    return text

# ================= SPELL CHECK ================= #

def check_spelling(text):

    try:

        blob = TextBlob(text)

        corrected = str(blob.correct())

        return corrected

    except:

        return text

# ================= HEADER ================= #

st.title("AI Resume Analyzer")

st.markdown(
    """
    <div class="subtitle">
    Professional ATS Resume Evaluation Platform
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ================= TOP METRICS ================= #

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("AI Model", "Active")

with m2:
    st.metric("ATS Engine", "Running")

with m3:
    st.metric("Occupations", "30+")

st.markdown("---")

# ================= CENTERED INPUT SECTION ================= #

center1, center2, center3 = st.columns([1, 6, 1])

with center2:

    st.subheader("Upload Resume")

    job_role = st.text_input(
        "Target Occupation / Job Role",
        placeholder="Doctor, Carpenter, Software Engineer, Painter..."
    )

    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

# ================= ANALYSIS ================= #

if uploaded_file is not None:

    resume_text = extract_text_from_pdf(
        uploaded_file
    )

    st.success(
        "Resume uploaded successfully."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= CENTER CONTENT ================= #

    left, main, right = st.columns([1, 8, 1])

    with main:

        # ================= RESUME CONTENT ================= #

        st.subheader("Resume Content")

        st.text_area(
            "Extracted Resume Text",
            resume_text,
            height=300
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ================= ANALYZE BUTTON ================= #

        if st.button("Analyze Resume"):

            combined_text = (
                resume_text + " " + job_role
            )

            vector = tfidf.transform(
                [combined_text]
            )

            prediction = model.predict_proba(
                vector
            )[0][1]

            ats_score = int(prediction * 100)

            # ================= AI ANALYSIS ================= #

            st.markdown("---")

            st.subheader("AI Resume Analysis")

            # ================= ATS SCORE ================= #

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

                font={
                    'color': "white"
                },

                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ================= MISSING SKILLS ================= #

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

                st.success(
                    "No major skills missing."
                )

            # ================= SPELLING ================= #

            st.markdown("---")

            st.subheader(
                "Spelling Suggestions"
            )

            corrected_text = check_spelling(
                resume_text
            )

            if corrected_text != resume_text:

                st.info(
                    "Possible spelling improvements detected."
                )

                st.text_area(
                    "Corrected Resume",
                    corrected_text,
                    height=200
                )

            else:

                st.success(
                    "No major spelling issues found."
                )

            # ================= RESUME STRENGTH ================= #

            st.markdown("---")

            st.subheader(
                "Resume Strength"
            )

            strength = 0

            keywords = [
                "project",
                "experience",
                "skills",
                "education",
                "certificate"
            ]

            for word in keywords:

                if word in resume_lower:

                    strength += 20

            st.progress(
                strength / 100
            )

            st.write(
                f"Resume Strength: {strength}%"
            )

            # ================= IMPROVEMENTS ================= #

            st.markdown("---")

            st.subheader(
                "Improvement Suggestions"
            )

            if ats_score < 50:

                st.error(
                    "Add more role-specific skills and projects."
                )

                st.warning(
                    "Resume needs stronger ATS keywords."
                )

            elif ats_score < 75:

                st.warning(
                    "Resume is moderately optimized."
                )

                st.info(
                    "Improve technical skills and achievements."
                )

            else:

                st.success(
                    "Resume is highly optimized."
                )

                st.info(
                    "Maintain concise professional formatting."
                )

            # ================= FINAL RECOMMENDATION ================= #

            st.markdown("---")

            st.subheader(
                "Final AI Recommendation"
            )

            st.info(
                "Use measurable achievements, certifications, and role-specific keywords to improve hiring chances."
            )