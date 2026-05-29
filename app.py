import streamlit as st
import pandas as pd
import pickle
import pdfplumber
import re

from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {

    background-image:
    linear-gradient(
        rgba(8,12,25,0.94),
        rgba(8,12,25,0.94)
    ),
    url("https://images.unsplash.com/photo-1516321318423-f06f85e504b3");

    background-size: cover;

    background-position: center;

    background-attachment: fixed;

    color: white;
}

/* HEADINGS */

h1, h2, h3, h4 {

    color: white;
}

/* CARDS */

div[data-testid="stVerticalBlock"] > div {

    background: rgba(18, 24, 38, 0.82);

    padding: 22px;

    border-radius: 18px;

    border: 1px solid rgba(255,255,255,0.08);

    margin-bottom: 18px;

    box-shadow: 0px 4px 20px rgba(0,0,0,0.35);
}

/* BUTTON */

.stButton>button {

    background: linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );

    color: white;

    border-radius: 10px;

    border: none;

    height: 3em;

    font-size: 16px;

    width: 100%;
}

/* INPUT BOX */

.stTextInput>div>div>input {

    background-color: rgba(20,25,40,0.9);

    color: white;

    border-radius: 10px;
}

/* FILE UPLOADER */

section[data-testid="stFileUploader"] {

    background: rgba(20,25,40,0.8);

    padding: 15px;

    border-radius: 15px;
}

/* PROGRESS BAR */

div.stProgress > div > div > div > div {

    background: linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );
}

/* TEXT */

p, label, div {

    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(
    open("models/model.pkl", "rb")
)

tfidf = pickle.load(
    open("models/tfidf.pkl", "rb")
)

# -----------------------------
# LOAD JOB DATASET
# -----------------------------
jobs = pd.read_csv(
    "datasets/job_dataset.csv"
)

jobs = jobs.fillna("")

# -----------------------------
# SMART ROLE SKILLS DATABASE
# -----------------------------
role_skill_map = {

    "doctor": [
        "MBBS",
        "Patient Care",
        "Medical Knowledge",
        "Diagnosis",
        "Clinical Skills",
        "Communication"
    ],

    "teacher": [
        "Communication",
        "Presentation",
        "Mentoring",
        "Classroom Management",
        "Leadership"
    ],

    "sales": [
        "Sales",
        "Negotiation",
        "Communication",
        "Customer Handling",
        "Marketing"
    ],

    "marketing": [
        "SEO",
        "Digital Marketing",
        "Social Media",
        "Branding",
        "Communication"
    ],

    "software engineer": [
        "Python",
        "Java",
        "SQL",
        "Git",
        "Problem Solving"
    ],

    "data scientist": [
        "Python",
        "Machine Learning",
        "SQL",
        "Data Analysis",
        "Pandas",
        "Scikit Learn"
    ],

    "web developer": [
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Node.js"
    ],

    "designer": [
        "Photoshop",
        "Creativity",
        "UI UX",
        "Figma",
        "Communication"
    ]
}

# -----------------------------
# APP TITLE
# -----------------------------
st.title(
    "AI Resume Analyzer & Career Guidance System"
)

st.write(
    "Upload your resume and let AI analyze "
    "your ATS score, career suitability, "
    "missing skills, and job compatibility."
)

# -----------------------------
# USER TARGET ROLE
# -----------------------------
selected_role = st.text_input(
    "Enter Your Desired Job Role",
    placeholder="Example: Doctor, Teacher, Data Scientist"
)

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

# -----------------------------
# EXTRACT PDF TEXT
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
# CLEAN TEXT
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9 ]',
        ' ',
        text
    )

    return text

# -----------------------------
# GET ROLE SKILLS
# -----------------------------
def get_role_skills(role):

    role = role.lower()

    for key in role_skill_map:

        if key in role:

            return role_skill_map[key]

    return [
        "Communication",
        "Problem Solving",
        "Teamwork"
    ]

# -----------------------------
# JOB MATCHING
# -----------------------------
def calculate_job_match(resume_text, jobs):

    jobs["combined"] = (

        jobs["Title"].astype(str) + " " +

        jobs["Skills"].astype(str) + " " +

        jobs["Keywords"].astype(str) + " " +

        jobs["Responsibilities"].astype(str)
    )

    # JOB VECTORS
    job_vectors = tfidf.transform(
        jobs["combined"]
    )

    # RESUME VECTOR
    resume_vector = tfidf.transform(
        [resume_text]
    )

    # COSINE SIMILARITY
    similarity = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]

    jobs["match_score"] = similarity * 100

    # TOP JOBS
    top_jobs = jobs.sort_values(
        by="match_score",
        ascending=False
    ).head(3)

    return top_jobs

# -----------------------------
# MAIN APP
# -----------------------------
if uploaded_file is not None:

    if selected_role == "":

        st.warning(
            "Please enter your desired job role."
        )

    else:

        # EXTRACT RESUME TEXT
        resume_text = extract_text(
            uploaded_file
        )

        cleaned_resume = clean_text(
            resume_text
        )

        # NAME EXTRACTION
        lines = resume_text.split("\n")

        user_name = lines[0]

        # VECTORIZE
        resume_vector = tfidf.transform(
            [cleaned_resume]
        )

        # PREDICTION
        prediction = model.predict(
            resume_vector
        )[0]

        # JOB MATCHING
        top_jobs = calculate_job_match(
            cleaned_resume,
            jobs
        )

        best_job = top_jobs.iloc[0]

        match_score = round(
            best_job["match_score"],
            2
        )

        # REQUIRED SKILLS
        required_skills = get_role_skills(
            selected_role
        )

        found_skills = []

        missing_skills = []

        # ATS SCORE
        score = 50

        for skill in required_skills:

            if skill.lower() in cleaned_resume:

                found_skills.append(skill)

                score += 8

            else:

                missing_skills.append(skill)

        if score > 100:

            score = 100

        # -----------------------------
        # HEADER
        # -----------------------------
        st.success(
            f"Hello {user_name}, "
            f"your AI resume analysis is ready."
        )

        # -----------------------------
        # DASHBOARD
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        # CAREER DOMAIN
        with col1:

            st.subheader(
                "Detected Career Domain"
            )

            st.success(prediction)

        # ATS SCORE
        with col2:

            st.subheader("ATS Score")

            st.progress(score / 100)

            st.write(f"{score}%")

        # JOB MATCH
        with col3:

            st.subheader(
                "Job Compatibility"
            )

            st.progress(match_score / 100)

            st.write(f"{match_score}%")

        # -----------------------------
        # TARGET ROLE
        # -----------------------------
        st.subheader(
            "Target Role"
        )

        st.write(selected_role)

        # -----------------------------
        # BEST MATCHED CAREER
        # -----------------------------
        st.subheader(
            "Best Matched Career"
        )

        st.write(best_job["Title"])

        # -----------------------------
        # RECOMMENDED CAREERS
        # -----------------------------
        st.subheader(
            "Recommended Careers"
        )

        for index, row in top_jobs.iterrows():

            st.write(
                f"• {row['Title']} "
                f"({round(row['match_score'], 2)}% Match)"
            )

        # -----------------------------
        # REQUIRED SKILLS
        # -----------------------------
        st.subheader(
            "Skills Required For Your Desired Role"
        )

        for skill in required_skills:

            st.write(f"• {skill}")

        # -----------------------------
        # DETECTED SKILLS
        # -----------------------------
        st.subheader(
            "Skills Detected In Resume"
        )

        if len(found_skills) > 0:

            for skill in found_skills:

                st.write(f"✔ {skill}")

        else:

            st.warning(
                "Your resume currently lacks strong keywords "
                "for this target role."
            )

        # -----------------------------
        # MISSING SKILLS
        # -----------------------------
        st.subheader(
            "Skill Gap Analysis"
        )

        if len(missing_skills) > 0:

            st.warning(
                f"To become a successful "
                f"{selected_role}, "
                f"you should improve these skills:"
            )

            for skill in missing_skills:

                st.write(f"• {skill}")

        else:

            st.success(
                "Your resume already contains "
                "most required skills."
            )

        # -----------------------------
        # AI CAREER RECOMMENDATION
        # -----------------------------
        st.subheader(
            "AI Career Recommendation"
        )

        if selected_role.lower() in prediction.lower():

            st.success(
                f"Your resume aligns well "
                f"with the {selected_role} role."
            )

        else:

            st.warning(
                f"Your resume does not strongly "
                f"match the {selected_role} role."
            )

            st.info(
                f"Based on your resume and skills, "
                f"you are more suitable for "
                f"{prediction} related careers."
            )

            st.write(
                "Better career options for you:"
            )

            for index, row in top_jobs.iterrows():

                st.write(
                    f"• {row['Title']}"
                )

        # -----------------------------
        # ACHIEVEMENT ANALYSIS
        # -----------------------------
        st.subheader(
            "Achievement Analysis"
        )

        numbers = re.findall(
            r'\d+%?',
            resume_text
        )

        if len(numbers) > 0:

            st.success(
                "Your resume includes measurable achievements."
            )

        else:

            st.warning(
                "Add measurable achievements like:"
            )

            st.write(
                "• Increased sales by 20%"
            )

            st.write(
                "• Improved efficiency by 15%"
            )

            st.write(
                "• Managed a team of 10 members"
            )

        # -----------------------------
        # ATS TIPS
        # -----------------------------
        st.subheader(
            "ATS Resume Tips"
        )

        st.write(
            "• Use standard section headings"
        )

        st.write(
            "• Add measurable achievements"
        )

        st.write(
            "• Include role-specific keywords"
        )

        st.write(
            "• Keep formatting simple"
        )

        st.write(
            "• Avoid too many graphics or tables"
        )

        # -----------------------------
        # FINAL FEEDBACK
        # -----------------------------
        st.subheader(
            "Final AI Feedback"
        )

        if score >= 80 and match_score >= 70:

            st.success(
                "Your resume is highly optimized "
                "and professionally aligned "
                "for industry-level recruitment systems."
            )

        elif score >= 60:

            st.warning(
                "Your resume has good potential "
                "but still requires improvement "
                "in skills and ATS optimization."
            )

        else:

            st.error(
                "Your resume currently has low "
                "compatibility for your desired role."
            )