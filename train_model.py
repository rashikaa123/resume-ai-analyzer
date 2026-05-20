# ====================== train_model.py ======================

import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {

    "resume": [

        "Python machine learning data analysis SQL projects",

        "HTML CSS JavaScript React frontend development",

        "Java Spring Boot backend APIs MySQL",

        "Python deep learning NLP TensorFlow AI",

        "Excel Power BI SQL analytics dashboard",

        "AutoCAD construction site management structural design",

        "Patient care diagnosis communication clinical experience",

        "Woodworking furniture cutting measurement",

        "Marketing branding SEO communication",

        "Accounting GST taxation financial reporting"
    ],

    "job_description": [

        "Data Scientist Python SQL Machine Learning",

        "Frontend Web Developer React JavaScript",

        "Backend Java Developer Spring Boot",

        "AI Engineer Deep Learning NLP",

        "Data Analyst Power BI SQL",

        "Civil Engineer AutoCAD Construction",

        "Doctor Clinical Patient Care",

        "Carpenter Furniture Woodworking",

        "Marketing SEO Branding",

        "Accountant GST Taxation"
    ],

    "label": [1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

df["combined_text"] = (
    df["resume"] + " " + df["job_description"]
)

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df["combined_text"])

y = df["label"]

model = LogisticRegression()

model.fit(X, y)

pickle.dump(
    model,
    open("models/model.pkl", "wb")
)

pickle.dump(
    tfidf,
    open("models/tfidf.pkl", "wb")
)

print("Model trained successfully.")