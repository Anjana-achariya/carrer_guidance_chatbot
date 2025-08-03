#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import joblib
import json
from functions import extract_file, preprocess, predict_roles
from rapidfuzz import fuzz
import re


# In[10]:


le = joblib.load("encoder.pkl")
with open("nbmodel.pkl", "rb") as f:
    model_nb = joblib.load(f)
with open("vectorizer.pkl", "rb") as f:
    tf_idf = joblib.load(f)
with open("skills.json", "r") as f:
    skills = json.load(f)
with open("job_salaries.json", "r") as f:
    salary_data = json.load(f)


# In[ ]:


st.title("Career Guidance Chatbot")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "doc", "docx"])

if uploaded_file is not None:
    resume_text = extract_file(uploaded_file)
    preprocessed_text = preprocess(resume_text)

    top_roles = predict_roles(model_nb, tf_idf, preprocessed_text)
    decoded_roles = {le.inverse_transform([int(role)])[0]: prob for role, prob in top_roles.items()}

    st.subheader("Top 5 Role Suggestions:")
    for role, prob in decoded_roles.items():
        st.write(f"{role} ({prob*100:.2f}%)")


    selected_role = st.selectbox("🤓 Do you want to see required skills for the roles?", list(decoded_roles.keys()))
    if selected_role:
        required_skills = skills.get(selected_role, [])
        st.write(f"**Skills for {selected_role}:**")
        st.write(", ".join(required_skills) if required_skills else "No skill data available.")

st.markdown("---")
st.subheader("🤔 Ask about Job Salaries")
user_input = st.text_input("Type your salary query:")

if user_input:
    msg = user_input.lower()

    matched_role = None
    highest_score = 0
    for role in salary_data:
        score = fuzz.partial_ratio(msg, role.lower())
        if score > highest_score and score > 60:
            matched_role = role
            highest_score = score

    if matched_role:
        role_salary = salary_data[matched_role]

        if "min" in msg or "minimum" in msg:
            st.success(f" Minimum salary for **{matched_role}** is ₹{min(s['min'] for s in role_salary.values()):,}")

        elif "max" in msg or "maximum" in msg:
            st.success(f" Maximum salary for **{matched_role}** is ₹{max(s['max'] for s in role_salary.values()):,}")

        elif "intern" in msg or "fresher" in msg:
            sal = role_salary.get("Intern") or role_salary.get("Fresher")
            if sal:
                st.success(f"Salary for **{matched_role} Intern/Fresher**: ₹{sal['min']:,} - ₹{sal['max']:,}")
            else:
                st.warning("No intern/fresher data available for this role.")

        elif match := re.search(r"(\d)[^\d]*(?:to|–|-)?[^\d]*(\d)?\s*years?", msg):
            exp1 = int(match.group(1))
            exp2 = int(match.group(2)) if match.group(2) else exp1
            found = False
            for exp_range, sal in role_salary.items():
                nums = re.findall(r"\d+", exp_range)
                if nums and len(nums) == 2:
                    low, high = map(int, nums)
                    if low <= exp1 <= high:
                        st.success(f"💼 Salary for **{matched_role}** with **{exp_range}** experience: ₹{sal['min']:,} - ₹{sal['max']:,}")
                        found = True
                        break
            if not found:
                st.warning("Couldn't find salary data for that experience level.")

        else:
            st.info(f"💼 Salary ranges for **{matched_role}**:")
            for level, sal in role_salary.items():
                st.write(f"- **{level}**: ₹{sal['min']:,} - ₹{sal['max']:,}")
    else:
        st.error("Sorry, I couldn't detect the job role. Try rephrasing.")


# In[ ]:




