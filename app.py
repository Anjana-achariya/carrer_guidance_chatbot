#!/usr/bin/env python
# coding: utf-8

# In[2]:


from functions import preprocess,extract_file,predict_roles


# In[3]:


import streamlit as st


# In[4]:


import json
import joblib


# In[ ]:


le = joblib.load("encoder.pkl")

with open("nbmodel.pkl","rb") as f:
  model_nb = joblib.load(f)

with open("vectorizer.pkl","rb") as f:
  tf_idf = joblib.load(f)

with open("skills.json","r") as f:
  skills = json.load(f)

st.title("carrer guidance chatbot")

uploaded_file = st.file_uploader("upload your resume " , type =["pdf ", "doc" ,"docx"])
if uploaded_file is not None:
  resume_text = extract_file(uploaded_file)
  preprocessed_text = preprocess(resume_text)

  top_roles = predict_roles(model_nb,tf_idf,preprocessed_text)
  decoded_roles = {le.inverse_transform([int(role)])[0]: prob for role, prob in top_roles.items()}  

  st.subheader(" top 5 role suggestions:")

  for role,prob in decoded_roles.items():
    st.write(f"{role} ({prob*100:.2f}%)")

  selected_role = st.selectbox("Do you want to see required skills for the roles?" , 
                               list(decoded_roles.keys()))


  if selected_role:
    required_skills = skills.get(selected_role,[])
    st.write(f"**Skills for {selected_role}:**")
    st.write(", ".join(required_skills) if required_skills else "No skill data available.")


# In[ ]:




