#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))


# In[2]:


def preprocess(resume):
    resume = re.sub("[^a-zA-Z]"," ",resume)
    text = resume.lower()
    text = text.split()
    text = [words for words in text if words not in sw]
    text = " ".join(text)
    return text



# In[3]:


import json


# In[4]:


with open("skills.json") as f:
    job_skills = json.load(f)


# In[ ]:


job_skills


# In[6]:


import joblib
model = joblib.load("nbmodel.pkl")
vect = joblib.load("vectorizer.pkl")
encoder = joblib.load("encoder.pkl")


# In[10]:


import fitz
import docx2txt
import string
import textract


# In[12]:
try:
    import textract
except ImportError:
    textract = None

def extract_file(resume):
    fn = resume.name.lower()
    with open(fn , "wb") as f:
        f.write(resume.read())

    if fn.endswith(".pdf"):
        try:
            pdf_file = fitz.open(fn)
            text = ""
            for page in pdf_file:
                text+=page.get_text()
            return text
        except Exception as e:
            return f"error reading the file {e}"

    elif fn.endswith(".docx"):
        try:
            docxfile = docx2txt.process(fn)
            return docxfile
        except Exception as e:
            return f"error reading the file {e}"

    elif fn.endswith(".doc"):
        try:
            return textract.process(fn).decode("utf-8")
        except Exception as e:
            return f"error reading the file {e}"

    else:
        return "Unsupported file format. Please upload a PDF, DOCX, or DOC file."


# In[13]:


def predict_roles(model,vectorizer,text,top_n=5):
  vec = vect.transform([text])
  probs = model.predict_proba(vec)[0]
  classes = model.classes_
  top_indices = probs.argsort()[-top_n:][::-1]
  top_classes = classes[top_indices]
  top_probs = probs[top_indices]
  return dict(zip(top_classes,top_probs))


# In[ ]:





