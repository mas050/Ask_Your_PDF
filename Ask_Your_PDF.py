import streamlit as st
from PyPDF2 import PdfReader
import requests
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq
import os

# Groq API credentials
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Choice of LLM
AVAILABLE_MODELS = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
model_selected = "llama3-70b-8192"

# LLM Instruction so it focus on the documents uploaded
llm_instructions = f"""
    You are a agent that is specialized in answering user's question about the documents they provide to you. \
    First you will get the most relevant information related to their question that we call "Context".\
    Your task will be to answer the user's question based on this context only.\
    If you cannot answer the user's question based on the information provided in the "context", simply say\
    that the document do not contain the information related to their question.\
    Format your answer so it is easy to read for the user.\
    
    Here are the context and the user question:\
"""

#Output your answer so it outline clearly the elements the user want to understand and provide a bullet points summary.\

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to vectorize text chunks
def vectorize_chunks(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    return vectors, vectorizer

# Function to find the most similar chunk to a query
def find_most_similar_chunk(query, chunks, vectors, vectorizer):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors)
    most_similar_index = np.argmax(similarities)
    return chunks[most_similar_index]

# Function to interact with Groq's LLM
def get_llm_response(prompt,model_input):

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_input,
        temperature=0,
        max_tokens=25000,
        top_p=1
    )
    return response.choices[0].message.content

# Streamlit app
st.title("Ask your PDF in Private")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        combined_text = ""
        for uploaded_file in uploaded_files:
            pdf_text = extract_text_from_pdf(uploaded_file)
            combined_text += pdf_text + "\n"

        chunks = chunk_text(combined_text)
        vectors, vectorizer = vectorize_chunks(chunks)
    st.success("PDFs uploaded and processed successfully!")

    user_question = st.text_input("Ask a question based on the PDF content:")

    if user_question:
        with st.spinner("Fetching response..."):
            most_similar_chunk = find_most_similar_chunk(user_question, chunks, vectors, vectorizer)
            response = get_llm_response(llm_instructions + "\n\nContext: " + most_similar_chunk + "\n\nUser Question: " + user_question, model_selected)
            st.write("Answer:", response)
