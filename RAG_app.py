import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

from dotenv import load_dotenv
import os

load_dotenv()

import streamlit as st
import fitz  # PyMuPDF



import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
#new version
#from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma


warnings.filterwarnings("ignore")
# restart python kernal if issues with langchain import.
from langchain_google_genai import ChatGoogleGenerativeAI

#Models
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.environ["API_KEY"],temperature=0.7,convert_system_message_to_human=True)

#pdf
def pdf_loader(uploaded_pdf):
    #make a temp file and pass as PyPDFLoader() accepts only location 
    pdf_path = f"temp_notebook.pdf"
    with open(pdf_path, "wb") as temp_file:
        temp_file.write(uploaded_pdf.getvalue())

    pdf_loader = PyPDFLoader(pdf_path)
  
    pages = pdf_loader.load_and_split()
    return pages


from langchain_google_genai import GoogleGenerativeAIEmbeddings
def splitter(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    return texts


def embed(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ["API_KEY"])
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    return vector_index


def qna(vector_index,question):
    template = """Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer, replay Your Welcome! only when Thank you is said.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )


    ques = question
    result = qa_chain({"query": ques})

    return result['result']



#streamlit app

def main():
    st.title("Chat with PDF")

    uploaded_file = st.file_uploader(" ### Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write("### PDF Preview:")
        pages=pdf_loader(uploaded_file)

        texts=splitter(pages)

        vector_index=embed(texts)

        

        question=st.text_area("## Ask Ques")
        if st.button("Send"):

            ans=qna(vector_index,question)

            st.markdown(f"## Ans:\n\n{ans}", unsafe_allow_html=True)


if __name__ == "__main__":
    main()