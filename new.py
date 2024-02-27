import streamlit as st
#from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader

# Your API_KEY and other imports here...
#import pandas as pd
#from langchain import PromptTemplate
#from langchain.prompts import PromptTemplate
#from langchain.chains.question_answering import load_qa_chain
#new version
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
#from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai

API_KEY = "AIzaSyAezKZT5ODtVc6bczVqg2FWQZ2YSIerbbY"
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY, temperature=0.8, convert_system_message_to_human=True)

def pdf_loader(uploaded_pdf):
    pdf_path = "temp_notebook.pdf"
    with open(pdf_path, "wb") as temp_file:
        temp_file.write(uploaded_pdf.getvalue())

    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()
    #print(pages)
    result_string = "\n\n".join([doc.page_content for doc in pages])
    #print(result_string)
    return result_string


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}



def get(pages):
     
    page="".join(pages)
    print(page)

    response=model.invoke("explian and rephrase the text in a html code format, seo optimized" +page)
    #st.code(response.content)
    #genai.configure(api_key=API_KEY)
    #model = genai.GenerativeModel('gemini-pro')
    #response = model.generate_content("explian and rephrase the text in a html code format, seo optimized" +page)
    st.code(response.content)

    #response1=model.invoke(response.content + "clean the code and remove \n")
    #st.code(response1.content)



# Main Streamlit app
def main():


    st.title("PDF Processor App")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Process the PDF when the user clicks a button
        if st.button("Process PDF"):
            pages=pdf_loader(uploaded_file)

            get(pages)



if __name__ == "__main__":
    main()
