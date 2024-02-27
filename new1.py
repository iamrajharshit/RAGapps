import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader

# LangChain configuration
LLM = OpenAI(model_name="gpt-3.5-turbo",openai_api_key='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')  # Update with the correct LangChain model

def pdf_loader(uploaded_pdf):
    pdf_path = "temp_notebook.pdf"
    with open(pdf_path, "wb") as temp_file:
        temp_file.write(uploaded_pdf.getvalue())

    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()
    result_string = "\n\n".join([doc.page_content for doc in pages])
    return result_string

def get(pages):
    page = "".join(pages)
    print(page)

    # Using LangChain LLMChain for text generation
    prompt_template = """Question: {question}

    Answer: Let's think step by step."""
    
    llm_chain = LLMChain(prompt=PromptTemplate(template=prompt_template, input_variables=["question"]), llm=LLM)

    question = "explain and rephrase the text in an HTML code format, SEO optimized " + page
    response = llm_chain.run(question)

    st.code(response)

# Main Streamlit app
def main():
    st.title("PDF Processor App")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Process the PDF when the user clicks a button
        if st.button("Process PDF"):
            pages = pdf_loader(uploaded_file)
            get(pages)

if __name__ == "__main__":
    main()
