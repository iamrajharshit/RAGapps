#api_key
API_KEY ="AIzaSyAezKZT5ODtVc6bczVqg2FWQZ2YSIerbbY"

import streamlit as st
import nbformat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

def write(uploaded_file):
    notebook_path = f"temp_notebook.ipynb"
    with open(notebook_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())
    return notebook_path    


def extract_code_from_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

        code_cells = []
        for cell in notebook_content['cells']:
            if cell['cell_type'] == 'code':
                code_cells.append(cell['source'])
        print(code_cells)
        return code_cells


def process_ccells(cells):
     # Initialize Langchain with Gemini Pro
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=API_KEY,temperature=0.5,convert_system_message_to_human=True)
    # Process code cells individually with RetrievalQA
    for cell_code in cells:
        retrieval_qa = RetrievalQA(llm)  # Create a new RetrievalQA chain for each cell
        retrieval_qa.add_node(cell_code, "code")
    return retrieval_qa


def main():
    st.title("Code Explainer")

    uploaded_file = st.file_uploader("Choose a Jupyter Notebook file", type=["ipynb"])

    if uploaded_file is not None:
        st.write("### Code Cells:")
        
        #to save the code file gen. path
        notebook_path=write(uploaded_file)

        #list of code cells
        code_cells = extract_code_from_notebook(notebook_path)

        retrieval_qa=process_ccells(code_cells)

        st.title("Code Analysis with Langchain & Gemini Pro")

        for i, code_cell in enumerate(code_cells, start=1):
            st.write(f"**Code Cell {i}:**")
            st.code(code_cell, language="python")
            st.markdown("---")


        col1,col2, col3= st.columns(3)
        with col1:
            b1=st.button("Explain methods")
            if st.button(b1):
                extend="in the mentioned code explain all the methods and parameters in detail"
                response = retrieval_qa.run(extend)
                st.write(f"**Response:**\n{response}")
        

        with col2:
            b2=st.button("Explain in Brief")
            if st.button(b2):
                extend="in the mentioned code explain in brief in 50 words"
                response = retrieval_qa.run(extend)
                st.write(f"**Response:**\n{response}")
            

        with col3:
            b3=st.button("Explain in detail")
            if st.button(b3):
                extend="in the mentioned code explain everything in detail"
                response = retrieval_qa.run(extend)
                st.write(f"**Response:**\n{response}")
             
            
        text=st.text_area("### Ask anything with respect to cell no:") 

        if st.button("Send"):
                response = retrieval_qa.run(text)
                st.write(f"**Response:**\n{response}")
           




if __name__ == "__main__":
    main()
