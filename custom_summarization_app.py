import openai
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

openai.api.key = os.environ["OPENAI_API_KEY"]

def custom_summary(docs, llm, custom_prompt, chain_type, num_summaries):
    custom_prompt = custom_prompt + """:\n {text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt,input_variables = ["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n{text}", input_variables=["text"])
    if chain_type == "map_reduce":
        chain = load_summarize_chain(llm,chain_type=chain_type,map_prompt=MAP_PROMPT,combine_prompt=COMBINE_PROMPT)
    
    else:
        chain = load_summarize_chain(llm,chain_type=chain_type)

    summaries = []
    for i in range(num_summaries):
        summary_output= chain({"input_documents":docs}, return_only_outputs=True)
        summaries.append(summary_output)
    
    return summaries

@st.cache_data
def setup_document(pdf_file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs
    

def color_chunks():
    pass

def main():
    st.set_page_config(layout="wide")
    st.title("Custom Summarization App")

if __name__ == "__main__":
    main()

