import dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQAWithSourcesChain
import streamlit as st

dotenv.load_dotenv(dotenv_path='.env')
key= os.getenv("my_api_key")
llm= GoogleGenerativeAI(model= "gemini-2.5-flash", api_key= key)
#essay= llm.invoke("write a poem on pen?")
#print(essay)

embeddings = HuggingFaceEmbeddings()
vectordb_file_path= "C:\\Users\\Priya Adhwaria\\Desktop\\LLM Project\\QA LLM\\faiss_index.faiss"
#Creating Chunks
def create_chunks():
    loader = TextLoader(file_path="C:\\Users\\Priya Adhwaria\\Desktop\\LLM Project\\QA LLM\\train-v2.0.json")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=10, separators=["\n", "]"])
    split_data = text_splitter.split_documents(data)
    return split_data

#Creating Vector Database
def create_vectordb():
    vectordb = FAISS.from_documents(documents= create_chunks(), embedding= embeddings)
    vectordb.save_local(vectordb_file_path)

#Creating RetrievalQA Chain
def qa_chain():
    new_db = FAISS.load_local(vectordb_file_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    temp = """ For given following context and query, provide answer of query from the context of most relevant chunk obtained from retriever from the source document only. If the answer is not present in the context, show 'I don't know'. Don't hallucinate the answer.
    summaries: {summaries}
    question: {question}"""

    prompt = PromptTemplate(template=temp, input_variables=["summaries", "question"])
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever, combine_prompt=prompt)
    return chain

def get_answer():
    st.title("Question_Answer_Model")
    question= st.text_input("Enter the question: ")

    if question:
        llm_chain= qa_chain()
        response= llm_chain.invoke(question)

        st.header("Answer: ")
        st.write(response["answer"])

if __name__ == "__main__":
    get_answer()





