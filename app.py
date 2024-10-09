import json
import sys
import os
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Define Bedrock embeddings and models
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1", client=bedrock)

def data_ingestion():
    loader = PyPDFDirectoryLoader("Data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vec_store(documents):
    vectorstore_faiss = FAISS.from_documents(
        documents, bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")  # Save the FAISS index locally
    return vectorstore_faiss

def load_vec_store():
    # Load the vector store only if the FAISS index exists
    if os.path.exists("faiss_index"):
        vectorstore_faiss = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        return vectorstore_faiss
    else:
        return None

def get_claud_model():
    llm = Bedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock, model_kwargs={'max_tokens': 1000})
    return llm

def get_llama3_model():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Define the prompt template
prompt_template = """
You are a VALORANT esports assistant. Use the context provided to answer questions about player performance and team composition.

Context: {context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Manual retrieval and LLM question answering
def retrieve_documents(query, vectorstore_faiss, k=3):
    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    docs = retriever.get_relevant_documents(query)
    print("Retrieved Docs",docs)
    return docs

def answer_question_with_docs(llm, query, retrieved_docs):
    # Combine the content of the retrieved documents
    context = " ".join([doc.page_content for doc in retrieved_docs])
    
    print("Context for LLM:", context)  # Debugging statement to check context

    # Create a new prompt with context and question
    full_prompt = PROMPT.format(context=context, question=query)
    
    # Step 3: Use the LLM to generate an answer
    qa_chain = LLMChain(llm=llm, prompt=PROMPT)
    answer = qa_chain.run({"context": context, "question": query})
    
    return answer, retrieved_docs

# Example usage:
def get_response_llm(llm, vectorstore_faiss, query):
    retrieved_docs = retrieve_documents(query, vectorstore_faiss)
    answer, docs = answer_question_with_docs(llm, query, retrieved_docs)
    return answer, docs

def main():
    # Streamlit UI
    st.title("Valorant Q/A Chatbot")
    
    # Sidebar for creating or updating the vector store
    st.sidebar.title("Update or Create Vector Store")
    
    # Text input for user query
    user_question = st.text_input("Ask a question:")
    
    # Button to trigger the LLM response
    if st.button("Submit"):
        if user_question:
            # Load the vector store
            vectorstore_faiss = load_vec_store()
            if vectorstore_faiss:
                # Choose the LLM model (Claude or Llama3)
                llm = get_llama3_model()  # You can swap to get_llama3_model if needed
                
                # Get the response from the LLM
                answer, docs = get_response_llm(llm, vectorstore_faiss, user_question)
                
                # Display the answer
                st.write("Answer:", answer)
            else:
                st.warning("Please create the vector store first.")
        else:
            st.warning("Please enter a question.")

    # Button to update or create vector store
    if st.sidebar.button("Create/Update Vector Store"):
        # Ingest data and create vector store
        documents = data_ingestion()
        vectorstore_faiss = get_vec_store(documents)
        st.sidebar.success("Vector store created/updated successfully!")

if __name__ == "__main__":
    main()
