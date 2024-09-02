import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Set environment variables and initialize embeddings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_files):
    """
    Reads the content of PDF files and extracts text.

    Args:
        pdf_files (list): A list of PDF files uploaded by the user.

    Returns:
        str: A concatenated string containing all extracted text from the PDFs.
    """
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page
    return text

def get_chunks(text):
    """
    Splits the extracted text into smaller chunks for better processing.

    Args:
        text (str): The extracted text from the PDF files.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)  # Split text into manageable chunks
    return chunks

def vector_store(text_chunks):
    """
    Creates and saves a vector store (FAISS) from the text chunks.

    Args:
        text_chunks (list): A list of text chunks.

    Returns:
        None
    """
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")  # Save the vector store locally

def get_conversational_chain(tools, question):
    """
    Creates a conversational agent to answer user queries based on the provided context.

    Args:
        tools (object): A tool that retrieves answers from the vector store.
        question (str): The user's question.

    Returns:
        None
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
                provided context just say, "answer is not available in the context", don't provide the wrong answer""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": question})
    print(response)  # Print response in console for debugging
    st.write("Reply: ", response['output'])  # Display response in Streamlit

def user_input(user_question):
    """
    Handles user input by loading the vector store, creating a retriever, and generating an answer.

    Args:
        user_question (str): The user's question.

    Returns:
        None
    """
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answers to queries from the PDF")
    get_conversational_chain(retrieval_chain, user_question)

def main():
    """
    Main function to set up the Streamlit app, handle user interactions, and display results.

    Returns:
        None
    """
    # Set up the Streamlit app configuration
    st.set_page_config("Chat PDF")
    st.header("RAG-based Chat with PDF")

    # Input field for user's question
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    # Sidebar for PDF file upload
    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_files)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Processing Completed Successfully")

if __name__ == "__main__":
    main()

