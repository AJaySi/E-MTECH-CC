import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


def get_pdf_text(pdf_docs):
    """
    Extract text from the uploaded PDF documents.
    
    Args:
        pdf_docs (list): List of uploaded PDF file objects.
    
    Returns:
        str: Concatenated text extracted from all provided PDFs.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text


def get_text_chunks(text):
    """
    Split the text into chunks for processing.
    
    Args:
        text (str): The text to be split into chunks.
    
    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(chunks):
    """
    Create and save a vector store from the text chunks.
    
    Args:
        chunks (list): List of text chunks to be converted into vectors.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating a knowledge base from the documents: {e}")


def get_conversational_chain(learning_style, explanation_depth, tone, language, programming_level):
    """
    Create a conversational chain for interacting with AI based on the provided parameters.
    
    Args:
    - learning_style (str): The student's preferred learning style.
    - explanation_depth (str): The level of detail required in explanations.
    - tone (str): The tone in which the AI should respond.
    - language (str): The preferred language for AI responses.
    - programming_level (str): The student's programming expertise level.
    
    Returns:
    - object: A LangChain conversational chain object.
    """
    prompt_template = f"""
    You are an AI tutor for MTech computer science students who are learning programming. 
    The course focuses on Python for Cloud Computing, but students might also ask about C or R programming.
    Use the following guidelines to answer questions:
    - If the answer is found in the provided context from the PDFs, use that context.
    - If the answer is not in the context, but the question is about Python, C, or R, create step-by-step explanations in {language}.
    - Adjust your explanations to the student's programming knowledge level ({programming_level}):
      - **Beginner**: Provide very basic, easy-to-follow explanations with detailed comments and breakdowns of each step.
      - **Medium**: Offer moderately detailed explanations with a focus on common practices and key concepts.
      - **Advanced**: Provide concise explanations focusing on optimization, advanced functions, and best practices.
      - **Expert**: Deliver brief, high-level explanations with a focus on complex concepts, edge cases, and performance optimization.
    - For coding examples, include comments and explanations suitable for the specified programming knowledge level.
    - Ensure your tone matches the student's preferences ({tone}) and explain concepts in the chosen language ({language}).

    If the answer is not found in the documents and is not related to Python, C, or R programming, say: 
    'The answer is not available in the provided material or is unrelated to this course.'

    Context:
    {{context}}
    
    Question:
    {{question}}
    
    Answer:
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Updated to use ConversationalRetrievalChain
    chain = ConversationalRetrievalChain(
        retriever=FAISS.from_texts([prompt_template], embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")), 
        llm=model
    )
    return chain


def embed_given_context(pdf_docs):
    """Common function to embed given doc """
    raw_text = get_pdf_text(pdf_docs)
    if raw_text:
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)


def clear_chat_history():
    """
    Clear the chat history in the session state.
    """
    st.session_state.messages = [{"role": "assistant", "content": "Please upload your course PDFs and ask me any question related to them!"}]


def start_pdf_chat(user_question, learning_style, explanation_depth, tone, language, programming_level):
    """
    Process the user's question and provide an answer based on the uploaded PDFs and user preferences.
    
    Args:
        user_question (str): The question asked by the user.
        learning_style (str): The preferred learning style of the student.
        explanation_depth (str): The level of detail required in explanations.
        tone (str): The preferred tone for responses.
        language (str): The preferred language for responses.
    
    Returns:
        dict: The response from the AI, including the answer text.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(learning_style, explanation_depth, tone, language, programming_level)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response
    except Exception as e:
        st.error(f"Error processing your question: {e}")
        return None
