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
import filetype

load_dotenv()


def save_api_key(api_key):
    """
    Save the Gemini API key to a .env file.

    Args:
        api_key (str): The API key provided by the user.
    """
    with open(".env", "w") as f:
        f.write(f"GOOGLE_API_KEY={api_key}")


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
    You are an AI tutor for MTech computer science students who are learning programming. The course focuses on Python for Cloud Computing, 
    but students might also ask about C or R programming.

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
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)



def clear_chat_history():
    """
    Clear the chat history in the session state.
    """
    st.session_state.messages = [{"role": "assistant", "content": "Please upload your course PDFs and ask me any question related to them!"}]


def user_input(user_question, learning_style, explanation_depth, tone, language, programming_level):
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


def main():
    st.title("Interactive AI Tutor - EMTECH-CC")
    st.write("Welcome! I’m here to help you learn from your course materials. Just ask me any question based on the PDFs you’ve uploaded.")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        clear_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("What would you like to learn today? 💡"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Finding best answer from attached materials... 🔍"):
                response = user_input(prompt, learning_style, explanation_depth, tone, language, programming_level)
                if response:
                    full_response = response['output_text']
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.title("Setup Your AI Tutor 📂")
    
    # Show API key input only if key is missing
    if not os.getenv("GOOGLE_API_KEY"):
        api_key_input = st.text_input(
            "**Enter your Gemini API Key 🔑**",
            type="password",
            help="Get your key here: [Google AI Studio](https://aistudio.google.com/app/apikey)",
        )
        if api_key_input:
            save_api_key(api_key_input)
            st.success("Done! You can now use the app. 🔓")
            genai.configure(api_key=api_key_input)

    with st.expander("🏗️ Personalize Your Learning ⚙️", expanded=False):
        st.write("Choose your preferences for a personalized learning experience:")
        language = st.selectbox("Preferred Language 🗣️ ", ["English", "Hindi", "Tamil", "Telugu", "Marathi", "Bengali", "Gujarati", "Kannada", "Malayalam"])
        learning_style = st.selectbox("Learning Style 🎨", ["Visual (Images, diagrams)", "Auditory (Listening, discussions)", "Reading/Writing", "Kinesthetic (Hands-on practice)"])
        explanation_depth = st.selectbox("Depth of Explanation 📝", ["Brief (Concise answers)", "Detailed (In-depth explanations)"])
        tone = st.selectbox("Preferred Tone 🗣️ ", ["Formal", "Friendly", "Motivational"])
        programming_level = st.selectbox("Programming Experience", ["Beginner", "Medium", "Advanced"])

    uploaded_files = st.file_uploader("Upload your files (PDF, Video, Image, PPT). Click 'Submit & Process' to start! 📖", 
                                  accept_multiple_files=True)

    # Button to process files
    if st.button("📖 Submit & Process"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_extension = get_file_type(uploaded_file)
                
                if file_extension == 'pdf':
                    process_pdf([uploaded_file])  # PDF processing
                elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:  # Supported video formats
                    process_video([uploaded_file])  # Video processing
                elif file_extension in ['jpg', 'jpeg', 'png', 'gif']:  # Supported image formats
                    process_image([uploaded_file])  # Image processing
                elif file_extension in ['ppt', 'pptx']:  # Supported PPT formats
                    process_ppt([uploaded_file])  # PPT processing
                else:
                    st.warning(f"Unsupported file type: {file_extension}. Please upload a PDF, video, image, or PPT file.")
        else:
            st.warning("No files were uploaded. Please upload at least one file.")    

# Placeholder functions for different file types
def process_pdf(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    if raw_text:
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("PDF content processed! Ask me questions based on the content. 🎉")

def process_video(video_files):
    # Placeholder for video processing logic
    st.info("Video processing function called. 🎥")

def process_image(image_files):
    # Placeholder for image processing logic
    st.info("Image processing function called. 🖼️")

def process_ppt(ppt_files):
    # Placeholder for PPT processing logic
    st.info("PPT processing function called. 📊")

def get_file_type(file):
    kind = filetype.guess(file.read())
    file.seek(0)  # Reset file pointer after reading
    return kind.extension if kind else None


if __name__ == "__main__":
    main()
