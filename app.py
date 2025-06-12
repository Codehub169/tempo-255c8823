import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv
import time

# --- Helper Function to Load CSS --- 
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Default Streamlit styling will be used.")

# --- Core Logic Functions --- 
@st.cache_data(show_spinner=False) # Cache extracted text for a given PDF file content
def extract_text_from_pdf(pdf_file_bytes):
    """Extracts text from uploaded PDF file bytes."""
    try:
        import io
        pdf_file_like_object = io.BytesIO(pdf_file_bytes)
        pdf_reader = PdfReader(pdf_file_like_object)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}. The file might be corrupted or password-protected.")
        return ""

@st.cache_data(show_spinner=False)
def get_text_chunks(text):
    """Splits text into manageable chunks for embedding."""
    if not text or not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200, 
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource(show_spinner="Generating document embeddings...")
def get_vector_store(_text_chunks, embedding_model_instance):
    """Creates embeddings for text chunks and stores them in a FAISS vector store."""
    if not _text_chunks:
        st.warning("No text chunks to process for vector store.")
        return None
    try:
        embeddings = embedding_model_instance.encode(_text_chunks, show_progress_bar=False) # Progress bar in spinner
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  
        index.add(np.array(embeddings).astype('float32')) 
        return index
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_openai_response(prompt, api_key, model_name="gpt-3.5-turbo"):
    """Gets a response from OpenAI's ChatCompletion API."""
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        return "Error: OpenAI API Key not configured. Please set it in the sidebar."
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Your answers should be based SOLELY on the provided document context. If the information is not in the context, clearly state that. Do not make up information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500 
        )
        return response.choices[0].message.content.strip()
    except openai.error.RateLimitError:
        return "OpenAI API: Rate limit exceeded. Please try again later or check your OpenAI plan."
    except openai.error.AuthenticationError:
        return "OpenAI API: Authentication failed. Please check your API key."
    except openai.error.APIError as e:
        return f"OpenAI API: An error occurred: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred while communicating with OpenAI: {str(e)}"

# --- Main Application --- 
def main():
    load_dotenv() 
    st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
    load_css("static/style.css")

    # Initialize embedding model once and cache
    @st.cache_resource(show_spinner="Loading embedding model...")
    def load_embedding_model():
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}. Chat functionality will be impaired.")
            return None
    embedding_model = load_embedding_model()

    # Session State Initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store_data" not in st.session_state: # Store tuple (vector_store, text_chunks)
        st.session_state.vector_store_data = None
    if "pdf_processed_name" not in st.session_state:
        st.session_state.pdf_processed_name = None
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = str(int(time.time()))

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='font-family: Poppins, sans-serif; color: #1A73E8;'>Settings</h2>", unsafe_allow_html=True)
        st.markdown("--- ")
        
        current_api_key = st.session_state.openai_api_key
        new_api_key = st.text_input("OpenAI API Key", value=current_api_key if current_api_key and current_api_key != "YOUR_API_KEY_HERE" else "", type="password", help="Get yours from platform.openai.com")
        if new_api_key != current_api_key:
            st.session_state.openai_api_key = new_api_key
            if new_api_key and new_api_key != "YOUR_API_KEY_HERE":
                st.success("API Key updated!")
            elif not new_api_key:
                 st.info("API Key cleared.")
            else:
                st.warning("Please enter a valid API Key.")

        st.markdown("--- ")
        st.markdown("<h3 style='font-family: Poppins, sans-serif;'>Document Control</h3>", unsafe_allow_html=True)
        if st.button("Clear & Upload New PDF", key="clear_session_button", type="primary"):
            st.session_state.chat_history = []
            st.session_state.vector_store_data = None
            st.session_state.pdf_processed_name = None
            st.session_state.uploader_key = str(int(time.time()) + 1) # Force re-render of uploader
            st.success("Session cleared. Upload a new PDF.")
            st.experimental_rerun()

        if st.session_state.pdf_processed_name:
            st.info(f"Active PDF: **{st.session_state.pdf_processed_name}**")
        st.markdown("--- ")
        st.caption("App by Gemini Code Assist")

    # Main Page Layout
    st.markdown("<h1 style='text-align: center; color: #1A73E8; font-family: Poppins, sans-serif;'>Chat With Your PDF 📄</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload a PDF, let it process, and ask questions to get insights instantly!</p>", unsafe_allow_html=True)
    
    # File Uploader and Processing Logic
    if not st.session_state.pdf_processed_name:
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf", 
            key=st.session_state.uploader_key, 
            help="Max file size 200MB. Text-based PDFs work best."
        )
        if uploaded_file is not None:
            if embedding_model is None:
                st.error("Embedding model failed to load. Cannot process PDF.")
            else:
                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    try:
                        pdf_bytes = uploaded_file.getvalue()
                        raw_text = extract_text_from_pdf(pdf_bytes)
                        if not raw_text.strip():
                            st.error("No text extracted. PDF might be image-based or empty.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            if not text_chunks:
                                st.error("Text extracted but could not be chunked.")
                            else:
                                vector_store = get_vector_store(text_chunks, embedding_model)
                                if vector_store:
                                    st.session_state.vector_store_data = (vector_store, text_chunks)
                                    st.session_state.pdf_processed_name = uploaded_file.name
                                    st.session_state.chat_history = [] # Clear history for new PDF
                                    st.success(f"'{uploaded_file.name}' processed! Ready to chat.")
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to create vector store for the PDF.")
                    except Exception as e:
                        st.error(f"Critical error during PDF processing: {e}")
                        st.session_state.pdf_processed_name = None # Reset on failure
    
    # Chat Interface
    if st.session_state.pdf_processed_name and st.session_state.vector_store_data:
        st.markdown(f"<h3 style='font-family: Poppins, sans-serif;'>Chat about: {st.session_state.pdf_processed_name}</h3>", unsafe_allow_html=True)
        vector_store, text_chunks_global = st.session_state.vector_store_data

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input(f"Ask a question about '{st.session_state.pdf_processed_name}'...", key="chat_input_main")
        if user_question:
            if not st.session_state.openai_api_key or st.session_state.openai_api_key == "YOUR_API_KEY_HERE":
                st.warning("Please enter your OpenAI API Key in the sidebar to ask questions.", icon="⚠️")
            elif embedding_model is None:
                 st.error("Embedding model is not available. Cannot process question.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                with st.spinner("🤖 Thinking..."):
                    try:
                        query_embedding = embedding_model.encode([user_question])[0]
                        distances, indices = vector_store.search(np.array([query_embedding]).astype('float32'), k=min(3, len(text_chunks_global)))
                        retrieved_chunks = [text_chunks_global[i] for i in indices[0]]
                        context_for_llm = "\n\n---\n\n".join(retrieved_chunks)
                        
                        prompt = f"""Context from the PDF document:
                        ---
                        {context_for_llm}
                        ---
                        User's Question: {user_question}
                        
                        Based ONLY on the context above, answer the user's question. If the answer is not found in the context, state 'The information is not found in the provided document context.'"""
                        
                        ai_response = get_openai_response(prompt, st.session_state.openai_api_key)
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                        with st.chat_message("assistant"):
                            st.markdown(ai_response)
                            
                    except Exception as e:
                        error_msg = f"Error generating response: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {error_msg}"})
    elif not st.session_state.pdf_processed_name:
        st.info("☝️ Upload a PDF document using the uploader above to begin.")

if __name__ == '__main__':
    main()
