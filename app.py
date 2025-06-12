import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import google.api_core.exceptions
import os
from dotenv import load_dotenv
import time
import io

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
        pdf_file_like_object = io.BytesIO(pdf_file_bytes)
        pdf_reader = PdfReader(pdf_file_like_object)
        text = ""
        if not pdf_reader.pages:
            st.warning("PDF has no pages or could not be read.")
            return ""
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
def get_vector_store(_text_chunks, _embedding_model_instance):
    """Creates embeddings for text chunks and stores them in a FAISS vector store."""
    if not _text_chunks:
        st.warning("No text chunks to process for vector store.")
        return None
    try:
        embeddings = _embedding_model_instance.encode(_text_chunks, show_progress_bar=False)
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            st.error(f"Embeddings are not in the expected format. Got shape: {embeddings.shape if isinstance(embeddings, np.ndarray) else type(embeddings)}")
            return None
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  
        index.add(np.array(embeddings).astype("float32")) 
        return index
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_gemini_response(prompt_text, api_key, model_name="gemini-1.5-flash-latest"):
    """Gets a response from Google Gemini API."""
    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        return "Error: Google API Key not configured. Please set it in the sidebar."
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"Error configuring Google API: {str(e)}"
    
    generation_config = genai.types.GenerationConfig(
        temperature=0.2,
        max_output_tokens=2000 # Increased slightly for potentially longer summaries/answers
    )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt_text,
            generation_config=generation_config
        )

        if response.parts:
            return response.text 
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"Gemini API: Content generation blocked. Reason: {response.prompt_feedback.block_reason.name} - {response.prompt_feedback.block_reason_message or ''}"
        elif response.candidates and response.candidates[0].finish_reason.name not in ["STOP", "MAX_TOKENS"]:
             finish_reason_details = response.candidates[0].finish_reason.name
             return f"Gemini API: Could not generate a complete response. Finish reason: {finish_reason_details}"
        elif not response.candidates:
             return "Gemini API: No candidates returned in the response."
        else:
            return response.text # Covers empty valid responses

    except google.api_core.exceptions.PermissionDenied as e:
        return f"Google API: Authentication failed (Permission Denied). Check your API key and ensure the Gemini API is enabled. Details: {str(e)}"
    except google.api_core.exceptions.InvalidArgument as e:
        return f"Google API: Invalid argument (e.g., model name '{model_name}' or other parameters). Details: {str(e)}"
    except google.api_core.exceptions.ResourceExhausted as e:
        return f"Google API: Rate limit exceeded or quota exhausted. Check your Google Cloud/AI Studio plan. Details: {str(e)}"
    except google.api_core.exceptions.GoogleAPIError as e: # Catch generic Google API errors
        return f"Google API: An API error occurred: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred while communicating with Google Gemini: {str(e)}"

# --- Main Application --- 
def main():
    load_dotenv() 
    st.set_page_config(page_title="Chat with PDF", page_icon="
ud83d
cc4", layout="wide") # Fixed page_icon
    load_css("static/style.css")

    @st.cache_resource(show_spinner="Loading embedding model...")
    def load_embedding_model():
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}. Chat functionality will be impaired.")
            return None
    embedding_model = load_embedding_model()

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store_data" not in st.session_state:
        st.session_state.vector_store_data = None # Will store (faiss_index, text_chunks)
    if "pdf_processed_name" not in st.session_state:
        st.session_state.pdf_processed_name = None
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")
    if "uploader_key" not in st.session_state: # To reset file uploader
        st.session_state.uploader_key = str(int(time.time()))

    with st.sidebar:
        st.markdown("<h2 style='font-family: Poppins, sans-serif; color: #1A73E8;'>Settings</h2>", unsafe_allow_html=True)
        st.markdown("---")

        # API Key Input
        api_key_value = st.session_state.google_api_key if st.session_state.google_api_key and st.session_state.google_api_key != "YOUR_GOOGLE_API_KEY_HERE" else ""
        new_api_key = st.text_input(
            "Google API Key", 
            value=api_key_value, 
            type="password", 
            help="Get yours from Google AI Studio (makersuite.google.com)"
        )
        if new_api_key != st.session_state.google_api_key:
            st.session_state.google_api_key = new_api_key
            if new_api_key and new_api_key != "YOUR_GOOGLE_API_KEY_HERE":
                st.success("Google API Key updated!")
            elif not new_api_key:
                 st.info("API Key cleared.")
            else: # Catches placeholder or other invalid short keys
                st.warning("Please enter a valid Google API Key.")
            st.rerun()

        st.markdown("---")
        st.markdown("<h3 style='font-family: Poppins, sans-serif;'>Document Control</h3>", unsafe_allow_html=True)
        if st.button("Clear & Upload New PDF", key="clear_session_button", type="primary"):
            st.session_state.chat_history = []
            st.session_state.vector_store_data = None
            st.session_state.pdf_processed_name = None
            st.session_state.uploader_key = str(int(time.time()) + 1) # Force re-render of file_uploader
            st.success("Session cleared. Upload a new PDF.")
            st.rerun()

        if st.session_state.pdf_processed_name:
            st.info(f"Active PDF: **{st.session_state.pdf_processed_name}**")
        st.markdown("---")
        st.caption("App by Gemini Code Assist")

    # Main page layout
    st.markdown("<h1 style='text-align: center; color: #1A73E8; font-family: Poppins, sans-serif;'>Chat With Your PDF 
ud83d
cac</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload a PDF, let it process, and ask questions to get insights instantly!</p>", unsafe_allow_html=True)
    
    # PDF Upload and Processing Section
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

                        if not raw_text or not raw_text.strip():
                            st.error("No text extracted. PDF might be image-based, empty, or corrupted.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            if not text_chunks:
                                st.error("Text extracted but could not be chunked. Document might be too short or lack textual content.")
                            else:
                                vector_store = get_vector_store(text_chunks, embedding_model)
                                if vector_store:
                                    st.session_state.vector_store_data = (vector_store, text_chunks)
                                    st.session_state.pdf_processed_name = uploaded_file.name
                                    st.session_state.chat_history = [] # Clear history for new PDF
                                    st.success(f"'{uploaded_file.name}' processed! Ready to chat.")
                                    st.rerun()
                                else:
                                    st.error("Failed to create vector store for the PDF. Document processing aborted.")
                    except Exception as e:
                        st.error(f"Critical error during PDF processing: {e}")
                        st.session_state.pdf_processed_name = None # Reset if critical error
                        st.session_state.vector_store_data = None

    # Chat Interface Section
    if st.session_state.pdf_processed_name and st.session_state.vector_store_data:
        st.markdown(f"<h3 style='font-family: Poppins, sans-serif;'>Chat about: {st.session_state.pdf_processed_name}</h3>", unsafe_allow_html=True)
        
        vector_store, text_chunks_from_store = st.session_state.vector_store_data

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        user_question = st.chat_input(f"Ask a question about '{st.session_state.pdf_processed_name}'...", key="chat_input_main")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            if not st.session_state.google_api_key or st.session_state.google_api_key == "YOUR_GOOGLE_API_KEY_HERE":
                error_msg = "Please enter your Google API Key in the sidebar to ask questions. 
ud83d
dd11"
                st.warning(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"): # Also show this in chat
                     st.markdown(error_msg)

            elif embedding_model is None:
                error_msg = "Embedding model is not available. Cannot process question."
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                     st.markdown(error_msg)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking... 
ud83e
dd14"):
                        try:
                            # 1. Embed user question
                            question_embedding = embedding_model.encode([user_question], show_progress_bar=False)
                            if question_embedding.ndim == 1: # Ensure 2D for FAISS
                                question_embedding = np.array([question_embedding])

                            # 2. Search FAISS vector store
                            k = 5 # Number of relevant chunks to retrieve
                            distances, indices = vector_store.search(question_embedding.astype('float32'), k)
                            
                            relevant_chunks = []
                            if indices.size > 0: # Check if any indices were returned
                                relevant_chunks = [
                                    text_chunks_from_store[i] for i in indices[0] 
                                    if 0 <= i < len(text_chunks_from_store) # Safety check
                                ]

                            if not relevant_chunks:
                                bot_response_content = "I couldn't find relevant information in the document to answer your question. Try rephrasing or asking something else."
                            else:
                                # 3. Construct prompt
                                context = "\n\n".join(relevant_chunks)
                                prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context from a PDF document.
If the answer is not found in the context, clearly state "I couldn't find the answer in the provided document context."
Do not make up information or answer from your general knowledge.

Context from the document:
---
{context}
---

User Question: {user_question}

Answer:
"""
                                # 4. Get Gemini response
                                bot_response_content = get_gemini_response(prompt, st.session_state.google_api_key)
                        
                        except Exception as e:
                            st.error(f"Error processing your question: {e}")
                            bot_response_content = f"Sorry, an error occurred while processing your request: {str(e)}"

                        st.markdown(bot_response_content)
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_response_content})
                        # No st.rerun() needed here, chat_input/chat_message updates UI.


if __name__ == "__main__":
    main()
