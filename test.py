import os
import torch
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO 

# LangChain Core
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage # For clean chat history
from langchain_core.documents import Document # For creating documents with metadata

# LangChain HuggingFace
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint

# HTML templates
from htmlTemplates import css, bot_template, user_template

def get_pdf_pages_as_documents(pdf_docs):
    """
    Extracts text from PDFs, treating each page as a Document
    with associated metadata (source and page number).
    Fixes multi-page text merging by adding a robust separator.
    """
    documents = []
    # Store filenames (Feature 4)
    st.session_state.pdf_filenames = [pdf.name for pdf in pdf_docs]

    progress_bar = st.progress(0, text="Extracting text from uploaded PDFs...")

    total_pages = 0
    pdf_streams = {} 
    
    # First Pass: Prepare streams and count pages
    for pdf in pdf_docs:
        try:
            # Read all content into a BytesIO stream
            stream = BytesIO(pdf.read())
            pdf_streams[pdf.name] = stream
            # Count pages
            total_pages += len(PdfReader(stream).pages)
            stream.seek(0) # Reset stream pointer after counting
        except Exception:
             st.warning(f"Could not read {pdf.name}. It may be corrupted or encrypted.")
             del pdf_streams[pdf.name]


    processed_pages = 0
    # Second Pass: Extract text page by page
    for file_name, stream in pdf_streams.items():
        try:
            pdf_reader = PdfReader(stream)
        except Exception as e:
            st.error(f"Failed to read PDF stream for {file_name}: {e}")
            continue

        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Add a robust separator after the page text to prevent word merging
                text = page_text + "\n\n-- PAGE_BREAK --\n\n"
                
                # Create a LangChain Document for the page, storing metadata
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_name, "page": i + 1} # Page numbers start at 1
                ))
            
            processed_pages += 1
            progress = processed_pages / total_pages if total_pages > 0 else 0
            progress_bar.progress(progress, text=f"Processing {file_name} (Page {i+1})...({int(progress*100)}%)")

    progress_bar.empty()
    st.success("PDF text extraction complete!")
    return documents


def get_text_chunks(documents, chunk_size, chunk_overlap):
    """Splits the list of page documents into manageable, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    # The splitter retains the metadata ("source", "page")
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks using HuggingFace embeddings."""
    device = "cpu"
    st.info(f"Using device: **{device}** for embeddings")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-xl",
        model_kwargs={"device": device}
    )
    
    # Use from_documents to retain the metadata in the vector store
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore


def get_llm_endpoint(token):
    """Helper function to initialize the HuggingFace LLM endpoint."""
    return HuggingFaceEndpoint(
        model="meta-llama/Llama-3.1-8B",
        temperature=0.1,
        # temperature=temp,
        max_new_tokens=100,
        huggingfacehub_api_token=token,
        provider="auto"
    )

def get_chain(vectorstore, token, with_memory=True):
    """Generic function to create either a conversational or standalone chain."""
    if not token:
        st.error("HUGGINGFACEHUB_API_TOKEN is not set.")
        return None

    llm = get_llm_endpoint(token)
    memory = None
    if with_memory:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # Crucial setting for page number tracking
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True, # MUST be True for page numbers
        verbose=False # Set to False to hide internal LLM thoughts
    )
    return chain

## 🗣️ Chat Handlers

def process_questions(questions_input):
    """
    Processes multiple user questions using the stateless chain for batch processing.
    """
    if not questions_input:
        return

    questions = [q.strip() for q in questions_input.split('\n') if q.strip()]
    if not questions or not st.session_state.conversation:
        st.warning("Please enter questions and process PDFs first.")
        return

    # 1. Initialize the stateless chain for batch queries
    try:
        vectorstore = st.session_state.conversation.retriever.vectorstore
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        temp_standalone_chain = get_chain(vectorstore, token, with_memory=False)
    except Exception as e:
        st.error(f"Could not initialize standalone chain for batch processing: {e}")
        return

    
    for question in questions:
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        
        with st.spinner(f"Answering: **{question[:50]}...**"):
            try:
                # Use the TEMPORARY, STANDALONE CHAIN with empty history
                response = temp_standalone_chain({'question': question, 'chat_history': []}) 
                
                # Extract Source Information
                source_info = extract_source_info(response.get('source_documents', []))
                
                # Format Bot Response with Sources
                bot_message_html = f"{response['answer']}<div class='source-info'>{source_info}</div>"

                # Append to main chat history 
                st.session_state.chat_history.append(HumanMessage(content=question))
                st.session_state.chat_history.append(AIMessage(content=bot_message_html))

            except Exception as e:
                st.error(f"An error occurred while answering '{question[:50]}...': {e}")
                st.session_state.chat_history.append(AIMessage(content=f"Sorry, I encountered an error: {e}"))
                break

    display_chat_history()


def extract_source_info(source_documents):
    """
    Processes the list of retrieved documents and formats the file names and page numbers.
    """
    if not source_documents:
        return "Source: No document chunks were used for this answer."
    
    sources = {}
    for doc in source_documents:
        file_name = doc.metadata.get("source", "Unknown File")
        page_number = doc.metadata.get("page", "Unknown Page")
        
        if file_name not in sources:
            sources[file_name] = set()
        sources[file_name].add(str(page_number))

    source_text = "Source Pages Used:<br>"
    for file_name, pages in sources.items():
        # Clean up file name for display (remove .pdf extension if present)
        display_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
        source_text += f" • **{display_name}**: Pages {', '.join(sorted(list(pages)))};<br>"
        
    return source_text.rstrip(';<br>')


def display_chat_history():
    """Displays the current chat history in the main UI area."""
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                # Bot message contains HTML with source info
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)


def clear_chat():
    """Clears the chat history and conversation chain memory. (Feature 2)"""
    st.session_state.chat_history = []
    if st.session_state.conversation and st.session_state.conversation.memory:
        st.session_state.conversation.memory.clear()
    st.success("Chat history cleared!")


def get_chat_download_content():
    """Generates the content of the chat log as a string. (Feature 5)"""
    content = "--- PDF Chat Log ---\n\n"
    if st.session_state.get('pdf_filenames'):
        content += "Documents Used: " + ", ".join(st.session_state.pdf_filenames) + "\n\n"
        
    if st.session_state.get('chat_history'):
        for i, message in enumerate(st.session_state.chat_history):
            speaker = "USER" if i % 2 == 0 else "BOT"
            # Strip HTML tags/source info for clean download log
            message_content = message.content
            if "<div class='source-info'>" in message_content:
                 message_content = message_content.split("<div class='source-info'>")[0]

            content += f"[{speaker}]: {message_content}\n"
    else:
        content += "No chat history to download.\n"
    
    return content

## 🏠 Main Application

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize Session States
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] 
    if "pdf_filenames" not in st.session_state:
        st.session_state.pdf_filenames = []

    st.header("Chat with multiple PDFs :books:")

    # Display processed filenames (Feature 4)
    if st.session_state.pdf_filenames:
        st.info(f"📚 Documents Processed: **{', '.join(st.session_state.pdf_filenames)}**")

    st.button("🧹 Clear Chat", on_click=clear_chat)
    st.markdown("---")
    
    # Multi-Question Input (Feature 3)
    questions_input = st.text_area(
        "Ask one or more questions (separate by newlines):",
        placeholder="e.g.\n1. What is the main finding of the report?\n2. Who are the key authors?"
    )

    if st.button("Submit Questions"):
        process_questions(questions_input)
    
    st.markdown("---")

    # Download Chat Button (Feature 5)
    st.download_button(
        label="⬇️ Download Chat Log",
        data=get_chat_download_content(),
        file_name="pdf_chat_log.txt",
        mime="text/plain"
    )
    
    display_chat_history()
    
    # Sidebar: PDF upload and Processing Settings
    with st.sidebar:
        st.subheader("Document Processing Settings")

        # Adjustable Chunk Size and Overlap (Feature 1)
        # temp = st.slider(
        #     "Temperature (Creativity/Randomness)",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=0.1, # Keep the original low value as default for RAG
        #     step=0.05,
        #     help="Higher temperature (closer to 1.0) makes the LLM more creative and less deterministic; lower (closer to 0.0) makes it more factual and safer for document Q&A."
        # )
        chunk_size = st.slider(
            "Chunk Size (Characters)",
            min_value=500,
            max_value=3000,
            value=1500,
            step=100
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (Characters)",
            min_value=0,
            max_value=500,
            value=250,
            step=50
        )

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click 'Process'", accept_multiple_files=True
        )
        if st.button("Process") and pdf_docs:
            token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not token:
                 st.error("HUGGINGFACEHUB_API_TOKEN is missing. Please set it in your environment or `.env` file.")
                 return

            with st.spinner("Processing PDFs..."):
                clear_chat() 
                
                # STEP 1: Get documents (pages) with metadata
                documents = get_pdf_pages_as_documents(pdf_docs)
                
                if not documents:
                    st.error("No extractable text found in uploaded files.")
                    return

                # STEP 2: Get text chunks (metadata retained)
                text_chunks = get_text_chunks(documents, chunk_size, chunk_overlap)
                st.success(f"Text split into **{len(text_chunks)}** chunks.")

                # STEP 3: Create Vector Store
                vectorstore = get_vectorstore(text_chunks)
                
                # STEP 4: Create Conversation Chain (with memory)
                st.session_state.conversation = get_chain(vectorstore, token, with_memory=True)
                st.success("Processing complete! You can now ask questions.")


if __name__ == "__main__":
    main()