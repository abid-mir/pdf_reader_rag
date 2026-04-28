import os
import torch
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain Core
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
# from langchain_classic.chains.question_answering import load_qa_chain
# from langchain_classic.prompts import PromptTemplate

# LangChain HuggingFace
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint

# HTML templates
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    progress_text = "Extracting text from uploaded PDFs..."
    progress_bar = st.progress(0, text=progress_text)

    total_pages = sum(len(PdfReader(pdf).pages) for pdf in pdf_docs)
    processed_pages = 0

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            processed_pages += 1
            progress = processed_pages / total_pages
            progress_bar.progress(progress, text=f"Processing pdf...({int(progress*100)}%)")


    progress_bar.empty()
    st.success("PDF text extraction complete!")
    return text


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    st.info(f"Using device: {device} for embeddings")
    
    embeddings = HuggingFaceEmbeddings(
        # uncomment the following line in a new device where embedding model has not yet been downloaded/cached
        model_name="hkunlp/instructor-xl",
        # model_name=r"C:\Users\adi77\.cache\huggingface\hub\models--hkunlp--instructor-xl\snapshots\ce48b213095e647a6c3536364b9fa00daf57f436",
        model_kwargs={"device": device}
    )
    # To check whether local embeddingm model is actually being used
    # print("Embedding type:", type(embeddings._client))

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # LLM
    llm = HuggingFaceEndpoint(
        model="meta-llama/Llama-3.1-8B",
        temperature=0.1,
        max_new_tokens=100,
        huggingfacehub_api_token=token,
        provider="auto"
    )

    # combine_docs_chain = load_qa_chain(llm=llm, chain_type="stuff")

    # prompt = PromptTemplate(template="Given the following conversation and a follow-up question, rephrase the follow-up question to be standalone.\n\nConversation: {chat_history}\nFollow-up question: {question}\nStandalone question:", input_variables=["chat_history", "question"])
    # question_generator = LLMChain(llm=llm, prompt=prompt)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        # combine_docs_chain=combine_docs_chain,
        # question_generator=question_generator,
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar: PDF upload
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click 'Process'", accept_multiple_files=True
        )
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete! You can now ask questions.")


if __name__ == "__main__":
    main()