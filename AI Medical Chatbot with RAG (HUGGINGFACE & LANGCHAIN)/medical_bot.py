import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Initialize session state for chat history and settings
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "max_length" not in st.session_state:
    st.session_state.max_length = 512
if "model" not in st.session_state:
    st.session_state.model = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    try:
        # Get the token from environment variables
        huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not huggingface_api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
        
        # Initialize the model with correct parameters
        llm = HuggingFaceHub(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=huggingface_api_token,
            task="text-generation",
            model_kwargs={
                "max_length": st.session_state.max_length,
                "temperature": st.session_state.temperature
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        st.error("Please check your HuggingFace API token and model configuration")
        return None

def initialize_qa_chain():
    HUGGINGFACE_REPO_ID = st.session_state.model
    
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    
    try:
        # Initialize embedding model with specific model kwargs
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load the FAISS database
        DB_FAISS_PATH = "vectorstore/db_faiss"
        if not os.path.exists(DB_FAISS_PATH):
            st.error(f"Vector store not found at {DB_FAISS_PATH}")
            return None
            
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        # Initialize LLM
        llm = load_llm(HUGGINGFACE_REPO_ID)
        if llm is None:
            return None
            
        # Create QA chain with specific parameters
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        return qa_chain
        
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        st.error("Check if your vector store exists and embedding model is accessible")
        return None

def create_sidebar():
    with st.sidebar:
        st.title("⚙️ Chat Settings")
        
        # Model selection
        st.subheader("Model Settings")
        model_options = {
            "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
            "LLAMA-2": "meta-llama/Llama-2-7b-chat-hf",
            "Falcon": "tiiuae/falcon-7b-instruct"
        }
        selected_model = st.selectbox(
            "Choose a model",
            options=list(model_options.keys()),
            index=0
        )
        st.session_state.model = model_options[selected_model]
        
        # Temperature slider
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher values make the output more random, lower values make it more focused and deterministic."
        )
        
        # Max length slider
        st.session_state.max_length = st.slider(
            "Maximum Length",
            min_value=64,
            max_value=1024,
            value=512,
            step=64,
            help="The maximum number of tokens to generate in the response."
        )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # About section
        st.subheader("About")
        st.markdown("""
        This Medical AI Assistant uses advanced language models to provide 
        medical information based on your questions. It references a 
        knowledge base of medical documents for accurate responses.
        
        **Note:** This is an AI assistant and should not replace 
        professional medical advice.
        """)
        
        # Display current settings
        st.subheader("Current Settings")
        st.write(f"Model: {selected_model}")
        st.write(f"Temperature: {st.session_state.temperature}")
        st.write(f"Max Length: {st.session_state.max_length}")

def main():
    st.set_page_config(
        page_title="Medical AI Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Create sidebar
    create_sidebar()
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🏥 Medical AI Assistant")
        st.markdown("""
        Welcome to the Medical AI Assistant! Ask any medical-related questions, and I'll provide answers based on my medical knowledge base.
        """)
        
        # Initialize QA chain
        qa_chain = initialize_qa_chain()
        
        if qa_chain is None:
            st.error("Failed to initialize the AI assistant. Please check your configuration.")
            return
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        st.markdown(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask your medical question here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = qa_chain.invoke({'query': prompt})
                        answer = response["result"]
                        
                        # Format source documents
                        sources = "\n\n".join([
                            f"**Source {i+1}:**\n{doc.page_content[:200]}..."
                            for i, doc in enumerate(response["source_documents"])
                        ])
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Display sources in expander
                        with st.expander("View Sources"):
                            st.markdown(sources)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    with col2:
        st.subheader("📚 Recent Sources")
        if st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if "sources" in last_message:
                st.markdown(last_message["sources"])
            else:
                st.info("No sources available for the last message.")
        else:
            st.info("Start a conversation to see sources here.")

if __name__ == "__main__":
    main()
