import streamlit as st
import os
from dotenv import load_dotenv

# Import the same LangChain modules
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# --- CACHING FUNCTIONS WITH THE transport="rest" FIX ---
@st.cache_resource
def load_llm():
    """Loads the Language Model, cached for the session."""
    # Add transport="rest" to use HTTPS instead of gRPC
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, transport="rest") # <<< CHANGE

@st.cache_resource
def load_embeddings_model():
    """Loads the Embeddings Model, cached for the session."""
    # Add transport="rest" to use HTTPS instead of gRPC
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest") # <<< CHANGE
# --------------------------------------------------------

# --- Main App Logic ---
st.set_page_config(page_title="Internal Docs Q&A", page_icon="🤖")
st.title("Internal Docs Q&A Assistant")
st.write("Ask any question about our internal processes and get an instant answer.")

# Use a session state to store the chain and avoid reloading on every interaction
if 'retrieval_chain' not in st.session_state:
    with st.spinner("Initializing AI Assistant... Please wait."):
        
        # Use the cached functions to load the models
        embeddings = load_embeddings_model()
        llm = load_llm()
        
        # Load the pre-built FAISS index from the local path
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Define the prompt
        prompt = ChatPromptTemplate.from_template("""
        Answer the user's question based only on the following context:
        <context>{context}</context>
        Question: {input}
        """)
        
        # Create the chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- User Interaction ---
question = st.text_input("e.g., What is our refund policy?")

if question:
    with st.spinner("Searching for the answer..."):
        response = st.session_state.retrieval_chain.invoke({"input": question})
        st.write("### Answer")
        st.write(response["answer"])

        # Optional: Show the sources it used to generate the answer
        with st.expander("Show Sources"):
            for doc in response["context"]:
                st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.write(doc.page_content)
                st.write("---")