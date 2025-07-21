import os
from dotenv import load_dotenv

# Import necessary LangChain modules
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# --- 1. SETUP: Load API Key ---
# Load environment variables from the .env file
load_dotenv()
# Make sure your GOOGLE_API_KEY is set in the .env file
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
    exit()

print("✅ API Key Loaded")

# --- 2. DATA LOADING & PROCESSING ---
# Load documents from the 'docs' directory
# You can change the glob to match your file types, e.g., "**/*.md" for Markdown
loader = DirectoryLoader('./docs/', glob="**/*.txt", show_progress=True)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents.")

# Split documents into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents)
print(f"✅ Split documents into {len(split_documents)} chunks.")

# --- 3. INDEXING (Create Vector Store) ---
# Create embeddings using Google's model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a local vector store using FAISS and save it
# This will create a 'faiss_index' folder to store the index
# This process can take a moment the first time you run it.
vector_store = FAISS.from_documents(split_documents, embeddings)
vector_store.save_local("faiss_index")
print("✅ Vector store created and saved locally.")

# --- 4. BUILD THE Q&A CHAIN ---
# Define the LLM you want to use for answering questions
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

# Create a prompt template that instructs the LLM
# It tells the model to answer based ONLY on the provided context (your docs)
prompt = ChatPromptTemplate.from_template("""
Answer the user's question based only on the following context:
<context>
{context}
</context>

Question: {input}
""")

# Create a chain that combines the prompt and the LLM
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the main retrieval chain
# This chain takes a question, retrieves relevant documents, and passes them to the document_chain
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("✅ Q&A chain is ready.")

# --- 5. ASK QUESTIONS! ---
def ask_question(question):
    """Function to ask a question to the retrieval chain."""
    print(f"\n🤔 Question: {question}")
    response = retrieval_chain.invoke({"input": question})
    print("\n💬 Answer:")
    print(response["answer"])

# --- Example Usage ---
if __name__ == "__main__":
    # Example questions you can try
    ask_question("What is our company's refund policy?")
    ask_question("How do I request design assets from the creative team?")
    ask_question(input("Please enter your question: "))