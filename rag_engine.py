import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# ---STEP 1: Extract text from PDF ----
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# ---STEP 2: Split text into chunks ----
def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )    
    return splitter.split_text(text)

# ---STEP 3: Create vector store ----
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/embedding-001"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") # Save locally
    return vector_store

# ---STEP 4: Build QA Chain ----
def get_qa_chain():
    prompt_template = """
    Answer the question using ONLY the context provided below.
    If the answer is not in the context, say:
    "I could not find this information in the uploaded doccument."
    Do not make up any information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model = "gemini-pro",
        temperature = 0.3    # Lower = more factual, less creative
    )
    prompt = PromptTemplate(
        template = prompt_template,
        input_variables = ["context", "question"]
    )
    return load_qa_chain(model, chain_type = "stuff", prompt = prompt)

# --- STEP 5: Answer a question ----
def answer_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/embedding-001"
    )
    # Load the saved vector store
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization = True
    )
    # Find top 4 most relevant chunks
    relevant_docs = vector_store.similarity_search(
        user_question, k = 4
    )
    # Run QA chain
    chain = get_qa_chain()
    response = chain(
        {"input_documents": relevant_docs, "question": user_question},
        return_only_outputs = True
    )
    return response["output_text"], relevant_docs