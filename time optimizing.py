import os
import time
import pickle
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain_community.llms.ollama import Ollama  
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnableSequence

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
MODEL = "llama2"

# Model Initialization
if MODEL.startswith("gpt"):
    print(OPENAI_API_KEY)  # This will print the API key; remove in production
    chat = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # Pass the API key here
else:
    chat = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings()

parser = StrOutputParser()

# Load the PDF
start_time = time.time()  # Start timing
loader = PyPDFLoader("mulu_training.pdf")
pages = loader.load_and_split()
print(f"PDF loading time: {time.time() - start_time:.2f} seconds")

# Check if embeddings already exist
try:
    with open("embeddings.pkl", "rb") as f:
        vectorstore = pickle.load(f)
except (FileNotFoundError, EOFError):  # Handle both cases
    # Create vector store and save embeddings
    vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

# Define the prompt template
template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know."
  
context: {context}
  
Question: {Question}
"""
prompt = PromptTemplate.from_template(template)

# Create retriever from vector store
retriever = vectorstore.as_retriever()

# Example retrieval query
retriever_response = retriever.invoke("main concept of Mulu", max_results=2)

# Ensure retriever_response is not empty and extract the first result
if retriever_response:  # Check if retriever_response is not empty
    first_document = retriever_response[0]  # Get the first Document
    context = first_document.page_content  # Access the page content directly
else:
    context = "No context found."  # Default context if no results found

# Construct the chain
chain = RunnableSequence(
    {
        "context": lambda x: context,  # Use extracted context
        "Question": lambda x: question  # Placeholder, updated later in the loop
    } | prompt | chat | parser  # Pass to prompt, then chat, and then parse the response
)

# Prompt the user for one or more questions
user_input = input("Enter your questions separated by a comma: ")
questions = [question.strip() for question in user_input.split(",")]

# Process each question
for question in questions:
    print(f"Question: {question} ")
    start_time = time.time()  # Start timing for each question
    answer = chain.invoke({"context": context, "Question": question})
    print(f"Answer: {answer}")
    print(f"Question processing time: {time.time() - start_time:.2f} seconds")
