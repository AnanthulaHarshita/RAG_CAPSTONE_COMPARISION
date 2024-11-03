import os
import time
import pickle
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama  
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader  
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

# Check if the documents file exists and is not empty
documents_file = "documents.pkl"

if os.path.exists(documents_file) and os.path.getsize(documents_file) > 0:
    # Try to load the documents from the file
    with open(documents_file, "rb") as f:
        pages = pickle.load(f)
    print("Documents loaded successfully from file.")
else:
    print("Documents file not found or empty. Using newly loaded pages.")
    # Save the documents to a pickle file
    with open(documents_file, "wb") as f:
        pickle.dump(pages, f)

# Create the vector store from the documents
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)

# Define the prompt template
template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know."

context: {context}

Question: {Question}
"""
prompt = PromptTemplate.from_template(template)

# Create vector store for document retrieval
retriever = vectorstore.as_retriever()
print("Vector store created.")

# Function to get context for a specific question
def get_context(question):
    retriever_response = retriever.invoke(question, max_results=2)

    # Ensure retriever_response is not empty and extract the first result
    if retriever_response:  # Check if retriever_response is not empty
        first_document = retriever_response[0]  # Get the first Document
        return first_document.page_content  # Access the page content directly
    else:
        return "No context found."  # Default context if no results found

# Construct the chain
chain = RunnableSequence(
    {
        "context": lambda x: get_context(x["Question"]),  # Use extracted context
        "Question": lambda x: x["Question"]  # Pass the question
    } | prompt | chat | parser  # Pass to prompt, then chat, and then parse the response
)

# Function to process a list of questions
def process_questions(questions):
    start_time = time.time()  # Start timing for all questions
    inputs = [{"Question": question} for question in questions]
    answers = chain.batch(inputs)  # Process all questions in batch
    for i, answer in enumerate(answers):
        print(f"Question {i+1}: {questions[i]}")
        print(f"Answer: {answer}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

# Collect all questions at once
questions = []
while True:
    user_input = input("Enter your question (or 'done' to finish): ")
    if user_input.lower() == 'done':
        break
    questions.append(user_input)

# Process the batch of questions
if questions:
    process_questions(questions)
else:
    print("No questions provided.")
