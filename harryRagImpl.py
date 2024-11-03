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

# Define the model to use
#MODEL = "gpt-3.5-turbo"
#MODEL = "mixtral"
MODEL = "llama2"

# Model Initialization
def initialize_model(model_name):
    start_time = time.time()
    if model_name.startswith("gpt"):
        api_key = os.getenv("OPENAI_KEY")
        chat = ChatOpenAI(api_key=api_key, model=model_name)
        embeddings = OpenAIEmbeddings(api_key=api_key)
    else:
        chat = Ollama(model=model_name)
        embeddings = OllamaEmbeddings()
    print(f"Model initialization time: {time.time() - start_time:.2f} seconds")
    return chat, embeddings

chat, embeddings = initialize_model(MODEL)
parser = StrOutputParser()

# PDF Loading
def load_pdf(file_path, cache_file):
    start_time = time.time()
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        with open(cache_file, "rb") as f:
            pages = pickle.load(f)
        print("Documents loaded from cache.")
    else:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        with open(cache_file, "wb") as f:
            pickle.dump(pages, f)
        print("Documents loaded from PDF and cached.")
    print(f"PDF loading time: {time.time() - start_time:.2f} seconds")
    return pages

# Load PDF and cache documents
pages = load_pdf("harrypotter.pdf", "documents.pkl")

# Create vector store from the documents
start_time = time.time()
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()
print(f"Vector store creation time: {time.time() - start_time:.2f} seconds")

# Define prompt template
prompt_template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know."

context: {context}

Question: {Question}
"""
prompt = PromptTemplate.from_template(prompt_template)

# Function to get context for a specific question
def get_context(question):
    start_time = time.time()
    retriever_response = retriever.invoke(question, max_results=2)
    elapsed_time = time.time() - start_time

    if retriever_response:  
        first_document = retriever_response[0]
        context = first_document.page_content
        print(f"Context retrieval time for '{question}': {elapsed_time:.2f} seconds")
        return context
    else:
        print(f"No context found for '{question}' in {elapsed_time:.2f} seconds.")
        return "No context found."

# Construct the chain
chain = RunnableSequence(
    {
        "context": lambda x: get_context(x["Question"]),
        "Question": lambda x: x["Question"]
    } | prompt | chat | parser
)

# Function to process a list of questions
def process_questions(questions):
    start_time = time.time()
    inputs = [{"Question": question} for question in questions]
    answers = chain.batch(inputs)
    for i, answer in enumerate(answers):
        print(f"Question {i+1}: {questions[i]}")
        print(f"Answer: {answer}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

# Collect all questions at once
def collect_questions():
    questions = []
    while True:
        user_input = input("Enter your question (or 'done' to finish): ")
        if user_input.lower() == 'done':
            break
        questions.append(user_input)
    return questions

# Main execution
if __name__ == "__main__":
    questions = collect_questions()
    if questions:
        process_questions(questions)
    else:
        print("No questions provided.")
