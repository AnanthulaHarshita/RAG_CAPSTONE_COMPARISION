import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama  
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader  
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough


# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
#MODEL="mixtral"
#MODEL="llama2"
MODEL="gpt-3.5-turbo"  # Ensure this is set correctly

if MODEL.startswith("GPT"):
    chat = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings()
else:
    chat = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings()

parser = StrOutputParser()

# Load the PDF
Loader = PyPDFLoader("mulu_training.pdf")
pages = Loader.load_and_split()

# Define the prompt template
template = """
 Answer the question based on the context below. If you can't
 answer the question, reply "I don't know."
  
 context: {context}
  
 Question: {Question}
"""
prompt = PromptTemplate.from_template(template)

# Create vector store for document retrieval
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Get retrieved context (first two results)
retriever_response = retriever.invoke("main concept", max_results=2)

# Check for valid results and extract context
if retriever_response:
    first_document = retriever_response[0]  # Get the first document
    context = first_document.page_content  # Extract the page content
    question = "What is the main concept?"  # Example question
else:
    context = "No context found."
    question = "What is the main concept?"

# Define the chain
chain = (
    {
        "context": RunnablePassthrough(),  # Allow context to pass through
        "Question": RunnablePassthrough(),  # Allow the question to pass through
    }
    | prompt  # Use the prompt with context and question
    | chat  # Pass to the chat model
    | parser  # Parse the response
)

# Invoke the chain
chain_response = chain.invoke({"context": context, "Question": question})

print(chain_response)
