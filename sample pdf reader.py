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
from langchain_core.runnables import RunnableSequence

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
#cMODEL="gpt-3.5-turbo"  # Ensure this is set correctly
MODEL="llama2"


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

retriever_response = retriever.invoke("main concept of Mulu", max_results=2)

# Ensure retriever_response is not empty and extract the first result
if retriever_response:  # Check if retriever_response is not empty
    first_document = retriever_response[0]  # Get the first Document
    context = first_document.page_content  # Access the page content directly
else:
    context = "No context found."  # Default context if no results found

question = "What is the main concept of the jewellry?"  # Static question for now

# Construct the chain
chain = RunnableSequence(
    {
        "context": lambda x: context,  # Use extracted context
        "Question": lambda x: question  # Pass the static question
    } | prompt | chat | parser  # Pass to prompt, then chat, and then parse the response
)

# Now invoke the chain with the extracted context and question
chain_response = chain.invoke({"context": context, "Question": question})  # Invoke chain with the extracted question

print(chain_response)
