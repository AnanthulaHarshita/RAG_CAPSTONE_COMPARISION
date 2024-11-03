import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama  
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader  

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
#MODEL="gpt-3.5-turbo"
#MODEL="mixtral"
MODEL="llama2"

if MODEL.startswith("GPT"):
    chat=ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
else:
    chat=Ollama(model=MODEL)

response = chat.invoke("Tell me a joke")  # Call invoke on the 'chat' object
#print(response)
 
parser=StrOutputParser()

chain= chat | parser
# Invoke the chain to get a joke
chain_response = chain.invoke("tell me a joke")

# Print the chain response
print(chain_response)




Loader=PyPDFLoader("mulu_training.pdf")
pages=Loader.load_and_split()
print(pages)