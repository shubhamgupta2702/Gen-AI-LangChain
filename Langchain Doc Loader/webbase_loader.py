from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='Qwen/Qwen3-Coder-Next',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()


url = "https://www.amazon.in/iPhone-17-256-Promotion-Resistance/dp/B0FQFYXCC4/ref=sr_1_3?sr=8-3"


loader = WebBaseLoader(web_path=url)
docs = loader.load()

prompt = PromptTemplate(
  template='Answer the following \n {question} from the following text : \n {text}',
  input_variables=['question', 'text'] 
)

chain = prompt | model | parser

res = chain.invoke({"question" : "What is the model and price of the product?","text":docs[0].page_content})


print(res)