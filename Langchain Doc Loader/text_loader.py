from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='Qwen/Qwen3-Coder-Next',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

loader = TextLoader('AI.txt', encoding='utf-8')

doc = loader.load()

# print(doc[0].page_content)
# print(doc[0].metadata)

prompt = PromptTemplate(
  template='Write a summary for this story :\n{story}',
  input_variables=['story']
)

chain = prompt | model | parser

res = chain.invoke({'story':doc[0].page_content})

print(res)