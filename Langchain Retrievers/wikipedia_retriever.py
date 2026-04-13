from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-Next",
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt = PromptTemplate(
  template = """Answer the following question based on the context given below: \n
  Context: {Context} \n
  Question: {Question} 
  """,
  input_variables=['Context', 'Question']
)

retriever = WikipediaRetriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {
  'Context': retriever | format_docs,
  'Question': RunnablePassthrough()
} | prompt | model | parser

res = chain.invoke('Dhurandhar 1')
print(res)