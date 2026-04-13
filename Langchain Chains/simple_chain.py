from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='Qwen/Qwen3-Coder-Next',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
  template="Write about {topic}",
  input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic":"Wormhole"})

print(result)

chain.get_graph().print_ascii()