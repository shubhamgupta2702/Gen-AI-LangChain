from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='Qwen/Qwen3-Coder-Next',
  task='text-generation',
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
  template="Write a brief summary of the following text: {topic}",
  input_variables=["topic"],
)

template2 = PromptTemplate(
  template="Write a 5 line summary of the following text: {text}",
  input_variables=["text"],
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Black Hole"})

print(result)