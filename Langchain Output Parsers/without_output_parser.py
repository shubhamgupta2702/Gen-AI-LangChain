from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke({'topic':"Black Hole"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result1.content})
result2 = model.invoke(prompt2)

print(result2.content)