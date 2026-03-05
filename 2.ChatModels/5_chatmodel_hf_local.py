from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
  model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  task='text-generation',
  pipeline_kwargs={
    'max_length': 100,
    'temperature': 0.7,
  }
)

model=ChatHuggingFace(llm=llm)

res = model.invoke("Tell me about Lord Hanuman")

print(res.content)