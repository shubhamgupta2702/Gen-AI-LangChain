from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='Qwen/Qwen3-Coder-Next',
  task='text-generation'
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
  template="generate short and simple note on the following text : {text} ",
  input_variables=['text']
)

prompt2 = PromptTemplate(
  template="generate 5 questions answers from the following text :{text}",
  input_variables=['text']
)

prompt3 = PromptTemplate(
  template="merge the provided notes and quiz into single document: notes -> {notes} and quiz -> {quiz}",
  input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
  'notes': prompt1 | model1 | parser,
  'quiz': prompt2 | model2 | parser
})

prompt3_chain = prompt3 | model1 | parser

merge_chain = parallel_chain | prompt3_chain    #parallel_chain output -> prompt3

result = merge_chain.invoke({'text':'Space'})

print(result)
merge_chain.get_graph().print_ascii()