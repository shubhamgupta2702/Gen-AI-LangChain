from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='Qwen/Qwen3-Coder-Next',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

class Feedback(BaseModel):
  sentiment: Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")
  
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
  template='Classify the following sentiment from the given feedback \n {feedback} \n {format_instruction}',
  input_variables=['feedback'],
  partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# res = classifier_chain.invoke({'feedback':'This is a best laptop.'})
# print(res.sentiment)

prompt2 = PromptTemplate(
  template="Write an appropriate response for this positive feedback \n {feedback}",
  input_variables=['feedback']
)

prompt3 = PromptTemplate(
  template="Write an appropriate response for this negative feedback \n {feedback}",
  input_variables=['feedback']
)

branch_chain = RunnableBranch(
  (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
  (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
  RunnableLambda(lambda x:'could not find any sentiment it may be neutral.')
)

final_chain = classifier_chain | branch_chain

res = final_chain.invoke({'feedback':'This is a below Average Laptop for coding.'})

print(res)

final_chain.get_graph().print_ascii()