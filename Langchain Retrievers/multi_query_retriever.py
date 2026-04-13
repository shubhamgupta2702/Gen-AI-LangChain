from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-Next",
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

prompt = PromptTemplate(
  template='Generate 20 points on {topic1} and {topic2}.',
  input_variables=['topic1', 'topic2']
)

chain = prompt | model | parser

docs_text = chain.invoke({'topic1':'WormHole', 'topic2':'Black Hole'})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

split_docs = text_splitter.create_documents([docs_text])

print(docs_text)

vector_store = Chroma.from_documents(
  documents=split_docs,
  embedding=embeddings,
  collection_name='My_Collection'
)

multiquery_retriever = MultiQueryRetriever.from_llm(
  retriever=vector_store.as_retriever(search_kwargs={"k":2}),
  llm=model
)

query = 'Story of Black Hole and Wormhole'

result = multiquery_retriever.invoke(query)


for i, doc in enumerate(result):
  print(f'\n-- Result {i+1}')
  print(doc.page_content)