from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
  path='books',
  glob='*.pdf',
  loader_cls=PyPDFLoader
)

# docs = loader.lazy_load()

docs = list(loader.lazy_load())

print(docs[66].page_content)