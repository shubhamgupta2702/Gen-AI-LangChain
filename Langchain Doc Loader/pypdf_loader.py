from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path='Langchain Campusx Notes.pdf')

docs = loader.load()

print(docs[0].page_content)