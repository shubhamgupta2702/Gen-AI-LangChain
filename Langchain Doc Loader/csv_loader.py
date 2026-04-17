from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('fmnist_small_sample.csv')

docs = loader.load()

print(len(docs))