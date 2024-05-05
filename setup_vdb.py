import yaml
import pickle
import time
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

DATA_PATH = config_data["VECTOR_DB_DISEASE_ENTITY_PATH"]
VECTOR_DB_NAME = config_data["VECTOR_DB_PATH"]
CHUNK_SIZE = int(config_data["VECTOR_DB_CHUNK_SIZE"])
CHUNK_OVERLAP = int(config_data["VECTOR_DB_CHUNK_OVERLAP"])
BATCH_SIZE = int(config_data["VECTOR_DB_BATCH_SIZE"])
SENTENCE_EMBEDDING_MODEL = config_data["VECTOR_DB_SENTENCE_EMBEDDING_MODEL"]


def create_vdb():
    start_time = time.time()
    with open(DATA_PATH, "rb") as pf:
        data = pickle.load(pf)
    metadata_list = list(map(lambda x: {"source": x + " from SPOKE KG"}, data))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.create_documents(data, metadatas=metadata_list)
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    vectorstore = Chroma(embedding_function=SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL), 
                         persist_directory=VECTOR_DB_NAME)
    for batch in batches:
        vectorstore.add_documents(documents=batch)
    end_time = round((time.time() - start_time)/60, 2)
    print("VectorDB is created in {} min".format(end_time))


print("Creating vectorDB ...")
create_vdb()
