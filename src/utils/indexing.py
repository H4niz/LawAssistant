from annoy import AnnoyIndex
from models.model_config import ModelConfig

def create_index(embeddings, index_path='legal_docs.ann'):
    index = AnnoyIndex(ModelConfig.embedding_dim, 'angular')
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(10)
    index.save(index_path)
    return index

def load_index(index_path='legal_docs.ann'):
    index = AnnoyIndex(ModelConfig.embedding_dim, 'angular')
    index.load(index_path)
    return index

def get_relevant_documents(query, index, documents, manager, k=3):
    from inference.embedding import get_embedding
    query_embedding = get_embedding(query, manager)
    neighbors = index.get_nns_by_vector(query_embedding.flatten(), k)
    return [documents[i] for i in neighbors]