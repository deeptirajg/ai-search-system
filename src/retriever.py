import faiss
import numpy as np

class DocumentRetriever:
    def __init__(self, index_path):
        self.index = faiss.read_index(index_path)

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return indices[0]

if __name__ == "__main__":
    retriever = DocumentRetriever("data/faiss_index.bin")
    print(retriever.search(np.random.rand(768)))  # Test retrieval
