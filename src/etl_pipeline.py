import pandas as pd
import faiss
import numpy as np

def process_data(file_path):
    df = pd.read_csv(file_path)
    embeddings = np.random.rand(len(df), 768)  # Simulating embeddings
    index = faiss.IndexFlatL2(768)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    faiss_index = process_data("data/documents.csv")
    faiss.write_index(faiss_index, "data/faiss_index.bin")
