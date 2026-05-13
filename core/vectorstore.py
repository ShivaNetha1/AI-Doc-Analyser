import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_texts(self, texts, embeddings, doc_name):
        self.index.add(np.array(embeddings).astype('float32'))
        for text in texts:
            self.metadata.append({"text": text, "source": doc_name})

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        results = []
        for i in indices[0]:
            if i != -1:
                results.append(self.metadata[i])
        return results
