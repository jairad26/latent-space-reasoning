import chromadb
import torch

import numpy as np

def cosine_similarity(a, b):
  """
  Computes the cosine similarity between two numpy arrays.

  Args:
    a: A numpy array.
    b: A numpy array.

  Returns:
    The cosine similarity between a and b.
  """
  dot_product = np.dot(a, b)
  magnitude_a = np.linalg.norm(a)
  magnitude_b = np.linalg.norm(b)
  return dot_product / (magnitude_a * magnitude_b)
client = chromadb.PersistentClient("./huginn_db")

collection = client.get_or_create_collection("lam_vectors")

results = collection.get(include=["documents", "embeddings", "metadatas"])

print("cosine similarity between embeddings: ", cosine_similarity(results["embeddings"][0], results["embeddings"][1]))

print("diff between embeddings: ", results["embeddings"][0] - results["embeddings"][1])

# print(results)