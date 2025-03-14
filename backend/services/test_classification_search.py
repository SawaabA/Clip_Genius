import openai
import tiktoken
import math
from utils import *
import numpy as np

openai.api_key = "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"

def num_of_tokens_from_string(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def split_text_tokens(text, min_chunk=275, max_chunk=325, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    if total_tokens <= max_chunk:
        return [text]
    target_chunk = (min_chunk + max_chunk) // 2
    num_chunks = math.ceil(total_tokens / target_chunk)
    base_size = total_tokens // num_chunks
    remainder = total_tokens % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        current_chunk_size = base_size + (1 if i < remainder else 0)
        chunk_tokens = tokens[start:start + current_chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += current_chunk_size
    return chunks

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding

def cosine_simularity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)