import argparse
import openai  
import pickle
from utils.vector_store import load_vector_store, search_index
from config import OPENAI_API_KEY, CHAT_MODEL, SIMILAR_CHUNKS

with open('embedding_model.txt', "r") as f:
    USER_EMB_MODEL = f.read().strip()
    if not USER_EMB_MODEL:
        raise ValueError("User Embedding model config file is empty.")
    
with open("vector_store/chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

print('1')

openai.api_key = OPENAI_API_KEY

print('2')

def ask_question(query,chat_model,top_k):
    print('3')
    index = load_vector_store()
    print('4')
    indices, _ = search_index(index, query, USER_EMB_MODEL, top_k)
    print(indices)
    context = "\n".join([all_chunks[i] for i in indices])
    print('6')
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

If the context does not provide enough information, respond: "I'm sorry, I couldn't find an answer in the documents."
"""
    print(prompt)
    response = openai.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Your question")
    parser.add_argument("--chat_model", default=CHAT_MODEL, help="Chat model to use")
    parser.add_argument("--top_k", type=int, default=SIMILAR_CHUNKS, help="Number of similar chunks to retrieve")
    args = parser.parse_args()
    answer, context = ask_question(args.query, args.chat_model, args.top_k)
    print("\nAnswer:\n", answer)
    #print("\nContext:\n", context)