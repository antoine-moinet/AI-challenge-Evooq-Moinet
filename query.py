import argparse
import openai  
from utils.vector_store import load_vector_store, search_index, get_stored_embedding_model
from config import OPENAI_API_KEY, CHAT_MODEL, SIMILAR_CHUNKS


openai.api_key = OPENAI_API_KEY

def ask_question(query,chat_model,top_k):
    """
    Ask a question to the LLM, find relevant chunks of texts, and returns an answer.

    Args:
        query (str): natural language query string
        chat_model (str): name of the LLM model to use
        top_k (int): number of text chunks to retrieve from index based on similarity with query

    Returns:
        tuple containing:
            LLM response
            relevance: distance of the closest chunk of text to the query, mapped to a 0-100% scale
            context: top_k chunks of text 
    """
    index, all_chunks = load_vector_store()
    user_emb_model = get_stored_embedding_model()
    indices, _, relevance = search_index(index, query, user_emb_model, top_k)
    context = "\n\n".join([all_chunks[i] for i in indices])
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

If the context does not provide enough information, respond: "I'm sorry, I couldn't find an answer in the documents."
"""
    response = openai.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, relevance, context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Your question")
    parser.add_argument("--chat_model", default=CHAT_MODEL, help="Chat model to use")
    parser.add_argument("--top_k", type=int, default=SIMILAR_CHUNKS, help="Number of similar chunks to retrieve")
    args = parser.parse_args()
    answer, relevance, context = ask_question(args.query, args.chat_model, args.top_k)
    #print(f"\nThe documents are {relevance:.1f}% relevant to your query")
    #print("\nContext:\n", context)
    print("\nAnswer:\n", answer,'\n')
    