import argparse
from techno.Indexers import FAISSIndexer
from config import CHAT_MODEL, SIMILAR_CHUNKS, INDEX_PATH
from techno.Embedders import OpenAIEmbedder
from techno.Chatters import OpenAIChatter
from user_model.user_defined_model import get_stored_embedding_model, get_stored_token_limit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Your question")
    parser.add_argument("--chat_model", default=CHAT_MODEL, help="Chat model to use")
    parser.add_argument("--top_k", type=int, default=SIMILAR_CHUNKS, help="Number of similar chunks to retrieve")
    args = parser.parse_args()

    user_emb_model = get_stored_embedding_model()
    user_token_limit = get_stored_token_limit()

    embedder = OpenAIEmbedder(user_emb_model,user_token_limit)
    indexer = FAISSIndexer(INDEX_PATH)

    index, chunks = indexer.load_index()
    query_emb = embedder.embed_text(args.query)

    indices, _ = indexer.search_index(index, query_emb, args.top_k)
    context = "\n\n".join([chunks[i] for i in indices])
    
    chatter = OpenAIChatter(context,args.query,args.chat_model)
    answer = chatter.ask_question()
    #print("\nContext:\n", context)
    print("\nAnswer:\n", answer,'\n')
    