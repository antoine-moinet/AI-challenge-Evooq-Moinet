import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# default values for system parameters
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"
CHUNK_SIZE = 500    # 500 words
CHUNK_OVERLAP = 3   # 3 sentences overlap between chunks
SIMILAR_CHUNKS = 5  # provide top 5 similar chunks as a context in LLM prompt
TOKEN_LIMIT = 8192  # for text-embedding-3-small

# define absolute paths for vector database and user defined embedding model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index_store")
USER_EMB_PATH = os.path.join(BASE_DIR, "user_model","user_embedding_model.txt")
USER_TOK_LIM_PATH = os.path.join(BASE_DIR, "user_model","user_emb_mod_token_limit.txt")