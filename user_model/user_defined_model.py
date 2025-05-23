import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import USER_EMB_PATH, USER_TOK_LIM_PATH

def store_user_model(emb_model,token_limit):
    os.makedirs(os.path.dirname(USER_EMB_PATH), exist_ok=True)
    with open(USER_EMB_PATH, "w") as f:
        f.write(emb_model)
    os.makedirs(os.path.dirname(USER_TOK_LIM_PATH), exist_ok=True)
    with open(USER_TOK_LIM_PATH, "w") as f:
        f.write(str(token_limit))
    return 

def get_stored_embedding_model():
    """
    Reads and returns the saved embedding model provided by the user at ingestion
    """
    if not os.path.exists(USER_EMB_PATH):
        raise FileNotFoundError(f"User embedding model file not found: {USER_EMB_PATH}")
    with open(USER_EMB_PATH, "r") as f:
        model = f.read().strip()
        if not model:
            raise ValueError("User embedding model file is empty.")
        return model
    
def get_stored_token_limit():
    """
    Reads and returns the saved token limit provided by the user at ingestion
    """
    if not os.path.exists(USER_TOK_LIM_PATH):
        raise FileNotFoundError(f"User token limit file not found: {USER_TOK_LIM_PATH}")
    with open(USER_TOK_LIM_PATH, "r") as f:
        tl = int(f.read())
        if not tl:
            raise ValueError("User token limit file is empty.")
        return tl