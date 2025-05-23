import openai  
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class OpenAIChatter:
    def __init__(self,context,query,chat_model):
        self.model = chat_model
        self.context = context
        self.query = query
    
    def ask_question(self):
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
        prompt = f"""
    You are a helpful assistant. Use the context below to answer the question.

    Context:
    {self.context}

    Question:
    {self.query}

    If the context does not provide enough information, respond: "I'm sorry, I couldn't find an answer in the documents."
    """
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
