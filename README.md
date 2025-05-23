# PDF Question Answering AI System

## Description
A command-line tool that lets users query information contained in a folder of PDF documents using OpenAI embeddings and GPT-4o.

## Requirements
- Python 3.8+
- OpenAI API key

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Provide API key:
bash
```bash
export OPENAI_API_KEY=sk-...
```
cmd
```cmd
set OPENAI_API_KEY=sk-...
```
powershell
```powershell
$env:OPENAI_API_KEY="sk-..."
```

### Ingest PDF Folder:
```bash
python ingest.py --pdf_folder <path_to_pdf_folder> --chunk_size <chunk_size> --chunk_overlap <overlap_size> --embedding_model <emb_model> --token_limit <token_limit>
```

- <path_to_pdf_folder>: directory containing the PDFs, e.g., ./papers (the folder should be placed in the main folder)
- <chunk_size>: number of words per chunk (default value is 500)
- <overlap_size>: number of sentences to overlap between chunks (default value is 3)
- <emb_model>: the model used for embeddings (must be from openai. default is text-embedding-3-small)
- <token_limit>: the max number of tokens in a single embedding request (default is 8192 for text-embedding-3-small)


### Ask a Question:
```bash
python query.py --query "<your_question>" --chat_model <chat_model_name> --top_k <number_of_chunks_to_use>
```

- <your_question>: the user's natural language question
- <chat_model_name>: the LLM used (must be from openai. default is gpt-4)
- <number_of_chunks_to_use>: how many similar chunks to retrieve for context (default value is 5)

note that the embedding model and the token limit specified at ingestion (or the defaults if not specified) will be stored in embedding_model.txt and emb_model_token_limit.txt and retrieved when query.py is run (to ensure that the query is embedded with the same embedding model)


## Assumptions
- PDFs are in English
- Context length limits are respected by chunking


## Author
Antoine Moinet

