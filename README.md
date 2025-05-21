# PDF Question Answering AI System

## Description
A command-line tool that lets users query information contained in a folder of PDF documents using OpenAI embeddings and GPT-4o.

## Requirements
- Python 3.8+
- OpenAI API key

## Installation
'''bash
pip install -r requirements.txt
'''

## Usage
### Ingest PDF Folder:
'''bash
python ingest.py --pdf_folder <path_to_pdf_folder> --chunk_size <chunk_size> --chunk_overlap <overlap_size> --embedding_model <emb_model>
'''

- <path_to_pdf_folder>: directory containing the PDFs e.g., ./pdf_folder (the folder should be placed in the Workspace)
- <chunk_size>: number of words per chunk (default value is 500)
- <overlap_size>: number of words to overlap between chunks (default value is 50)
- <emb_model>: the model used for embeddings (default is text-embedding-3-small)


### Ask a Question:
'''bash
python query.py --query "<your_question>" --chat_model <chat_model_name> --top_k <number_of_chunks_to_use>
'''

- <your_question>: the user's natural language question
- <chat_model_name>: the model to query (default is gpt-4o)
- <number_of_chunks_to_use>: how many similar chunks to retrieve for context (default value is 5)

note that the embedding model specified at ingestion (or the default if not specified) will be stored in embedding_model.txt and used for subsequent queries (to search for relevant chunks of text in the pdfs based on similarity)


## Assumptions
- PDFs are in English
- Context length limits are respected by chunking


## Author
Antoine Moinet

