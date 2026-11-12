# PDF Question-Answering Agent

A small Retrieval-Augmented Generation (RAG) project that searches PDF files and uses an AI model to answer questions about their contents.

## How it works

1. Extracts text from PDFs in the `pdfs/` folder.
2. Splits the text into overlapping chunks and creates embeddings.
3. Searches for relevant chunks when you ask a question.
4. Uses a DeepSeek model through OpenRouter to respond using the retrieved content.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenRouter API key

## Setup

Install the project dependencies:

```bash
uv sync
