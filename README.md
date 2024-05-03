# QnATrainingKnowledge-base
# Document Processing and Question-Answering Application

## Overview

This application is designed to extract, process, and analyze text data from various documents stored in a zip archive. It primarily utilizes text generation and retrieval-based question-answering capabilities to interactively answer questions based on the document contents. It's particularly useful for scenarios where rapid processing and querying of large text datasets are required.

## Features

- **Document Extraction:** Automatically extracts documents from a zip file stored in Google Drive.
- **Text Processing:** Utilizes text generation capabilities to process and respond to queries based on the extracted text.
- **Interactive Question Answering:** Implements a retrieval-based question-answering system using embeddings and a vectorized database to answer user queries.
- **Efficient Text Splitting and Embedding:** Splits documents into manageable chunks and uses advanced embeddings for efficient retrieval.

## Installation

Ensure you have Python and pip installed on your system. Then, install the required packages using:

```bash
pip install langchain tiktoken chromadb pypdf transformers InstructorEmbedding accelerate bitsandbytes sentencepiece Xformers
```

## Usage

1. Mount your Google Drive to access the zip file containing the documents.
2. Execute the script to extract documents, process them, and set up the environment for question answering.
3. Use the interactive prompt to ask questions about the content of the documents.

## Dependencies

- **langchain:** For building LLM-based applications.
- **transformers:** For utilizing pre-trained models like LlamaForCausalLM.
- **sentence_transformers, Chroma:** For text embeddings and vector storage.
- **pypdf:** For processing PDF files.
- **InstructorEmbedding:** For specialized instruction embeddings.

## Example Queries

- "How do you createa census record?"
- "What are the steps for calculating debt from census data?"

## Contributing

Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.

## License

Specify your license or state if the project is unlicensed.

---

Adjust the sections as needed based on the specific requirements and functionalities of your application.
