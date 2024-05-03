# -*- coding: utf-8 -*-

!pip -q install langchain tiktoken chromadb pypdf transformers InstructorEmbedding
!pip -q install accelerate bitsandbytes sentencepiece Xformers

!pip show langchain
!pip show transformers

import zipfile
import os

# Mount your Google Drive to Colab
from google.colab import drive
drive.mount('/content/drive')

# Create a ZipFile object with the path to the zip file
zip_file_path = '/content/drive/My Drive/training.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zipObj:

    # Create a directory to store the extracted files
    training_directory = 'training'
    if not os.path.exists(training_directory):
        os.mkdir(training_directory)

    # Use the extractall() method to extract the contents of the zip file to the specified location
    zipObj.extractall(training_directory)

# Delete the zipfile
#os.remove(zip_file_path)

# Get the list of files in the directory
# !mkdir "training"
# !ls -i "/content/drive/My Drive/calarm_training"
# !cp -r "/content/drive/My Drive/calarm_training/." "training"

!mv ./training/Calculate\ Amount\ Due\ Based\ on\ Census\ Data\ for\ Calarm\ \(1\).txt ./training/CalculateDebtFromCensus.txt
!mv ./training/Collect\ Payments\ \(1\).txt ./training/CollectPayments.txt
!mv ./training/Creating\ a\ Payment\ Plan\ and\ approval\ \(1\).txt ./training/CreatePaymentPlan.txt
!ls -i training

import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline

tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")
model = LlamaForCausalLM.from_pretrained("TheBloke/wizardLM-7B-HF",
                                                  load_in_8bit=True,
                                                  device_map='auto',
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True
                                                  )

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2000,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)

local_llm = HuggingFacePipeline(pipeline=pipe)

print(local_llm('What is the capital of Israel?'))

!pip install sentence_transformers

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Create a DirectoryLoader object
loader = DirectoryLoader("training", glob="*.txt", loader_cls=TextLoader)

# Load the text files
documents = loader.load()


# # Print the names of the loaded documents
# for document in documents:
#     with open(document.path, 'r', encoding='utf-8') as f:
#         print(document.name, f.read())

len(documents)

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceInstructEmbeddings

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## Here is the nmew embeddings being used
embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

## Cite sources

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# full example
query = "what are the steps for calculating debt from cencus"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "how do I enter in the bed days for a facility?"
llm_response = qa_chain(query)
process_llm_response(llm_response)
