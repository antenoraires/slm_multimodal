from slm_multmodal import control
from slm_multmodal import process
import itertools

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer


from dotenv import load_dotenv
import os

import logging
from huggingface_hub import login


logging.basicConfig(level=logging.INFO)

# Carrega as variáveis do .env para o ambiente
load_dotenv()

# Agora você pode acessar como se fosse uma variável de ambiente
token = os.getenv("HUG_KEY")

login(token=token)

#lista de Arquivos
sources = [
    "https://www.pwc.com/jm/en/research-publications/pdf/basic-understanding-of-a-companys-financials.pdf"
    ]

embeddings_model_path = "Snowflake/snowflake-arctic-embed-m"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

# carrehga o modelo de embeddings
model_id = "llava-hf/llava-1.5-7b-hf"  
# Carrega processador (lida com imagem + texto)
tokenizer = AutoProcessor.from_pretrained(model_id)

# Carrega o modelo (não é AutoModelForCausalLM!)
model = AutoModelForVision2Seq.from_pretrained(model_id)

conversions = process.conversions(sources)

texts = process.chunks_text(conversions, embeddings_tokenizer)
tables = process.chuck_tables(conversions, embeddings_tokenizer)
pictures = process.chuck_pictures(conversions, 
                                  model, tokenizer, 
                                  embeddings_tokenizer)

vector_db = control.vector_db_setup(embeddings_model)

# Combine all document types (text chunks, tables, and picture descriptions) into one list.
documents = list(itertools.chain(texts, tables, pictures))
# Add documents to the vector database and capture their assigned IDs.
ids = vector_db.add_documents(documents)
print(f"{len(ids)} documents added to the vector database")

rag_chain = control.rag_chain_setup(model, tokenizer, vector_db)