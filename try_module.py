import itertools
import logging
import os

from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from slm_multmodal import control, process

logging.basicConfig(level=logging.INFO)

# Key Model
load_dotenv()
token = os.getenv("HUG_KEY")
login(token=token)

config = control.config()
# lista de Arquivos
sources = config["sources"]

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
pictures = process.chuck_pictures(
    conversions, model, tokenizer, embeddings_tokenizer
)

vector_db = control.vector_db_setup(embeddings_model)

documents = list(itertools.chain(texts, tables, pictures))
# Add documents to the vector database and capture their assigned IDs.
ids = vector_db.add_documents(documents)
print(f"{len(ids)} documents added to the vector database")

rag_chain = control.rag_chain_setup(model, tokenizer, vector_db)

# Define the input query
control.retrivier_questions(rag_chain=rag_chain)
