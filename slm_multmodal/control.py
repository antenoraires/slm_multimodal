import tempfile
import yaml


from IPython.display import Markdown, display
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus

def config():
    with open('src/config.yaml') as f:
        config = yaml.safe_load(f)
        return config

def vector_db_setup(embeddings_model):
    # Create a temporary file for the vector database.
    db_file = tempfile.NamedTemporaryFile(
        prefix="vectorstore_", suffix=".db", delete=False
    ).name
    print(f"The vector database will be saved to {db_file}")

    vector_db: VectorStore = Milvus(
        embedding_function=embeddings_model,
        connection_args={"uri": db_file},
        auto_id=True,
        enable_dynamic_field=True,
        index_params={"index_type": "AUTOINDEX"},
    )
    return vector_db


def rag_chain_setup(model, tokenizer, vector_db):
    """
    Set up a retrieval augmented generation (RAG) chain using LangChain.
    """
    # Create a prompt for question-answering with the retrieved context
    prompt = tokenizer.apply_chat_template(
        conversation=[
            {
                "role": "user",
                "content": "{input}",
            }
        ],
        documents=[
            {
                "title": "placeholder",
                "text": "{context}",
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_template = PromptTemplate.from_template(template=prompt)

    # Create a document prompt template to wrap each retrieved document
    document_prompt_template = PromptTemplate.from_template(
        template="""\
    Document {doc_id}
    {page_content}"""
    )
    document_separator = "\n\n"

    # Assemble the retrieval-augmented generation chain
    # Modify the document_variable_name to match your prompt
    combine_docs_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt_template,
        document_prompt=document_prompt_template,
        document_separator=document_separator,
        document_variable_name="input",
    )
    # Instead, adjust the prompt as shown earlier

    # Correct approach:
    prompt = "{input} Given the context: {context}"
    prompt_template = PromptTemplate.from_template(template=prompt)

    combine_docs_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt_template,
        document_prompt=document_prompt_template,
        document_separator=document_separator,
    )

    rag_chain = create_retrieval_chain(
        retriever=vector_db.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )
    return rag_chain


def retrivier_questions(rag_chain):
    query = input("Digite sua pergunta: ")
    outputs = rag_chain.invoke({"input": query})
    display(Markdown(outputs["answer"]))
