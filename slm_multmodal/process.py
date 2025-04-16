# Extraind9o informaçoes dos documentos
# Import modules to handle image encoding.
import base64
import io
import time

import PIL.Image
import PIL.ImageOps
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document
from tqdm import tqdm


def conversions(sources):
    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        generate_picture_images=True,
    )
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
    }
    converter = DocumentConverter(format_options=format_options)
    # Define the sources (URLs) of the documents to be converted.
    # "https://arxiv.org/pdf/1706.03762"
    # Convert the PDF documents from the sources into an internal document format.
    conversions = {
        source: converter.convert(source=source).document for source in sources
    }

    return conversions


# Process the converted documents by splitting them into text chunks.
def chunks_text(conversions, embeddings_tokenizer):
    doc_id = 0
    texts: list[Document] = []
    for source, docling_document in conversions.items():
        # Use a hybrid chunker that leverages the tokenizer to split the document.
        for chunk in tqdm(
            HybridChunker(tokenizer=embeddings_tokenizer).chunk(
                docling_document
            ),
            desc="Processando tabelas",
        ):
            start_time = time.time()  # Marca o tempo de início da iteração
            items = chunk.meta.doc_items
            if len(items) == 1 and isinstance(items[0], TableItem):
                continue  # we will process tables later
            # Combine references from document items.
            refs = " ".join(map(lambda item: item.get_ref().cref, items))
            print(refs)
            text = chunk.text
            document = Document(
                page_content=text,
                metadata={
                    "doc_id": (doc_id := doc_id + 1),
                    "source": source,
                    "ref": refs,
                },
            )
            texts.append(document)
            end_time = time.time()  # Marca o tempo de fim da iteração
            print(f"Tempo da iteração: {end_time - start_time:.4f} segundos")
    return texts
    # Print the number of text documents created.
    print(f"{len(texts)} text documents created")


def chuck_tables(conversions, embeddings_tokenizer):
    doc_id = len(chunks_text(conversions, embeddings_tokenizer))
    tables: list[Document] = []
    for source, docling_document in conversions.items():
        for table in docling_document.tables:
            start_time = time.time()  # Marca o tempo de início da iteração
            if table.label in [DocItemLabel.TABLE]:
                ref = table.get_ref().cref
                print(ref)
                text = table.export_to_markdown()
                document = Document(
                    page_content=text,
                    metadata={
                        "doc_id": (doc_id := doc_id + 1),
                        "source": source,
                        "ref": ref,
                    },
                )
            tables.append(document)
            end_time = time.time()  # Marca o tempo de fim da iteração
            print(f"Tempo da iteração: {end_time - start_time:.4f} segundos")
    return tables
    # Print the number of table documents created.
    print(f"{len(tables)} table documents created")


def encode_image(image: PIL.Image.Image, format: str = "png") -> str:
    """
    Encode a PIL image to a base64 URI.
    This helps to embed images directly into prompts or HTML.
    """
    image = PIL.ImageOps.exif_transpose(image) or image
    image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format)
    encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
    uri = f"data:image/{format};base64,{encoding}"
    return uri


def chuck_pictures(conversions, vision_model, tokenizer, embeddings_tokenizer):
    for source, docling_document in conversions.items():
        # Set up a prompt template for processing images.
        # Feel free to experiment with this prompt
        image_prompt = ("If the image contains text,"
                        " explain the text in the image. The image is: {}")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_prompt},
                ],
            },
        ]

        # Apply the vision processor's chat template to the conversation.
        vision_prompt = tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
        )
        # Convert vision_prompt to a string if it is a list of integers
        vision_prompt = (
            tokenizer.decode(vision_prompt)
            if isinstance(vision_prompt, list)
            else vision_prompt
        )

        # Process the pictures embedded in the documents.
        pictures: list[Document] = []
        doc_id = len(chunks_text(conversions, embeddings_tokenizer)) + len(
            chuck_tables(conversions, embeddings_tokenizer)
        )
        for picture in docling_document.pictures:
            start_time = time.time()
            ref = picture.get_ref().cref
            print(ref)
            image = picture.get_image(docling_document)
            if image:
                encoded_image = encode_image(image)
                # Modify the prompt to include the encoded image
                # Insert the encoded image into the prompt.
                prompt_with_image = vision_prompt.format(encoded_image)
                text = vision_model.invoke(prompt_with_image)
                document = Document(
                    page_content=text,
                    metadata={
                        "doc_id": (doc_id := doc_id + 1),
                        "source": source,
                        "ref": ref,
                    },
                )
            pictures.append(document)
            end_time = time.time()  # Marca o tempo de fim da iteração
            print(f"Tempo da iteração: {end_time - start_time:.4f} segundos")
    return pictures
    # Print the number of image description documents created.
    print(f"{len(pictures)} image descriptions created")
