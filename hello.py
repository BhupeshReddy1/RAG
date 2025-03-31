from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Read the PDF file
pdfreader = PdfReader('budget_speech.pdf')

# Extract text from the PDF
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Split the text into paragraphs based on double newlines
paragraphs = raw_text.split(".\n")

# Debug: Print the first few paragraphs to verify splitting
print(f"Total paragraphs: {len(paragraphs)}")


# Use a pre-trained SentenceTransformer model for embeddings (e.g., `paraphrase-MiniLM-L6-v2`)
embedding_model_name = "paraphrase-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Create a FAISS vector store from the extracted texts and embeddings
document_search = FAISS.from_texts(paragraphs, embeddings)

# Load a question-answering pipeline using a Hugging Face model (e.g., BART or T5)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")


# Function to perform similarity search and generate answers
def answer_question(query):
    # Search for the most relevant documents
    docs = document_search.similarity_search(query, k=1)  # Retrieve top 3

    # Debug: Print the retrieved documents
    x=""
    for i, doc in enumerate(docs):
        x+=doc.page_content[:300]
    print(x)

    # Concatenate the relevant documents into a single text block
    context = "\n".join([doc.page_content for doc in docs])
    context = context[:1000]  # Ensure context size is manageable for the QA model

    # Use the QA model to answer the question based on the retrieved context
    answer = qa_model(question=query, context=context)

    # Return the answer limiting to 50 words
    return ' '.join(answer['answer'].split()[:100])

# Example query
query1 = input()
print(answer_question(query1))

