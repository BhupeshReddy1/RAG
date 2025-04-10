from flask import Flask, render_template, redirect, url_for, flash, request, session
from werkzeug.utils import secure_filename
from database import get_db_connection, create_users_table, create_chat_tables
from forms import RegistrationForm, LoginForm, QueryForm
from bcrypt import hashpw, gensalt, checkpw
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from huggingface_hub import HfApi
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pytesseract
import logging
import os
import re
import nltk
import pdfplumber
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTImage, LTFigure, LTPage
from pdfminer.image import ImageWriter
from datetime import datetime
# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Make sure the static/pdf_images directory exists
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = os.path.join('static', 'pdf_images')  # Move images to static folder


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see everything
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Force console output
)

# Ensure database tables are created
create_users_table()

# Setup Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import weaviate
from weaviate.connect import ConnectionParams
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.query import Filter
# Initialize Weaviate client
client = weaviate.WeaviateClient(
    connection_params=ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False
    ),
    additional_config=AdditionalConfig(
        timeout=Timeout(init=30)  # Increase timeout
    ),
    skip_init_checks=True  # Skip gRPC checks
)
try:
    client.connect()
except weaviate.exceptions.WeaviateGRPCUnavailableError as e:
    app.logger.error(f"Failed to connect to Weaviate gRPC: {e}")
    print("Exiting: Could not connect to Weaviate.")
    exit(1)
except Exception as e:
    app.logger.error(f"Unexpected error connecting to Weaviate: {e}")
    exit(1)

if not client.is_ready():
    print("Exiting: Weaviate not ready.")
    exit(1)

from weaviate.classes.config import Configure, Property, DataType

def create_weaviate_schema():
    # Check if collection exists
    collections = client.collections.list_all()
    if "DocumentChunk" not in collections:
        # Create collection with updated syntax
        client.collections.create(
            name="DocumentChunk",
            description="A chunk of text from a PDF document",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The text content of the chunk"
                ),
                Property(
                    name="pdf_filename",
                    data_type=DataType.TEXT,
                    description="The filename of the source PDF",
                    skip_vectorization=True  # Replaces index_filterable/searchable
                ),
                Property(
                    name="chunk_id",
                    data_type=DataType.TEXT,
                    description="A unique identifier for the chunk within the PDF",
                    skip_vectorization=True
                ),
                Property(
                    name="embedding",
                    data_type=DataType.BLOB,
                    description="The vector embedding of the chunk",
                    skip_vectorization=True
                ),
            ]
        )
        app.logger.info("Weaviate 'DocumentChunk' collection created")
    else:
        app.logger.info("Weaviate 'DocumentChunk' collection already exists")


# Function to extract embedded images from PDF using pdfminer
def extract_embedded_images(pdf_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = []
    image_count = 0

    image_writer = ImageWriter(output_folder)

    for page_layout in extract_pages(pdf_path):
        page_num = page_layout.pageid

        def extract_images_from_layout(layout_obj):
            nonlocal image_count
            if isinstance(layout_obj, LTImage):
                try:
                    filename = f"image_p{page_num}_{image_count}.jpg"
                    image_path = os.path.join(output_folder, filename)
                    image_data = image_writer.export_image(layout_obj)
                    if isinstance(image_data, str):
                        if image_data != image_path and os.path.exists(image_data):
                            os.rename(image_data, image_path)
                    else:
                        with open(image_path, 'wb') as f:
                            if hasattr(image_data, 'getvalue'):
                                f.write(image_data.getvalue())
                            else:
                                f.write(image_data)
                    image_paths.append(image_path)
                    image_count += 1
                except Exception as e:
                    app.logger.error(f"Error extracting image: {e}")
            elif isinstance(layout_obj, LTFigure):
                for item in layout_obj:
                    extract_images_from_layout(item)
            elif hasattr(layout_obj, "__iter__"):
                for item in layout_obj:
                    extract_images_from_layout(item)

        extract_images_from_layout(page_layout)

    app.logger.debug(f"Extracted {image_count} images from {pdf_path}")
    # Alternative extraction using pdfplumber as backup
    if len(image_paths) == 0:
        app.logger.debug("No images found with pdfminer, trying pdfplumber")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    for j, img in enumerate(page.images):
                        try:
                            image_bytes = img["stream"].get_data()
                            filename = f"image_p{i + 1}_{j + 1}.jpg"
                            image_path = os.path.join(output_folder, filename)
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)
                            image_paths.append(image_path)
                            image_count += 1
                        except Exception as e:
                            app.logger.error(f"Error with pdfplumber image extraction: {e}")
        except Exception as e:
            app.logger.error(f"Error with pdfplumber extraction: {e}")

    app.logger.debug(f"Total images extracted: {len(image_paths)}")
    return image_paths


# Function to extract text from images using OCR
def extract_text_from_images(image_paths):
    image_texts = []  # List to store text from each image separately
    for img_path in image_paths:
        try:
            # Read the image
            img = Image.open(img_path)

            # Convert to grayscale for better OCR results
            if img.mode != 'L':
                img = img.convert('L')

            # Perform OCR
            text = pytesseract.image_to_string(img)

            if text.strip():  # Only add non-empty text
                # Create a document for each image with source information
                image_doc = f"From image {os.path.basename(img_path)}:\n{text}"
                image_texts.append(image_doc)
                app.logger.debug(f"Extracted text from {img_path}: {text[:100]}...")
        except Exception as e:
            app.logger.error(f"Error processing image {img_path}: {e}")

    return image_texts


# Function: Clean text
def clean_text(text):
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    # Clean up line breaks that aren't paragraph breaks
    cleaned = re.sub(r'(?<!\.)(\n)(?![A-Z])', ' ', cleaned)
    return cleaned.strip()


# Function: Extract paragraphs from PDF with improved text extraction
# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Improved text extraction function
def extract_paragraphs(pdf_path):
    try:
        # Extract raw text from PDF
        pdfreader = PdfReader(pdf_path)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content + " "

        # Simple sentence splitting using regex
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', raw_text)

        # Group sentences into chunks, ensuring complete sentences
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed our target size and we already have content
            if len(current_chunk) + len(sentence) > 800 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Log a few chunks to verify
        for i, chunk in enumerate(chunks[:3]):
            app.logger.debug(f"Chunk {i + 1} (complete): {chunk}")

        return chunks
    except Exception as e:
        app.logger.error(f"Error extracting paragraphs from PDF: {e}")
        raise


def is_image_document(doc_content):
    # Check if the document starts with "From image" which we added in the extract_text_from_images function
    return doc_content.startswith("From image ")


def get_image_path_from_document(doc_content, image_folder):
    try:
        # Extract the image filename - it's in the format "From image image_p1_0.jpg:"
        match = re.search(r"From image (image_p\d+_\d+\.jpg):", doc_content)
        if match:
            image_filename = match.group(1)
            return os.path.join(image_folder, image_filename)
    except Exception as e:
        app.logger.error(f"Error extracting image path: {e}")
    return None


# Function to retrieve relevant documents from Weaviate
from weaviate.classes.query import MetadataQuery

def answer_question(query, pdf_filename, image_folder):
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embeddings_model.embed_query(query)

        document_chunk_collection = client.collections.get("DocumentChunk")
        if document_chunk_collection is None:
            app.logger.error("Weaviate 'DocumentChunk' collection not found.")
            return None, [], [], False

        response = document_chunk_collection.query.near_vector(
            near_vector=query_embedding,
            limit=3,
            filters=weaviate.classes.query.Filter.by_property("pdf_filename").equal(pdf_filename),
            return_metadata=MetadataQuery(distance=True)  # Request distance metadata
        )

        relevant_docs = []
        for obj in response.objects:
            relevant_docs.append((obj.properties["content"], obj.metadata.distance))

        app.logger.debug("Retrieved documents (FULL TEXT) from Weaviate:")
        for i, (doc, score) in enumerate(relevant_docs):
            app.logger.debug(f"COMPLETE DOCUMENT {i + 1} (Score: {score:.4f}):\n{doc}\n{'=' * 60}")

        top_doc_is_image = False
        image_paths = []
        display_docs = []

        for doc, score in relevant_docs[:3]:
            if is_image_document(doc):
                img_path = get_image_path_from_document(doc, image_folder)
                if img_path and os.path.exists(img_path):
                    image_paths.append((img_path, score))
                    if doc == relevant_docs[0][0]:
                        top_doc_is_image = True
            else:
                display_docs.append((clean_text(doc), score))

        top_docs_content = [doc for doc, _ in relevant_docs[:3]]
        context = "\n\n".join(top_docs_content)
        answer = qa_model(question=query, context=context)

        return answer['answer'], display_docs, image_paths, top_doc_is_image
    except Exception as e:
        app.logger.error(f"Error answering question using Weaviate: {type(e).__name__}: {str(e)}")
        flash('An error occurred while answering the query.', 'danger')
        return None, [], [], False


collection = client.collections.get("DocumentChunk")
response = collection.query.fetch_objects(limit=3)
for obj in response.objects:
    print(obj.properties["content"])

# Helper functions with error handling
def get_user_chats(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'chats'
            )
        """)
        table_exists = cursor.fetchone()[0]
        if not table_exists:
            app.logger.info("Chats table does not exist, creating it.")
            create_chat_tables()
            return []
        cursor.execute("""
            SELECT id, user_id, title, last_updated FROM chats 
            WHERE user_id = %s 
            ORDER BY last_updated DESC
        """, (user_id,))
        rows = cursor.fetchall()
        chats = [{'id': row[0], 'user_id': row[1], 'title': row[2], 'last_updated': row[3]} for row in rows]
        app.logger.debug(f"Retrieved {len(chats)} chats for user_id {user_id}")
        return chats
    except Exception as e:
        app.logger.error(f"Error retrieving user chats: {type(e).__name__}: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_chat_messages(chat_id):
    if not chat_id:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if the messages table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'messages'
            )
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            # If table doesn't exist, create it
            create_chat_tables()
            return []

        # Now it's safe to query the table
        cursor.execute("""
            SELECT * FROM messages 
            WHERE chat_id = %s 
            ORDER BY created_at ASC
        """, (chat_id,))
        messages = cursor.fetchall()
        return messages
    except Exception as e:
        app.logger.error(f"Error retrieving chat messages: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def save_message(chat_id, query, answer, pdf_filename):
    if not chat_id:
        return False

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if the messages table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'messages'
            )
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            # If table doesn't exist, create it
            create_chat_tables()

        # Save the message
        cursor.execute("""
            INSERT INTO messages (chat_id, query, answer, pdf_filename)
            VALUES (%s, %s, %s, %s)
        """, (chat_id, query, answer, pdf_filename))

        # Update the last_updated timestamp of the chat
        cursor.execute("""
            UPDATE chats 
            SET last_updated = CURRENT_TIMESTAMP 
            WHERE id = %s
        """, (chat_id,))

        conn.commit()
        return True
    except Exception as e:
        app.logger.error(f"Error saving message: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def update_chat_title(chat_id, title):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE chats 
            SET title = %s 
            WHERE id = %s
        """, (title, chat_id))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


# Function: Create FAISS index
def create_faiss_index(paragraphs):
    try:
        # Authenticate using the token
        hf_token = os.getenv('HF_TOKEN')

        # If you have a token, validate it
        if hf_token:
            try:
                api = HfApi()
                user_info = api.whoami(token=hf_token)
                print(f"Authenticated as: {user_info.get('name', 'Unknown User')}")
            except Exception as auth_error:
                print(f"Authentication failed: {auth_error}")

        # Use a universally accessible embedding model
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Create a FAISS vector store from the chunks and embeddings
        document_search = FAISS.from_texts(paragraphs, embeddings)
        return document_search
    except Exception as e:
        app.logger.error(f"Error creating FAISS index: {e}")
        raise


# QA Model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2",
                    tokenizer="deepset/roberta-base-squad2")


# Route: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        conn = get_db_connection()
        cursor = conn.cursor()

        username = form.username.data
        email = form.email.data
        password = hashpw(form.password.data.encode('utf-8'), gensalt()).decode('utf-8')

        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, password),
            )
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            app.logger.error(f"Error during registration: {e}")
            flash('Error: ' + str(e), 'danger')
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html', form=form)


# Route: Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        conn = get_db_connection()
        cursor = conn.cursor()

        email = form.email.data
        password = form.password.data

        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if user and checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                session['user_id'] = user['id']
                session['username'] = user['username']
                return redirect(url_for('home'))
            else:
                flash('Invalid email or password.', 'danger')

        except Exception as e:
            app.logger.error(f"Error during login: {e}")
            flash('An error occurred while processing your login.', 'danger')
        finally:
            cursor.close()
            conn.close()

    return render_template('login.html', form=form)


@app.route('/new_chat', methods=['GET', 'POST'])
def new_chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Create a new chat in the database
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check if the chats table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'chats'
            )
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            # If table doesn't exist, create it
            create_chat_tables()

        # Create a new chat with a default title
        cursor.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING id",
            (session['user_id'], f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        )
        new_chat_id = cursor.fetchone()['id']
        conn.commit()

        # Store the current chat ID in the session
        session['current_chat_id'] = new_chat_id

        # Redirect to home with a clean form
        return redirect(url_for('home'))

    except Exception as e:
        app.logger.error(f"Error creating new chat: {e}")
        flash('An error occurred while creating a new chat.', 'danger')

    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('home'))


@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    form = QueryForm()
    result = None
    documents = []
    image_urls = []
    top_doc_is_image = False
    chat_history = get_user_chats(session['user_id'])
    current_chat_messages = []
    if session.get('current_chat_id'):
        current_chat_messages = get_chat_messages(session['current_chat_id'])

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if form.new_chat.data:
        return redirect(url_for('new_chat'))

    chat_id = request.args.get('chat_id')
    if chat_id:
        session['current_chat_id'] = int(chat_id)
        return redirect(url_for('home'))

    if 'current_chat_id' not in session or not session['current_chat_id']:
        if chat_history:
            session['current_chat_id'] = chat_history[0]['id']
        else:
            session['current_chat_id'] = None

    if form.validate_on_submit() and form.submit.data:
        file = form.pdf.data
        query = form.query.data
        app.logger.debug(f"Form submitted: file={file}, query={query}")

        try:
            if file:
                pdf_filename = secure_filename(file.filename)
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
                file.save(pdf_path)
                session['current_pdf'] = pdf_filename
                app.logger.info(f"PDF uploaded: {pdf_filename}")

                # Check if chunks already exist in Weaviate
                document_chunk_collection = client.collections.get("DocumentChunk")
                response = document_chunk_collection.query.fetch_objects(
                    filters=weaviate.classes.query.Filter.by_property("pdf_filename").equal(pdf_filename),
                    limit=1
                )
                if not response.objects:  # If no chunks exist, process the PDF
                    image_paths_extracted = extract_embedded_images(pdf_path, app.config['IMAGE_FOLDER'])
                    image_texts = extract_text_from_images(image_paths_extracted)
                    paragraphs = extract_paragraphs(pdf_path)
                    all_documents = paragraphs + image_texts

                    if document_chunk_collection is None:
                        app.logger.error("Weaviate 'DocumentChunk' collection not found.")
                        flash('Error: Could not store document chunks.', 'danger')
                    else:
                        with document_chunk_collection.batch.dynamic() as batch:
                            for i, doc in enumerate(all_documents):
                                embedding = embeddings_model.embed_query(doc)
                                chunk_id = f"{pdf_filename}_chunk_{i}"
                                batch.add_object(
                                    properties={
                                        "content": doc,
                                        "pdf_filename": pdf_filename,
                                        "chunk_id": chunk_id,
                                    },
                                    vector=embedding
                                )
                        if document_chunk_collection.batch.failed_objects:
                            app.logger.error(f"Batch insertion failed: {document_chunk_collection.batch.failed_objects}")
                            flash('Error: Some document chunks failed to upload.', 'danger')
                        else:
                            flash(f"PDF '{pdf_filename}' processed and stored in Weaviate.", 'success')
                else:
                    app.logger.info(f"PDF '{pdf_filename}' already processed, skipping reprocessing.")

            if query and session.get('current_pdf'):
                app.logger.debug(f"Processing query: {query} for PDF: {session['current_pdf']}")
                result, documents, image_paths, top_doc_is_image = answer_question(
                    query, session['current_pdf'], app.config['IMAGE_FOLDER']
                )
                app.logger.debug(f"Query result: {result}, Documents: {len(documents)}, Images: {len(image_paths)}")
                image_urls = []
                for img_path, score in image_paths:
                    filename = os.path.basename(img_path)
                    url = url_for('static', filename=f'pdf_images/{filename}')
                    image_urls.append((url, score))
                if session.get('current_chat_id'):
                    save_message(session['current_chat_id'], query, result, session['current_pdf'])
                    chat_history = get_user_chats(session['user_id'])

        except Exception as e:
            app.logger.error(f"Error during file upload or query with Weaviate: {type(e).__name__}: {str(e)}")
            flash('An error occurred while processing the file or query.', 'danger')

    app.logger.debug(f"Rendering home: result={result}, documents={len(documents)}, image_urls={len(image_urls)}")
    return render_template('home.html', form=form, result=result, documents=documents,
                           image_urls=image_urls, top_doc_is_image=top_doc_is_image,
                           chat_history=chat_history, current_chat_id=session.get('current_chat_id'),
                           current_chat_messages=current_chat_messages, show_fresh_result=form.submit.data)
# Route: Welcome
@app.route('/', methods=['GET'])
def welcome():
    return render_template('base.html')


# Route: Logout
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()  # Clears the session data
    return redirect(url_for('login'))  # Redirect to login page after logging out


# Run the Flask application
if __name__ == '__main__':
    try:
        create_users_table()
        create_chat_tables()
        create_weaviate_schema()
        app.logger.info("Database tables initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing database tables: {e}")
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['IMAGE_FOLDER']):
        os.makedirs(app.config['IMAGE_FOLDER'])
    try:
        app.run(debug=True)
    finally:
        client.close()
