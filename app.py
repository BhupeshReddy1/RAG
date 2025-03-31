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
logging.basicConfig(level=logging.DEBUG)

# Ensure database tables are created
create_users_table()

# Setup Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to extract embedded images from PDF using pdfminer
def extract_embedded_images(pdf_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = []
    image_count = 0

    image_writer = ImageWriter(output_folder)

    for page_layout in extract_pages(pdf_path):
        page_num = page_layout.pageid

        # Function to recursively extract images from layout objects
        def extract_images_from_layout(layout_obj):
            nonlocal image_count

            if isinstance(layout_obj, LTImage):
                # Direct image object
                try:
                    filename = f"image_p{page_num}_{image_count}.jpg"
                    image_path = os.path.join(output_folder, filename)

                    # Save the image using the ImageWriter
                    image_writer.export_image(layout_obj, filename)
                    image_paths.append(image_path)
                    image_count += 1

                except Exception as e:
                    app.logger.error(f"Error extracting image: {e}")

            elif isinstance(layout_obj, LTFigure):
                # Figure may contain images
                for item in layout_obj:
                    extract_images_from_layout(item)

            # If it's a container with children, process all children
            elif hasattr(layout_obj, "__iter__"):
                for item in layout_obj:
                    extract_images_from_layout(item)

        # Process the page
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
                            # Get image bytes
                            image_bytes = img["stream"].get_data()

                            # Create a filename
                            filename = f"image_p{i + 1}_{j + 1}.jpg"
                            image_path = os.path.join(output_folder, filename)

                            # Save the image
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


# Function to retrieve relevant documents and rank by relevance
def answer_question(query, document_search, image_folder):
    try:
        # Perform similarity search
        docs = document_search.similarity_search_with_score(query, k=3)

        # Log COMPLETE documents with their scores
        app.logger.debug("Retrieved documents (FULL TEXT):")
        for i, (doc, score) in enumerate(docs):
            app.logger.debug(f"COMPLETE DOCUMENT {i + 1} (Score: {score:.4f}):\n{doc.page_content}\n{'=' * 60}")

        # Check if the top document is from an image
        top_doc_is_image = False
        image_paths = []

        # Process documents to identify images and prepare display data
        display_docs = []
        for doc, score in docs[:3]:
            if is_image_document(doc.page_content):
                # Get the image path
                img_path = get_image_path_from_document(doc.page_content, image_folder)
                if img_path and os.path.exists(img_path):
                    # Record that we have an image and store its path
                    image_paths.append((img_path, score))
                    # If this is the top document, flag it
                    if doc == docs[0][0]:
                        top_doc_is_image = True
            else:
                # For regular text documents, process as before
                display_docs.append((clean_text(doc.page_content), score))

        # Use top 3 most relevant documents for answering
        top_docs = [doc for doc, _ in docs[:3]]

        # Concatenate the relevant documents for context
        context = "\n\n".join([doc.page_content for doc in top_docs])

        # Use the QA model to answer the question
        answer = qa_model(question=query, context=context)

        # Return the answer along with document info and image paths
        return answer['answer'], display_docs, image_paths, top_doc_is_image
    except Exception as e:
        app.logger.error(f"Error answering question: {e}")
        flash('An error occurred while answering the query.', 'danger')
        return None, [], [], False


# Helper functions with error handling
def get_user_chats(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # First check if the chats table exists
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
            return []

        # Now it's safe to query the table
        cursor.execute("""
            SELECT * FROM chats 
            WHERE user_id = %s 
            ORDER BY last_updated DESC
        """, (user_id,))
        chats = cursor.fetchall()
        return chats
    except Exception as e:
        app.logger.error(f"Error retrieving user chats: {e}")
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
    image_paths = []
    image_urls = []
    top_doc_is_image = False

    # Get user's chat history with error handling
    try:
        chat_history = get_user_chats(session['user_id'])
    except Exception as e:
        app.logger.error(f"Error retrieving chat history: {e}")
        chat_history = []

    # Handle creating a new chat if the button is clicked
    if form.new_chat.data:
        return redirect(url_for('new_chat'))

    # Check if we need to load a specific chat
    chat_id = request.args.get('chat_id')
    if chat_id:
        # Load the specific chat
        session['current_chat_id'] = int(chat_id)
        return redirect(url_for('home'))

    # If there's no current chat, create one or get the latest
    if 'current_chat_id' not in session or not session['current_chat_id']:
        # Either get the latest chat or leave it as None for now
        if chat_history:
            session['current_chat_id'] = chat_history[0]['id']
        else:
            session['current_chat_id'] = None

    # Get the current chat's messages if any
    current_chat_messages = []
    if session.get('current_chat_id'):
        try:
            current_chat_messages = get_chat_messages(session['current_chat_id'])
        except Exception as e:
            app.logger.error(f"Error retrieving chat messages: {e}")

    # Process new query if submitted
    if form.validate_on_submit() and form.submit.data:
        file = form.pdf.data
        query = form.query.data

        try:
            # Save the uploaded PDF
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(pdf_path)

            # Extract only embedded images from PDF
            image_paths_extracted = extract_embedded_images(pdf_path, app.config['IMAGE_FOLDER'])

            # Extract text from images - now returns a list of texts, one per image
            image_texts = extract_text_from_images(image_paths_extracted)
            app.logger.debug(f"Extracted {len(image_texts)} text documents from images")

            # Extract text from PDF
            paragraphs = extract_paragraphs(pdf_path)

            # Add each image text as a separate document/paragraph
            paragraphs.extend(image_texts)
            app.logger.debug(f"Total paragraphs after adding image texts: {len(paragraphs)}")

            # Create FAISS index
            document_search = create_faiss_index(paragraphs)

            # Answer the question - THIS IS WHERE WE GET THE RESULTS
            result, documents, image_paths, top_doc_is_image = answer_question(query, document_search,
                                                                               app.config['IMAGE_FOLDER'])

            # Store the PDF filename in the session to maintain context across queries
            session['current_pdf'] = secure_filename(file.filename)

            # Debug logging to verify what's being returned
            app.logger.debug(f"Result: {result[:100]}...")
            app.logger.debug(f"Number of documents: {len(documents)}")
            app.logger.debug(f"Number of image paths: {len(image_paths)}")

            # Create relative URLs for images to pass to the template
            image_urls = []
            for img_path, score in image_paths:
                # Extract filename from path
                filename = os.path.basename(img_path)
                # Create a URL that points to the image file
                url = url_for('static', filename=f'pdf_images/{filename}')
                image_urls.append((url, score))

            # If there's no current chat, create one before saving the message
            if not session.get('current_chat_id'):
                conn = get_db_connection()
                cursor = conn.cursor()
                try:
                    # Create a new chat with a default title
                    cursor.execute(
                        "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING id",
                        (session['user_id'], f"Chat about {query[:30]}...")
                    )
                    new_chat_id = cursor.fetchone()['id']
                    conn.commit()
                    session['current_chat_id'] = new_chat_id
                except Exception as e:
                    app.logger.error(f"Error creating new chat: {e}")
                finally:
                    cursor.close()
                    conn.close()

            # Only save the message if we have a valid chat_id
            if session.get('current_chat_id'):
                try:
                    save_message(session['current_chat_id'], query, result, secure_filename(file.filename))
                    # Reload chat history - but don't reload messages yet so we can show fresh results
                    chat_history = get_user_chats(session['user_id'])
                except Exception as e:
                    app.logger.error(f"Error saving message: {e}")

        except Exception as e:
            app.logger.error(f"Error during file upload or processing: {e}")
            flash('An error occurred while processing the file.', 'danger')

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
        app.logger.info("Database tables initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing database tables: {e}")
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['IMAGE_FOLDER']):
        os.makedirs(app.config['IMAGE_FOLDER'])
    app.run(debug=True)