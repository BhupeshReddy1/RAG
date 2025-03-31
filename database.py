import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import os


def create_chat_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Create chats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                pdf_filename VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                top_documents JSONB,
                relevant_images BYTEA[],
                FOREIGN KEY (chat_id) REFERENCES chats(id)
            )
        """)

        conn.commit()
        return True
    except Exception as e:
        print(f"Error creating chat tables: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="rag db",
        user="postgres",
        password="admin@333",
        cursor_factory=RealDictCursor,
    )
    conn.autocommit = True
    conn.cursor_factory = psycopg2.extras.DictCursor
    return conn


def create_users_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password VARCHAR(200) NOT NULL
        )
    """)
    conn.commit()
    conn.close()