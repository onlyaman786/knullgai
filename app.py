import os
import asyncio
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import re
import wikipedia
import requests
from bs4 import BeautifulSoup
import base64
import io
from PIL import Image
import pytesseract
import logging
import cv2
import numpy as np
from sklearn.cluster import KMeans
import ollama
import functools
import sqlite3
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up Telegram bot
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not telegram_token:
    raise ValueError("Please set the TELEGRAM_BOT_TOKEN environment variable")

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try to import the image captioning model, but don't fail if it's not available
try:
    from transformers import pipeline
    caption_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    CAPTION_MODEL_AVAILABLE = True
except ImportError:
    CAPTION_MODEL_AVAILABLE = False
    logger.warning("Image captioning model not available. Install PyTorch and transformers for this feature.")

# Available models with their characteristics
MODELS = {
    "mistral": {"size": "medium", "capabilities": ["general"]},
    "llama2": {"size": "large", "capabilities": ["general", "complex"]},
    "llava": {"size": "large", "capabilities": ["vision"]},
    "stable-diffusion": {"size": "large", "capabilities": ["image-generation"]},
}

# Set up ChromaDB
chroma_client = Client(Settings(persist_directory="./chroma_db"))
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
document_collection = chroma_client.get_or_create_collection(name="documents", embedding_function=sentence_transformer_ef)

# Set up SQLite database
def init_db():
    conn = sqlite3.connect('bot_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY, user_id INTEGER, message TEXT, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

async def store_conversation(user_id, message, response):
    conn = sqlite3.connect('bot_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (user_id, message, response) VALUES (?, ?, ?)",
              (user_id, message, response))
    conn.commit()
    conn.close()

async def get_user_history(user_id, limit=5):
    conn = sqlite3.connect('bot_data.db')
    c = conn.cursor()
    c.execute("SELECT message, response FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
              (user_id, limit))
    history = c.fetchall()
    conn.close()
    return history

def select_model(question):
    if re.search(r'\b(image|picture|photo|visual|see)\b', question, re.IGNORECASE):
        return "llava"
    if re.search(r'\b(complex|difficult|advanced)\b', question, re.IGNORECASE):
        return "llama2"
    return "mistral"

@functools.lru_cache(maxsize=100)
def query_local_model(model_name, prompt):
    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        return response['response']
    except Exception as e:
        logger.error(f"Error querying local model {model_name}: {str(e)}")
        return None

async def search_wikipedia(query):
    try:
        return await asyncio.to_thread(wikipedia.summary, query, sentences=3)
    except Exception as e:
        logger.error(f"Wikipedia search failed: {str(e)}")
        return None

async def search_duckduckgo(query):
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                results = soup.find_all('div', class_='result__body')
                return results[0].get_text().strip() if results else None
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {str(e)}")
        return None

async def extract_text_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return await asyncio.to_thread(pytesseract.image_to_string, image)
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return ""

async def generate_image_caption(image_bytes):
    if CAPTION_MODEL_AVAILABLE:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            caption = caption_model(image)[0]['generated_text']
            return caption
        except Exception as e:
            logger.error(f"Image captioning failed: {str(e)}")
    return "Image captioning not available"

def get_dominant_colors(image_bytes, num_colors=5):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.copy()
    image.thumbnail((100, 100))
    image = image.convert('RGB')
    pixels = np.array(image.getdata())
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = [f'#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}' for color in kmeans.cluster_centers_]
    return colors

async def detect_faces(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces)

async def analyze_image(image_bytes):
    caption = await generate_image_caption(image_bytes)
    dominant_colors = get_dominant_colors(image_bytes)
    face_count = await detect_faces(image_bytes)
    return {
        'caption': caption,
        'dominant_colors': dominant_colors,
        'face_count': face_count
    }

async def handle_image(update, context):
    try:
        file = await context.bot.get_file(update.message.photo[-1].file_id)
        image_bytes = await file.download_as_bytearray()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Parallel execution of OCR, image analysis, and LLaVA analysis
        extracted_text, image_analysis, llava_analysis = await asyncio.gather(
            extract_text_from_image(image_bytes),
            analyze_image(image_bytes),
            asyncio.to_thread(query_local_model, "llava", f"Describe this image in detail: data:image/jpeg;base64,{base64_image}")
        )
        
        caption = update.message.caption or "Please describe this image in detail and answer any questions about it."
        
        # Combine all image analysis results
        combined_analysis = f"""LLaVA Analysis: {llava_analysis}

Generated Image Caption: {image_analysis['caption']}
Dominant Colors: {', '.join(image_analysis['dominant_colors'])}
Number of Faces Detected: {image_analysis['face_count']}

Extracted Text: {extracted_text}

User Caption: {caption}"""

        # Search for additional information
        wiki_result, ddg_result = await asyncio.gather(
            search_wikipedia(combined_analysis),
            search_duckduckgo(combined_analysis)
        )
        
        additional_info = ""
        if wiki_result:
            additional_info += f"\n\nWikipedia says: {wiki_result}"
        if ddg_result:
            additional_info += f"\n\nAdditional information found: {ddg_result}"
        
        final_prompt = f"""{combined_analysis}

Additional Information:
{additional_info}

Please provide a comprehensive response that addresses the user's question about the image, incorporates the LLaVA analysis, the generated caption (if available), dominant colors, face detection results, any text found in the image, and includes relevant information from the additional sources."""

        ai_response = await asyncio.to_thread(query_local_model, "llama2", final_prompt)
        await update.message.reply_text(ai_response)
        
        # Store the conversation
        await store_conversation(update.effective_user.id, caption, ai_response)
    except Exception as e:
        error_message = f"An error occurred while processing the image: {str(e)}"
        logger.error(error_message)
        await update.message.reply_text(error_message)

async def start(update, context):
    await update.message.reply_text("Hello! I'm an advanced AI bot powered by local models. I can analyze text, images, chat with documents, and provide information from various sources. How can I assist you today?")

async def handle_message(update, context):
    user_message = update.message.text
    user_id = update.effective_user.id
    selected_model = select_model(user_message)
    
    try:
        # Get user history
        user_history = await get_user_history(user_id)
        context_prompt = "Previous conversations:\n" + "\n".join([f"User: {msg}\nBot: {resp}" for msg, resp in user_history])
        
        # Search documents
        doc_results = document_collection.query(query_texts=[user_message], n_results=2)
        doc_context = "\n".join([doc['document'] for doc in doc_results['documents'][0]]) if doc_results['documents'] else ""
        
        wiki_result, ddg_result = await asyncio.gather(
            search_wikipedia(user_message),
            search_duckduckgo(user_message)
        )
        
        additional_info = ""
        if wiki_result:
            additional_info += f"Wikipedia says: {wiki_result}\n\n"
        if ddg_result:
            additional_info += f"DuckDuckGo search result: {ddg_result}\n\n"
        if doc_context:
            additional_info += f"Relevant document context: {doc_context}\n\n"
        
        prompt = f"""Question: {user_message}

User History:
{context_prompt}

Additional information:
{additional_info}

Please provide a comprehensive answer based on the question, user history, and the additional information provided. 
If the question asks for current information (like exchange rates, stock prices, or recent events), 
prioritize the information from DuckDuckGo or Wikipedia as it's likely to be more up-to-date. 
If using this external information or document context, clearly state the source in your response."""

        ai_response = await asyncio.to_thread(query_local_model, selected_model, prompt)
        model_info = f"\n\n(Model used: {selected_model})"
        full_response = ai_response + model_info
        await update.message.reply_text(full_response)
        
        # Store the conversation
        await store_conversation(user_id, user_message, ai_response)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(error_message)
        await update.message.reply_text(error_message)

@functools.lru_cache(maxsize=20)
def generate_image_stable_diffusion(prompt):
    try:
        response = ollama.generate(model="stable-diffusion", prompt=prompt)
        image_data = base64.b64decode(response['response'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Stable Diffusion image generation failed: {str(e)}")
        return None

async def handle_generate_command(update, context):
    prompt = ' '.join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a prompt for image generation. For example: /generate a futuristic city")
        return

    await update.message.reply_text("Generating image... This may take a moment.")

    try:
        image = await asyncio.to_thread(generate_image_stable_diffusion, prompt)

        if image:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=buffer, caption="Generated by Stable Diffusion")

            # Generate description of the image
            description_prompt = f"Describe the following image that was generated based on this prompt: '{prompt}'. Include details about the composition, colors, and elements present in the image."
            description = await asyncio.to_thread(query_local_model, "llama2", description_prompt)
            await update.message.reply_text(f"Image Description:\n\n{description}")
        else:
            await update.message.reply_text("Sorry, image generation failed. Please try again with a different prompt.")

    except Exception as e:
        error_message = f"An error occurred during image generation: {str(e)}"
        logger.error(error_message)
        await update.message.reply_text(error_message)

async def handle_add_document(update, context):
    if not context.args:
        await update.message.reply_text("Please provide the document text after the /add_document command.")
        return
    
    document_text = ' '.join(context.args)
    document_collection.add(documents=[document_text], ids=[f"doc_{len(document_collection.get()['ids']) + 1}"])
    await update.message.reply_text("Document added successfully!")

def main():
    init_db()
    application = Application.builder().token(telegram_token).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('generate', handle_generate_command))
    application.add_handler(CommandHandler('add_document', handle_add_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.run_polling(stop_signals=None)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped manually")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
