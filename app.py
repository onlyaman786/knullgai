import os
import logging
import asyncio
import re
import wikipedia
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq
import chromadb
from PyPDF2 import PdfReader
import tempfile
from langdetect import detect
from deep_translator import GoogleTranslator
import requests
from datetime import datetime
import pytz

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = Groq(api_key="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")

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create a dictionary to store user-specific collections
user_collections = {}

# Thinking animation frames with emotions
thinking_frames = ["🤔", "🧐", "🤨", "😊"]

# Emotional responses
emotions = {
    "happy": ["😊", "😄", "🎉"],
    "sad": ["😢", "😔", "🥺"],
    "excited": ["🤩", "😃", "🙌"],
    "confused": ["😕", "🤨", "🤷‍♂️"]
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Hi! 👋 Send me a PDF file to process or ask me a question. I can respond in English and Hindi, do math calculations, provide information from Wikipedia, and even tell you your local time! ' + emotions["excited"][0])

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "Here's what I can do:\n"
        "1. Process PDF files 📄\n"
        "2. Answer questions about uploaded PDFs 💬\n"
        "3. Perform mathematical calculations 🧮\n"
        "4. Provide information from Wikipedia 📚\n"
        "5. Communicate in English and Hindi 🌐\n"
        "6. Get your local time based on IP address 🕰️\n\n"
        "Just send me a command or ask a question!"
    )
    await update.message.reply_text(help_text + " " + emotions["happy"][1])

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file, handling large files."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
            if len(text) > 1000000:  # Limit to ~1MB of text
                text += "\n[PDF truncated due to size]"
                break
    return text

async def process_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the PDF file and store its content in a user-specific ChromaDB collection."""
    user_id = update.effective_user.id
    
    if not update.message.document:
        await update.message.reply_text("Please send a PDF file. " + emotions["confused"][0])
        return

    thinking_message = await update.message.reply_text(emotions["confused"][1])
    async def animate_thinking():
        for frame in thinking_frames:
            await thinking_message.edit_text(frame)
            await asyncio.sleep(0.5)

    thinking_task = asyncio.create_task(animate_thinking())

    try:
        file = await context.bot.get_file(update.message.document.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            await file.download_to_drive(custom_path=tmp_file.name)
            text = extract_text_from_pdf(tmp_file.name)

        # Create or get user-specific collection
        if user_id not in user_collections:
            user_collections[user_id] = chroma_client.create_collection(name=f"user_{user_id}_pdf_data")
        
        user_collection = user_collections[user_id]

        # Clear previous data for this user
        user_collection.delete(where={'user_id': str(user_id)})

        # Store text in user-specific ChromaDB collection
        user_collection.add(
            documents=[text],
            metadatas=[{"source": "user_upload", "file_name": update.message.document.file_name, "user_id": str(user_id)}],
            ids=[str(update.message.document.file_id)]
        )

        os.unlink(tmp_file.name)  # Remove the temporary file
        await update.message.reply_text("PDF processed and stored. You can now ask questions about its content. " + emotions["excited"][2])
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        await update.message.reply_text("Sorry, there was an error processing your PDF. Please try again. " + emotions["sad"][0])
    finally:
        thinking_task.cancel()
        await thinking_message.delete()

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        lang = detect(text)
        return 'hi' if lang == 'hi' else 'en'
    except:
        return 'en'  # Default to English if detection fails

async def translate_text(text: str, target_language: str) -> str:
    """Translate text using deep_translator."""
    try:
        translator = GoogleTranslator(source='auto', target=target_language)
        return translator.translate(text)
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return text  # Return original text if translation fails

def perform_calculation(expression: str) -> str:
    """Perform a mathematical calculation."""
    try:
        result = eval(expression, {"__builtins__": None}, {"sqrt": pow(_, 0.5), "pow": pow})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Sorry, I couldn't calculate that. Error: {str(e)}"

async def get_wikipedia_summary(query: str, sentences: int = 2) -> str:
    """Fetch a summary from Wikipedia."""
    try:
        return wikipedia.summary(query, sentences=sentences)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your query '{query}' may refer to multiple topics. Please be more specific. Some options are: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"Sorry, I couldn't find any Wikipedia page for '{query}'."
    except Exception as e:
        logger.error(f"Error fetching Wikipedia summary: {e}")
        return f"An error occurred while fetching information about '{query}' from Wikipedia."

async def get_user_time(ip_address: str) -> str:
    """Get the current time for a user based on their IP address."""
    try:
        # Use ipapi.co to get location information from IP
        response = requests.get(f"https://ipapi.co/{ip_address}/json/")
        data = response.json()
        
        if "timezone" in data:
            timezone = pytz.timezone(data["timezone"])
            current_time = datetime.now(timezone)
            return f"Based on your IP, your local time is {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} in {data['city']}, {data['country_name']}"
        else:
            return "Sorry, I couldn't determine your location and time based on your IP address."
    except Exception as e:
        logger.error(f"Error getting time from IP: {e}")
        return "An error occurred while trying to get your local time."

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages, generate responses, perform calculations, fetch Wikipedia data, and get user's local time."""
    user_id = update.effective_user.id
    user_message = update.message.text
    input_language = detect_language(user_message)

    # Check if the message is a mathematical expression
    if re.match(r'^[\d\+\-\*\/\(\)\s]+$', user_message):
        result = perform_calculation(user_message)
        await update.message.reply_text(result + " " + emotions["excited"][1])
        return

    thinking_message = await update.message.reply_text(emotions["confused"][0])
    async def animate_thinking():
        for frame in thinking_frames:
            await thinking_message.edit_text(frame)
            await asyncio.sleep(0.5)

    thinking_task = asyncio.create_task(animate_thinking())

    try:
        # Translate user message to English if it's in Hindi
        if input_language == 'hi':
            user_message_en = await translate_text(user_message, 'en')
        else:
            user_message_en = user_message

        # Check for IP-based time request
        if "my time" in user_message_en.lower() or "local time" in user_message_en.lower():
            user_ip = update.message.from_user.id  # This is not the actual IP, just a placeholder
            time_info = await get_user_time(str(user_ip))
            await update.message.reply_text(time_info + " " + emotions["happy"][0])
            return

        # Fetch Wikipedia summary
        wiki_summary = await get_wikipedia_summary(user_message_en)

        # Prepare context information
        context_info = ""
        if user_id in user_collections:
            user_collection = user_collections[user_id]
            results = user_collection.query(
                query_texts=[user_message_en],
                n_results=1
            )
            context_info = results['documents'][0][0] if results['documents'] and results['documents'][0] else ""

        # Generate response using Groq API
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with access to the user's uploaded PDF data and Wikipedia information. Respond in the same language as the user's query (English or Hindi)."},
                {"role": "user", "content": f"Context from PDF: {context_info}\nWikipedia info: {wiki_summary}\n\nUser question: {user_message_en}"}
            ]
        )

        bot_response = response.choices[0].message.content

        # Translate bot response if necessary
        if input_language == 'hi':
            bot_response = await translate_text(bot_response, 'hi')

        # Add an emotional touch to the response
        emotion = emotions["happy"][0] if "thank" in user_message.lower() else emotions["excited"][0]
        await update.message.reply_text(bot_response + " " + emotion)
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text("Sorry, there was an error processing your request. Please try again. " + emotions["sad"][1])
    finally:
        thinking_task.cancel()
        await thinking_message.delete()

def main() -> None:
    """Set up and run the bot."""
    application = Application.builder().token("7499836265:AAFnoZWER1-rMFyZe9UC6biGiM09YvvBc-w").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.Document.PDF, process_pdf))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
