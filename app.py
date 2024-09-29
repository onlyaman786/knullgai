import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import requests
import wikipedia
import chromadb
from PIL import Image
import io
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models and APIs
def initialize_models():
    # Login to Hugging Face
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    # Mistral AI model
    mistral_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", use_auth_token=True)
    mistral_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", use_auth_token=True)
    
    # LLaVA model for image description
    llava_processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_auth_token=True)
    llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", use_auth_token=True)
    
    # Stable Diffusion for image generation
    sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, use_auth_token=True)
    sd_pipeline = sd_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        sd_pipeline.enable_attention_slicing()
    
    # ChromaDB for data storage
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="bot_memory")
    
    return mistral_tokenizer, mistral_model, llava_processor, llava_model, sd_pipeline, collection

mistral_tokenizer, mistral_model, llava_processor, llava_model, sd_pipeline, chroma_collection = initialize_models()

# Helper functions
def search_duckduckgo(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    return response.json()

def get_wikipedia_info(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return "No Wikipedia information found."

def generate_image(prompt):
    image = sd_pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    return image

def process_image(image):
    inputs = llava_processor(images=image, return_tensors="pt")
    outputs = llava_model.generate(**inputs, max_new_tokens=100)
    description = llava_processor.decode(outputs[0], skip_special_tokens=True)
    return description

def get_response_from_chroma(query):
    results = chroma_collection.query(query_texts=[query], n_results=1)
    if results['documents'][0]:
        return results['documents'][0][0]
    return None

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! I'm an advanced AI bot. How can I assist you today?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    chat_id = update.effective_chat.id
    
    # Check ChromaDB for existing response
    chroma_response = get_response_from_chroma(message)
    if chroma_response:
        await context.bot.send_message(chat_id=chat_id, text=chroma_response)
        return
    
    # Process the message using Mistral AI
    inputs = mistral_tokenizer(message, return_tensors="pt")
    outputs = mistral_model.generate(**inputs, max_length=100)
    response = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # If Mistral AI doesn't have a good response, search online
    if "I don't have information about that" in response.lower():
        web_info = search_duckduckgo(message)
        wiki_info = get_wikipedia_info(message)
        response = f"Here's what I found:\n\nDuckDuckGo: {web_info}\n\nWikipedia: {wiki_info}"
    
    # Store the interaction in ChromaDB
    chroma_collection.add(
        documents=[message, response],
        metadatas=[{"type": "user"}, {"type": "bot"}],
        ids=[f"user_{chat_id}_{update.message.message_id}", f"bot_{chat_id}_{update.message.message_id}"]
    )
    
    await context.bot.send_message(chat_id=chat_id, text=response)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await context.bot.get_file(update.message.photo[-1].file_id)
    f = io.BytesIO(await file.download_as_bytearray())
    image = Image.open(f)
    
    description = process_image(image)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"I see: {description}")

async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    image = generate_image(prompt)
    bio = io.BytesIO()
    image.save(bio, 'JPEG')
    bio.seek(0)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=bio)

def main():
    # Set up Telegram bot
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("generate_image", generate_image_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
