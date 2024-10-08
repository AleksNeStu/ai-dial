import os
from pathlib import Path

import openai
from dotenv import find_dotenv, load_dotenv

# Load environment variables
env_file = find_dotenv(filename='.env.local')
load_dotenv(env_file, override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DIAL_URL = os.environ.get("DIAL_URL")
APP_NAME = os.environ.get

# def fib()

openai.api_key = OPENAI_API_KEY
current_dir = Path(__file__).parent.resolve()
root_dir = Path(__file__).parent.parent.resolve()