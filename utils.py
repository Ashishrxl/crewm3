import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    _ = load_dotenv(find_dotenv())

def get_gemini_api_key():
    load_env()
    return os.getenv("GEMINI_API_KEY")

def get_exa_api_key():
    load_env()
    return os.getenv("EXA_API_KEY")
