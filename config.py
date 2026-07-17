from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()                      
client = OpenAI()


EMBED_MODEL = "text-embedding-3-small"   
CHAT_MODEL = "gpt-4.1-nano"                 

CHUNK_SIZE = 800                   
OVERLAP_SENTENCES = 1              
TOP_K = 4                          
SCORE_THRESHOLD = 0.25             

DOCS_FOLDER = "docs"               
COMPANY_NAME = "JournalSay"