import os
from dotenv import load_dotenv

load_dotenv()


openweather_api = os.getenv("OPENWEATHER_API_KEY")
