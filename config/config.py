import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OMDB_API_KEY = os.getenv('OMDB_API_KEY', 'your_omdb_api_key_here')
    TMDB_API_KEY = os.getenv('TMDB_API_KEY', 'your_tmdb_api_key_here')  # ADD TMDB
    OMDB_BASE_URL = "http://www.omdbapi.com/"
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42