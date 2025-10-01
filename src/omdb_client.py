import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from config.config import Config

class OMDbClient:
    def __init__(self):
        self.api_key = Config.OMDB_API_KEY
        self.base_url = Config.OMDB_BASE_URL
    
    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Search for movie details by title"""
        params = {
            'apikey': self.api_key,
            't': title,
            'type': 'movie'
        }
        
        if year:
            params['y'] = year
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if data.get('Response') == 'True':
                return self._parse_movie_data(data)
            else:
                print(f"Movie not found: {title} ({year})")
                return None
                
        except Exception as e:
            print(f"Error fetching data for {title}: {e}")
            return None
    
    def _parse_movie_data(self, data: Dict) -> Dict:
        """Parse OMDb response into standardized format"""
        return {
            'Title': data.get('Title', ''),
            'Year': int(data.get('Year', '0').split('–')[0]) if data.get('Year') else 0,
            'IMDb_Rating': float(data.get('imdbRating', 0)) if data.get('imdbRating') != 'N/A' else 0,
            'Runtime_mins': int(data.get('Runtime', '0').split(' ')[0]) if data.get('Runtime') != 'N/A' else 0,
            'Genres': data.get('Genre', ''),
            'Directors': data.get('Director', ''),
            'Country': data.get('Country', ''),
            'Language': data.get('Language', ''),
            'BoxOffice': data.get('BoxOffice', 'N/A'),
            'Production': data.get('Production', 'N/A'),
            'Plot': data.get('Plot', ''),
            'Metascore': int(data.get('Metascore', 0)) if data.get('Metascore') != 'N/A' else 0,
            'imdbVotes': int(data.get('imdbVotes', '0').replace(',', '')) if data.get('imdbVotes') != 'N/A' else 0,
            'Type': data.get('Type', ''),
            'Rated': data.get('Rated', ''),
            'Released': data.get('Released', ''),
            'Actors': data.get('Actors', ''),
            'Writer': data.get('Writer', '')
        }
    
    def batch_search_movies(self, titles: List[str], delay: float = 0.2) -> pd.DataFrame:
        """Search multiple movies with delay to respect API limits"""
        movies_data = []
        
        for title in titles:
            movie_data = self.search_movie(title)
            if movie_data:
                movies_data.append(movie_data)
            time.sleep(delay)  # Be nice to the API
        
        return pd.DataFrame(movies_data)