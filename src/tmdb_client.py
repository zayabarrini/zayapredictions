import requests
import os
from typing import List, Dict, Optional
from config.config import Config

class TMDBClient:
    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
    
    def search_movie(self, title: str, year: Optional[int] = None, original_title: str = None) -> Optional[Dict]:
        """Enhanced search for international movies with multiple fallback strategies"""
        if not self.api_key or self.api_key == 'your_tmdb_api_key_here':
            return None
            
        strategies = [
            self._search_by_title_year,
            self._search_by_original_title,
            self._search_fuzzy_match,
            self._search_without_year
        ]
        
        for strategy in strategies:
            result = strategy(title, year, original_title)
            if result:
                print(f"✅ TMDB: Found '{title}' using {strategy.__name__}")
                return result
        
        print(f"❌ TMDB: Movie not found after all strategies: {title} ({year})")
        return None
    
    def _search_by_title_year(self, title: str, year: Optional[int] = None, original_title: str = None) -> Optional[Dict]:
        """Search by title and year (most precise)"""
        params = {
            'api_key': self.api_key,
            'query': title,
            'language': 'en-US'
        }
        
        if year:
            params['year'] = year
        
        try:
            response = requests.get(f"{self.base_url}/search/movie", params=params, timeout=10)
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                movie_id = data['results'][0]['id']
                return self._get_movie_details(movie_id)
        except Exception as e:
            print(f"⚠️ TMDB search error for {title}: {e}")
        
        return None
    
    def _search_by_original_title(self, title: str, year: Optional[int] = None, original_title: str = None) -> Optional[Dict]:
        """Search by original title if available"""
        if not original_title or original_title == title:
            return None
            
        params = {
            'api_key': self.api_key,
            'query': original_title,
            'language': 'en-US'
        }
        
        if year:
            params['year'] = year
        
        try:
            response = requests.get(f"{self.base_url}/search/movie", params=params, timeout=10)
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                movie_id = data['results'][0]['id']
                return self._get_movie_details(movie_id)
        except Exception:
            pass
        
        return None
    
    def _search_fuzzy_match(self, title: str, year: Optional[int] = None, original_title: str = None) -> Optional[Dict]:
        """Try fuzzy matching by removing common prefixes and being less strict"""
        # Remove common prefixes that might differ between databases
        clean_title = self._clean_title(title)
        
        params = {
            'api_key': self.api_key,
            'query': clean_title,
            'language': 'en-US'
        }
        
        try:
            response = requests.get(f"{self.base_url}/search/movie", params=params, timeout=10)
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                # Try to find the best match
                for result in data['results']:
                    result_title = result.get('title', '').lower()
                    result_original = result.get('original_title', '').lower()
                    clean_result = self._clean_title(result_title)
                    
                    # Check if titles are similar
                    if (clean_title in result_title or clean_title in result_original or 
                        result_title in clean_title or self._titles_similar(clean_title, clean_result)):
                        movie_id = result['id']
                        return self._get_movie_details(movie_id)
        except Exception:
            pass
        
        return None
    
    def _search_without_year(self, title: str, year: Optional[int] = None, original_title: str = None) -> Optional[Dict]:
        """Search without year constraint"""
        params = {
            'api_key': self.api_key,
            'query': title,
            'language': 'en-US'
        }
        
        try:
            response = requests.get(f"{self.base_url}/search/movie", params=params, timeout=10)
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                # Get the first result (most popular)
                movie_id = data['results'][0]['id']
                return self._get_movie_details(movie_id)
        except Exception:
            pass
        
        return None
    
    def _clean_title(self, title: str) -> str:
        """Clean title for better matching"""
        if not title:
            return ""
        
        # Convert to lowercase
        clean = title.lower()
        
        # Remove common prefixes
        prefixes = ['the ', 'la ', 'le ', 'el ', 'der ', 'die ', 'das ']
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        
        # Remove punctuation and extra spaces
        import re
        clean = re.sub(r'[^\w\s]', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.7) -> bool:
        """Check if two titles are similar using simple string comparison"""
        if not title1 or not title2:
            return False
        
        # Simple similarity check
        set1 = set(title1.split())
        set2 = set(title2.split())
        
        if not set1 or not set2:
            return False
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return False
            
        similarity = intersection / union
        return similarity >= threshold
    
    def _get_movie_details(self, movie_id: int) -> Dict:
        """Get detailed movie information including keywords"""
        try:
            # Get movie details with multiple endpoints
            details_response = requests.get(
                f"{self.base_url}/movie/{movie_id}",
                params={'api_key': self.api_key, 'language': 'en-US'},
                timeout=10
            )
            details_data = details_response.json()
            
            # Get keywords
            keywords_response = requests.get(
                f"{self.base_url}/movie/{movie_id}/keywords",
                params={'api_key': self.api_key},
                timeout=10
            )
            keywords_data = keywords_response.json()
            
            # Get alternative titles (for international films)
            alt_titles_response = requests.get(
                f"{self.base_url}/movie/{movie_id}/alternative_titles",
                params={'api_key': self.api_key},
                timeout=10
            )
            alt_titles_data = alt_titles_response.json()
            
            # Extract relevant information
            tmdb_data = {
                'tmdb_id': movie_id,
                'tmdb_title': details_data.get('title', ''),
                'tmdb_original_title': details_data.get('original_title', ''),
                'tmdb_overview': details_data.get('overview', ''),
                'keywords': self._extract_keywords(keywords_data),
                'alternative_titles': self._extract_alternative_titles(alt_titles_data),
                'tagline': details_data.get('tagline', ''),
                'budget': details_data.get('budget', 0),
                'revenue': details_data.get('revenue', 0),
                'popularity': details_data.get('popularity', 0),
                'vote_average': details_data.get('vote_average', 0),
                'vote_count': details_data.get('vote_count', 0),
                'production_companies': self._extract_production_companies(details_data),
                'production_countries': self._extract_production_countries(details_data),
                'spoken_languages': self._extract_spoken_languages(details_data),
                'genres': self._extract_genres(details_data),
                'runtime': details_data.get('runtime', 0),
            }
            
            return tmdb_data
            
        except Exception as e:
            print(f"❌ TMDB Error fetching details for movie {movie_id}: {e}")
            return {}
    
    def _extract_keywords(self, keywords_data: Dict) -> str:
        """Extract keywords from TMDB response"""
        keywords = []
        if 'keywords' in keywords_data:
            for keyword in keywords_data['keywords']:
                keywords.append(keyword['name'])
        return ', '.join(keywords) if keywords else ''
    
    def _extract_alternative_titles(self, alt_titles_data: Dict) -> str:
        """Extract alternative titles for international films"""
        titles = []
        if 'titles' in alt_titles_data:
            for title_info in alt_titles_data['titles']:
                titles.append(f"{title_info['title']} ({title_info.get('iso_3166_1', '')})")
        return '; '.join(titles) if titles else ''
    
    def _extract_production_companies(self, details_data: Dict) -> str:
        """Extract production companies"""
        companies = []
        if 'production_companies' in details_data:
            for company in details_data['production_companies']:
                companies.append(company['name'])
        return ', '.join(companies) if companies else ''
    
    def _extract_production_countries(self, details_data: Dict) -> str:
        """Extract production countries"""
        countries = []
        if 'production_countries' in details_data:
            for country in details_data['production_countries']:
                countries.append(country['name'])
        return ', '.join(countries) if countries else ''
    
    def _extract_spoken_languages(self, details_data: Dict) -> str:
        """Extract spoken languages"""
        languages = []
        if 'spoken_languages' in details_data:
            for language in details_data['spoken_languages']:
                languages.append(language['english_name'])
        return ', '.join(languages) if languages else ''
    
    def _extract_genres(self, details_data: Dict) -> str:
        """Extract genres"""
        genres = []
        if 'genres' in details_data:
            for genre in details_data['genres']:
                genres.append(genre['name'])
        return ', '.join(genres) if genres else ''