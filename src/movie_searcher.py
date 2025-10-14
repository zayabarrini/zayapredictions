#!/usr/bin/env python3
"""
Movie Search System - Find highest rated movies by various criteria
"""

import pandas as pd
import requests
import time
import os
from typing import List, Dict, Optional
import argparse

class MovieSearcher:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OMDB_API_KEY')
        self.base_url = "http://www.omdbapi.com/"
        
    def search_movies_by_person(self, person_name: str, role: str = "actor", max_results: int = 50) -> List[Dict]:
        """
        Search for movies by a specific person (actor, actress, director)
        """
        movies = []
        page = 1
        
        while len(movies) < max_results:
            params = {
                'apikey': self.api_key,
                's': person_name,
                'type': 'movie',
                'page': page
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if data.get('Response') == 'True':
                    for movie_data in data['Search']:
                        # Get detailed movie info
                        detailed_movie = self.get_movie_details(movie_data['imdbID'])
                        if detailed_movie and self._person_in_movie(detailed_movie, person_name, role):
                            movies.append(detailed_movie)
                            
                    page += 1
                    time.sleep(0.1)  # Rate limiting
                else:
                    break
                    
            except Exception as e:
                print(f"Error searching for {person_name}: {e}")
                break
        
        return movies[:max_results]
    
    def search_movies_by_genre(self, genre: str, max_results: int = 50) -> List[Dict]:
        """
        Search for movies by genre
        """
        params = {
            'apikey': self.api_key,
            's': '',  # Empty search to get popular movies
            'type': 'movie',
            'page': 1
        }
        
        movies = []
        page = 1
        
        while len(movies) < max_results:
            try:
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if data.get('Response') == 'True':
                    for movie_data in data['Search']:
                        detailed_movie = self.get_movie_details(movie_data['imdbID'])
                        if detailed_movie and genre.lower() in detailed_movie.get('Genre', '').lower():
                            movies.append(detailed_movie)
                    
                    page += 1
                    params['page'] = page
                    time.sleep(0.1)
                else:
                    break
                    
            except Exception as e:
                print(f"Error searching for genre {genre}: {e}")
                break
        
        return movies[:max_results]
    
    def get_movie_details(self, imdb_id: str) -> Optional[Dict]:
        """
        Get detailed movie information by IMDb ID
        """
        params = {
            'apikey': self.api_key,
            'i': imdb_id,
            'plot': 'short'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if data.get('Response') == 'True':
                return {
                    'Title': data.get('Title', ''),
                    'Year': data.get('Year', ''),
                    'Rated': data.get('Rated', ''),
                    'Released': data.get('Released', ''),
                    'Runtime': data.get('Runtime', ''),
                    'Genre': data.get('Genre', ''),
                    'Director': data.get('Director', ''),
                    'Actors': data.get('Actors', ''),
                    'Plot': data.get('Plot', ''),
                    'Language': data.get('Language', ''),
                    'Country': data.get('Country', ''),
                    'Awards': data.get('Awards', ''),
                    'Poster': data.get('Poster', ''),
                    'imdbRating': data.get('imdbRating', ''),
                    'imdbVotes': data.get('imdbVotes', ''),
                    'imdbID': data.get('imdbID', ''),
                    'Type': data.get('Type', ''),
                    'BoxOffice': data.get('BoxOffice', '')
                }
        except Exception as e:
            print(f"Error getting details for {imdb_id}: {e}")
        
        return None
    
    def _person_in_movie(self, movie: Dict, person_name: str, role: str) -> bool:
        """
        Check if a person is involved in a movie in the specified role
        """
        if role.lower() in ['actor', 'actress']:
            return person_name.lower() in movie.get('Actors', '').lower()
        elif role.lower() == 'director':
            return person_name.lower() in movie.get('Director', '').lower()
        return False
    
    def filter_and_sort_movies(self, movies: List[Dict], min_rating: float = 0.0, min_votes: int = 0) -> List[Dict]:
        """
        Filter movies by rating and votes, then sort by rating
        """
        filtered_movies = []
        
        for movie in movies:
            try:
                rating = float(movie.get('imdbRating', 0)) if movie.get('imdbRating', 'N/A') != 'N/A' else 0
                votes_str = movie.get('imdbVotes', '0').replace(',', '')
                votes = int(votes_str) if votes_str.isdigit() else 0
                
                if rating >= min_rating and votes >= min_votes:
                    filtered_movies.append(movie)
            except (ValueError, TypeError):
                continue
        
        # Sort by IMDb rating (descending)
        return sorted(filtered_movies, key=lambda x: float(x.get('imdbRating', 0)) if x.get('imdbRating', 'N/A') != 'N/A' else 0, reverse=True)


def load_actress_list(file_path: str) -> List[str]:
    """Load actress names from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df['Name'].tolist()
    except Exception as e:
        print(f"Error loading actress list: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Find highest rated movies by various criteria')
    parser.add_argument('--actresses', type=str, help='CSV file with actress list')
    parser.add_argument('--actor', type=str, help='Single actor/actress name')
    parser.add_argument('--director', type=str, help='Director name')
    parser.add_argument('--genre', type=str, help='Movie genre')
    parser.add_argument('--min-rating', type=float, default=7.0, help='Minimum IMDb rating (default: 7.0)')
    parser.add_argument('--min-votes', type=int, default=1000, help='Minimum number of votes (default: 1000)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top movies to show (default: 10)')
    parser.add_argument('--output', type=str, help='Output CSV file')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('OMDB_API_KEY')
    if not api_key:
        print("❌ Error: OMDB_API_KEY environment variable not set")
        print("💡 Get a free API key from: http://www.omdbapi.com/apikey.aspx")
        return
    
    searcher = MovieSearcher(api_key)
    all_movies = []
    
    print("🎬 Movie Search System")
    print("=" * 60)
    
    # Search based on criteria
    if args.actresses:
        print(f"👩 Searching for movies featuring actresses from: {args.actresses}")
        actresses = load_actress_list(args.actresses)
        for actress in actresses:
            print(f"   Searching for: {actress}")
            movies = searcher.search_movies_by_person(actress, role="actress")
            all_movies.extend(movies)
            time.sleep(0.5)  # Rate limiting
    
    if args.actor:
        print(f"🎭 Searching for movies featuring: {args.actor}")
        movies = searcher.search_movies_by_person(args.actor, role="actor")
        all_movies.extend(movies)
    
    if args.director:
        print(f"🎬 Searching for movies directed by: {args.director}")
        movies = searcher.search_movies_by_person(args.director, role="director")
        all_movies.extend(movies)
    
    if args.genre:
        print(f"🎪 Searching for {args.genre} movies")
        movies = searcher.search_movies_by_genre(args.genre)
        all_movies.extend(movies)
    
    if not any([args.actresses, args.actor, args.director, args.genre]):
        print("❌ Please specify at least one search criteria")
        parser.print_help()
        return
    
    # Remove duplicates
    unique_movies = {}
    for movie in all_movies:
        imdb_id = movie.get('imdbID')
        if imdb_id and imdb_id not in unique_movies:
            unique_movies[imdb_id] = movie
    
    unique_movies_list = list(unique_movies.values())
    
    print(f"\n📊 Found {len(unique_movies_list)} unique movies")
    
    # Filter and sort
    filtered_movies = searcher.filter_and_sort_movies(
        unique_movies_list, 
        min_rating=args.min_rating, 
        min_votes=args.min_votes
    )
    
    top_movies = filtered_movies[:args.top_n]
    
    print(f"🎯 Showing top {len(top_movies)} movies (Rating ≥ {args.min_rating}, Votes ≥ {args.min_votes})")
    print("=" * 80)
    
    for idx, movie in enumerate(top_movies, 1):
        print(f"\n{idx}. {movie['Title']} ({movie.get('Year', 'N/A')})")
        print(f"   ⭐ IMDb Rating: {movie.get('imdbRating', 'N/A')} ({movie.get('imdbVotes', '0')} votes)")
        print(f"   🎭 Genres: {movie.get('Genre', 'N/A')}")
        print(f"   👨‍💼 Director: {movie.get('Director', 'N/A')}")
        print(f"   🎬 Actors: {movie.get('Actors', 'N/A')[:80]}...")
        print(f"   ⏱️  Runtime: {movie.get('Runtime', 'N/A')}")
        print(f"   🏆 Awards: {movie.get('Awards', 'N/A')}")
        print("-" * 80)
    
    # Save to CSV if requested
    if args.output and top_movies:
        df = pd.DataFrame(top_movies)
        df.to_csv(args.output, index=False)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()