import pandas as pd
import numpy as np
import requests
import time
from typing import List, Dict, Optional
from config.config import Config

class RecommendationGenerator:
    def __init__(self, omdb_client):
        self.omdb_client = omdb_client
        self.popular_movies_cache = None
        
    def analyze_user_preferences(self, ratings_df: pd.DataFrame) -> Dict:
        """Analyze user's rating patterns to understand preferences"""
        
        high_rated = ratings_df[ratings_df['Your_Rating'] >= 8]
        low_rated = ratings_df[ratings_df['Your_Rating'] <= 5]
        
        preferences = {
            # Genre preferences
            'preferred_genres': self._extract_top_genres(high_rated),
            'disliked_genres': self._extract_top_genres(low_rated),
            
            # Rating patterns
            'avg_rating': ratings_df['Your_Rating'].mean(),
            'preferred_imdb_range': self._calculate_preferred_imdb_range(high_rated),
            'preferred_runtime_range': self._calculate_preferred_runtime_range(high_rated),
            
            # Feature preferences
            'preferred_features': self._extract_feature_preferences(high_rated),
            'disliked_features': self._extract_feature_preferences(low_rated),
            
            # Director preferences
            'preferred_directors': self._extract_top_directors(high_rated),
            
            # Year preferences
            'preferred_years': self._extract_year_preferences(high_rated)
        }
        
        return preferences
    
    def _extract_top_genres(self, df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Extract most common genres from dataframe"""
        if len(df) == 0:
            return []
            
        all_genres = []
        for genres in df['Genres'].dropna():
            all_genres.extend([genre.strip() for genre in genres.split(',')])
        
        genre_counts = pd.Series(all_genres).value_counts()
        return genre_counts.head(top_n).index.tolist()
    
    def _calculate_preferred_imdb_range(self, high_rated: pd.DataFrame) -> tuple:
        """Calculate preferred IMDb rating range"""
        if len(high_rated) == 0:
            return (6.0, 9.0)
        
        imdb_ratings = high_rated['IMDb_Rating'].dropna()
        if len(imdb_ratings) == 0:
            return (6.0, 9.0)
            
        lower_bound = max(5.0, imdb_ratings.quantile(0.25) - 0.5)
        upper_bound = min(10.0, imdb_ratings.quantile(0.75) + 0.5)
        return (lower_bound, upper_bound)
    
    def _calculate_preferred_runtime_range(self, high_rated: pd.DataFrame) -> tuple:
        """Calculate preferred runtime range"""
        if len(high_rated) == 0:
            return (80, 180)
        
        runtimes = high_rated['Runtime_mins'].dropna()
        if len(runtimes) == 0:
            return (80, 180)
            
        lower_bound = max(60, runtimes.quantile(0.25) - 20)
        upper_bound = min(240, runtimes.quantile(0.75) + 20)
        return (lower_bound, upper_bound)
    
    def _extract_feature_preferences(self, df: pd.DataFrame) -> List[str]:
        """Extract which features user prefers"""
        features = ['Has_LGBT', 'Has_Cinematography', 'Has_Screenplay', 'Has_Plot_Twist']
        preferred = []
        
        for feature in features:
            if feature in df.columns and df[feature].mean() > 0.3:
                preferred.append(feature)
        
        return preferred
    
    def _extract_top_directors(self, high_rated: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Extract most frequently high-rated directors"""
        if len(high_rated) == 0 or 'Directors' not in high_rated.columns:
            return []
            
        all_directors = []
        for directors in high_rated['Directors'].dropna():
            # Handle multiple directors separated by commas
            director_list = [d.strip() for d in str(directors).split(',')]
            all_directors.extend(director_list)
        
        director_counts = pd.Series(all_directors).value_counts()
        return director_counts.head(top_n).index.tolist()
    
    def _extract_year_preferences(self, high_rated: pd.DataFrame) -> tuple:
        """Extract preferred release year range"""
        if len(high_rated) == 0:
            return (1950, 2023)
        
        years = high_rated['Year'].dropna()
        if len(years) == 0:
            return (1950, 2023)
            
        lower_bound = max(1900, years.quantile(0.25) - 10)
        upper_bound = min(2025, years.quantile(0.75) + 10)
        return (lower_bound, upper_bound)
    
    def get_popular_movies_by_criteria(self, preferences: Dict, count: int = 50) -> List[Dict]:
        """Get popular movies that match user preferences"""
        
        # This is a simplified version - in practice, you might use:
        # 1. TMDB API for popular movies
        # 2. IMDb datasets
        # 3. Pre-compiled movie databases
        
        # For now, we'll use a combination of OMDb searches and known popular movies
        popular_movies = self._get_known_popular_movies()
        
        # Filter by preferences
        filtered_movies = []
        
        for movie in popular_movies:
            score = self._calculate_match_score(movie, preferences)
            if score > 0.3:  # Minimum match threshold
                movie['match_score'] = score
                filtered_movies.append(movie)
        
        # Sort by match score and return top ones
        filtered_movies.sort(key=lambda x: x['match_score'], reverse=True)
        return filtered_movies[:count]
    
    def _get_known_popular_movies(self) -> List[Dict]:
        """Get a list of generally popular/acclaimed movies"""
        # This is a curated list - you could expand this significantly
        popular_titles = [
            # Award winners and critically acclaimed
            "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction",
            "Schindler's List", "The Lord of the Rings: The Return of the King", "Forrest Gump",
            "Inception", "The Matrix", "Goodfellas", "Star Wars", "The Silence of the Lambs",
            
            # Modern popular
            "Parasite", "Joker", "Get Out", "La La Land", "Mad Max: Fury Road", "The Social Network",
            "Django Unchained", "The Grand Budapest Hotel", "Black Swan", "The Departed",
            
            # International
            "Spirited Away", "Amélie", "City of God", "Pan's Labyrinth", "Oldboy", "A Separation",
            
            # LGBT themes
            "Moonlight", "Brokeback Mountain", "Carol", "Call Me By Your Name", "The Favourite",
            "Portrait of a Lady on Fire", "Paris Is Burning",
            
            # Strong cinematography
            "Blade Runner 2049", "The Revenant", "1917", "Roma", "Birdman", "Gravity",
            
            # Great screenplays
            "Eternal Sunshine of the Spotless Mind", "Her", "Before Sunrise", "The Truman Show",
            "Fargo", "American Beauty"
        ]
        
        movies = []
        for title in popular_titles:
            movies.append({
                'Title': title,
                'Year': None,  # Will be filled by OMDb
                'Genres': '',  # Will be filled by OMDb
                'Director': '',  # Will be filled by OMDb
                'IMDb_Rating': 0,  # Will be filled by OMDb
                'Runtime_mins': 0  # Will be filled by OMDb
            })
        
        return movies
    
    def _calculate_match_score(self, movie: Dict, preferences: Dict) -> float:
        """Calculate how well a movie matches user preferences"""
        score = 0.0
        max_score = 0.0
        
        # Genre match (40% weight)
        if 'Genres' in movie and preferences['preferred_genres']:
            movie_genres = [genre.strip().lower() for genre in movie['Genres'].split(',')]
            preferred_genres = [genre.lower() for genre in preferences['preferred_genres']]
            
            genre_match = len(set(movie_genres) & set(preferred_genres)) / len(preferred_genres)
            score += genre_match * 0.4
            max_score += 0.4
        
        # Director match (20% weight)
        if 'Director' in movie and preferences['preferred_directors']:
            movie_directors = [d.strip().lower() for d in movie['Director'].split(',')]
            preferred_directors = [d.lower() for d in preferences['preferred_directors']]
            
            director_match = len(set(movie_directors) & set(preferred_directors)) / len(preferred_directors)
            score += director_match * 0.2
            max_score += 0.2
        
        # Year match (10% weight)
        if 'Year' in movie and movie['Year']:
            year_low, year_high = preferences['preferred_years']
            if year_low <= movie['Year'] <= year_high:
                score += 0.1
            max_score += 0.1
        
        # IMDb rating match (20% weight)
        if 'IMDb_Rating' in movie and movie['IMDb_Rating']:
            imdb_low, imdb_high = preferences['preferred_imdb_range']
            if imdb_low <= movie['IMDb_Rating'] <= imdb_high:
                score += 0.2
            max_score += 0.2
        
        # Runtime match (10% weight)
        if 'Runtime_mins' in movie and movie['Runtime_mins']:
            runtime_low, runtime_high = preferences['preferred_runtime_range']
            if runtime_low <= movie['Runtime_mins'] <= runtime_high:
                score += 0.1
            max_score += 0.1
        
        return score / max_score if max_score > 0 else 0
    
    def generate_recommendation_list(self, ratings_df: pd.DataFrame, 
                                   num_recommendations: int = 50) -> pd.DataFrame:
        """Generate a list of movie recommendations based on user preferences"""
        
        print("🔍 Analyzing your movie preferences...")
        preferences = self.analyze_user_preferences(ratings_df)
        
        print("🎯 Finding movies that match your taste...")
        potential_movies = self.get_popular_movies_by_criteria(preferences, num_recommendations)
        
        print("📡 Fetching movie details from OMDb...")
        detailed_movies = []
        
        for movie in potential_movies:
            # Fetch detailed information from OMDb
            movie_data = self.omdb_client.search_movie(movie['Title'])
            if movie_data:
                movie_data['Match_Score'] = movie['match_score']
                detailed_movies.append(movie_data)
            time.sleep(0.2)  # Be nice to the API
        
        # Convert to DataFrame
        if detailed_movies:
            recommendations_df = pd.DataFrame(detailed_movies)
            
            # Add some analysis columns
            recommendations_df['Why_Recommended'] = recommendations_df.apply(
                lambda row: self._generate_recommendation_reason(row, preferences), 
                axis=1
            )
            
            return recommendations_df.sort_values('Match_Score', ascending=False)
        else:
            return pd.DataFrame()
    
    def _generate_recommendation_reason(self, movie: pd.Series, preferences: Dict) -> str:
        """Generate a human-readable reason why the movie is recommended"""
        reasons = []
        
        # Check genre match
        if 'Genres' in movie and preferences['preferred_genres']:
            movie_genres = [genre.strip().lower() for genre in str(movie['Genres']).split(',')]
            matched_genres = set(movie_genres) & set([g.lower() for g in preferences['preferred_genres']])
            if matched_genres:
                reasons.append(f"genres: {', '.join(matched_genres)}")
        
        # Check director match
        if 'Director' in movie and preferences['preferred_directors']:
            movie_directors = [d.strip().lower() for d in str(movie['Director']).split(',')]
            matched_directors = set(movie_directors) & set([d.lower() for d in preferences['preferred_directors']])
            if matched_directors:
                reasons.append(f"director: {', '.join(matched_directors)}")
        
        # Check features
        if preferences['preferred_features']:
            feature_reasons = []
            if 'Has_LGBT' in preferences['preferred_features'] and 'lgbt' in str(movie.get('Plot', '')).lower():
                feature_reasons.append('LGBT themes')
            if 'Has_Cinematography' in preferences['preferred_features'] and movie.get('IMDb_Rating', 0) > 7.5:
                feature_reasons.append('high visual quality')
            if feature_reasons:
                reasons.append(f"features: {', '.join(feature_reasons)}")
        
        return f"Matches your preference for {', '.join(reasons)}" if reasons else "Popular and highly rated"
    
    def save_recommendations_for_rating(self, recommendations_df: pd.DataFrame, 
                                      filename: str = "data/auto_generated_movies_to_rate.csv"):
        """Save recommendations in the format needed for movies_to_rate.csv"""
        
        if recommendations_df.empty:
            print("❌ No recommendations to save")
            return
        
        # Create the output DataFrame in the required format
        output_data = []
        
        for _, movie in recommendations_df.iterrows():
            output_row = {
                'Film (Year)': f"{movie['Title']} ({int(movie['Year']) if pd.notna(movie['Year']) else 'Unknown'})",
                'Director': movie.get('Director', 'Unknown'),
                'Country / Language': f"{movie.get('Country', 'Unknown')} / {movie.get('Language', 'Unknown')}",
                'M/F Narrative': 'Unknown',  # Would need additional data source
                'Main Awards & Recognition': 'To be discovered',
                'Keywords / Tags': movie.get('Genres', '') + ', ' + self._extract_keywords(movie),
                'Notable Critiques / Context': f"Match score: {movie.get('Match_Score', 0):.2f}. {movie.get('Why_Recommended', 'Recommended based on your preferences')}"
            }
            output_data.append(output_row)
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(filename, index=False)
        print(f"💾 Auto-generated recommendations saved to: {filename}")
        print(f"📝 Generated {len(output_df)} movie recommendations for you to rate")
        
        return output_df
    
    def _extract_keywords(self, movie: pd.Series) -> str:
        """Extract keywords from movie data"""
        keywords = []
        
        if movie.get('IMDb_Rating', 0) > 8.0:
            keywords.append('Highly Acclaimed')
        elif movie.get('IMDb_Rating', 0) > 7.0:
            keywords.append('Well Rated')
        
        if movie.get('Runtime_mins', 0) > 150:
            keywords.append('Epic')
        elif movie.get('Runtime_mins', 0) < 90:
            keywords.append('Short')
        
        # Add genre-based keywords
        genres = str(movie.get('Genres', '')).split(',')
        for genre in genres:
            genre = genre.strip()
            if genre in ['Drama', 'Comedy', 'Action', 'Thriller']:
                keywords.append(genre)
        
        return ', '.join(keywords)