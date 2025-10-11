import requests
import pandas as pd
import time
from typing import Dict, List, Optional, Any
from config.config import Config

class OMDbClient:
    def __init__(self):
        self.api_key = Config.OMDB_API_KEY
        self.base_url = Config.OMDB_BASE_URL
    
    def search_movie(self, title: str, year: Optional[int] = None, fallback_data: Dict = None) -> Optional[Dict]:
        """Search for movie by title and year with language support"""
        if fallback_data is None:
            fallback_data = {}
        
        # Extract language from fallback data if available
        language = fallback_data.get('Language_Code', '')
        
        # Try multiple search strategies
        strategies = [
            self._search_by_title_year,
            self._search_by_title, 
            self._search_with_fallback
        ]
        
        for strategy in strategies:
            try:
                if strategy == self._search_with_fallback:
                    movie_data = strategy(title, year, fallback_data)
                else:
                    movie_data = strategy(title, year)
                    
                if movie_data and movie_data.get('Response') != 'False':
                    # Add language information to the result
                    if language:
                        movie_data['Language_From_CSV'] = language
                        movie_data['Primary_Language'] = language
                    # Enhance with fallback data
                    movie_data = self._enhance_with_fallback_data(movie_data, fallback_data)
                    return movie_data
            except Exception as e:
                continue
        
        # If all strategies fail, use fallback data
        print(f"🔄 Using fallback data for: {title}")
        return self._create_fallback_movie_data(title, year, fallback_data)

    def _search_by_title_year(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Search by title and year"""
        params = {
            't': title,
            'apikey': self.api_key,
            'type': 'movie',
            'plot': 'full'
        }
        
        if year and year > 1900:
            params['y'] = int(year)
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'True':
                    return self._parse_enhanced_movie_data(data)
        except Exception as e:
            print(f"❌ Error in _search_by_title_year for {title}: {e}")
        
        return None

    def _search_by_title(self, title: str) -> Optional[Dict]:
        """Search by title only"""
        return self._search_by_title_year(title, None)

    def _search_with_fallback(self, title: str, year: Optional[int], fallback_data: Dict) -> Optional[Dict]:
        """Search with fallback data enhancement"""
        # Try the standard search first
        movie_data = self._search_by_title_year(title, year)
        
        if movie_data:
            # Enhance with fallback data
            movie_data = self._enhance_with_fallback_data(movie_data, fallback_data)
            return movie_data
        
        return None
    
    def _enhance_with_fallback_data(self, movie_data: Dict, fallback_data: Dict) -> Dict:
        """Enhance OMDb data with fallback data from CSV"""
        # Preserve language information
        if 'Language_Code' in fallback_data:
            movie_data['Language_From_CSV'] = fallback_data['Language_Code']
            movie_data['Primary_Language'] = fallback_data['Language_Code']
        
        # Use CSV data to fill gaps in OMDb data
        if not movie_data.get('Director') and fallback_data.get('Director'):
            movie_data['Director'] = fallback_data['Director']
        
        if not movie_data.get('Country') and fallback_data.get('Country_From_CSV'):
            movie_data['Country'] = fallback_data['Country_From_CSV']
        
        if not movie_data.get('Plot') and fallback_data.get('Description'):
            movie_data['Plot'] = fallback_data['Description']
        
        # Preserve other CSV data
        csv_fields = [
            'Keywords_From_CSV', 'Female_Critiques_From_CSV', 
            'Hours_Themes_Alignment_From_CSV', 'Awards_From_CSV',
            'Narrative_Type_From_CSV', 'Has_Oscar_From_CSV'
        ]
        
        for field in csv_fields:
            if field in fallback_data:
                movie_data[field] = fallback_data[field]
        
        return movie_data
    
    def _create_fallback_movie_data(self, title: str, year: Optional[int], fallback_data: Dict) -> Dict:
        """Create basic movie data using fallback information"""
        fallback_movie = {
            'Title': title,
            'Year': year if year else 0,
            'Rated': 'N/A',
            'Released': fallback_data.get('Release_Date', ''),
            'Runtime': f"{fallback_data.get('Runtime_mins_From_CSV', 0)} min",
            'Runtime_mins': fallback_data.get('Runtime_mins_From_CSV', 0),
            'Genre': fallback_data.get('Genres_From_CSV', ''),
            'Director': fallback_data.get('Director', ''),
            'Writer': '',
            'Actors': '',
            'Plot': fallback_data.get('Description', ''),
            'Language': fallback_data.get('Language_Code', ''),  # Add language
            'Country': fallback_data.get('Country_From_CSV', ''),
            'imdbID': fallback_data.get('Const', ''),
            'IMDb_Rating': fallback_data.get('IMDb_Rating_From_CSV', 0),
            'imdbRating': fallback_data.get('IMDb_Rating_From_CSV', 0),
            'imdbVotes': fallback_data.get('Num_Votes_From_CSV', 0),
            'Type': 'movie',
            # Add language fields
            'Language_From_CSV': fallback_data.get('Language_Code', ''),
            'Primary_Language': fallback_data.get('Language_Code', ''),
        }
        
        # Add other fallback CSV data
        for field in ['Keywords_From_CSV', 'Awards_From_CSV', 'Has_Oscar_From_CSV']:
            if field in fallback_data:
                fallback_movie[field] = fallback_data[field]
        
        return fallback_movie
    
    def _parse_enhanced_movie_data(self, data: Dict) -> Dict:
        """Parse OMDb response into comprehensive standardized format"""
        
        # Basic Info
        movie_data = {
            # Core Identification
            'Title': data.get('Title', ''),
            'Year': self._safe_int(data.get('Year', '0').split('–')[0]),
            'Rated': data.get('Rated', 'Not Rated'),
            'Released': data.get('Released', ''),
            'Runtime': data.get('Runtime', ''),
            'Runtime_mins': self._extract_runtime_minutes(data.get('Runtime', '')),
            
            # Content Classification
            'Content_Warnings': data.get('Rated', 'Not Rated'),
            'Genre': data.get('Genre', ''),
            'Primary_Genre': self._extract_primary_genre(data.get('Genre', '')),
            'Secondary_Genre': self._extract_secondary_genre(data.get('Genre', '')),
            
            # Creative Team
            'Director': data.get('Director', ''),
            'Writer': data.get('Writer', ''),
            'Actors': data.get('Actors', ''),
            
            # Plot & Context
            'Plot': data.get('Plot', ''),
            'Language': data.get('Language', ''),
            'Country': data.get('Country', ''),
            
            # Awards & Recognition
            'Awards': data.get('Awards', ''),
            'Has_Oscar': self._has_oscar(data.get('Awards', '')),
            'Award_Count': self._count_awards(data.get('Awards', '')),
            
            # Ratings - FIX: Use both IMDb_Rating (for ML) and imdbRating (for display)
            'imdbID': data.get('imdbID', ''),
            'IMDb_Rating': self._safe_float(data.get('imdbRating', 0)),  # FIX: Map imdbRating to IMDb_Rating
            'imdbRating': self._safe_float(data.get('imdbRating', 0)),   # Keep original for display
            'imdbVotes': self._safe_int(data.get('imdbVotes', '0').replace(',', '')),
            'Metascore': self._safe_int(data.get('Metascore', 0)),
            
            # Additional Ratings
            'Ratings': str(data.get('Ratings', [])),
            
            # Technical Details
            'Type': data.get('Type', ''),
            'DVD': data.get('DVD', ''),
            'BoxOffice': data.get('BoxOffice', ''),
            'Production': data.get('Production', ''),
            'Website': data.get('Website', ''),
            
            # Derived Technical Features
            'Is_Adaptation': self._is_adaptation(data.get('Plot', '') + data.get('Title', '')),
            'Budget_Level': self._estimate_budget_level(data.get('BoxOffice', ''), data.get('Production', '')),
            
            # Content Analysis
            'Mood_Tags': self._analyze_mood(data.get('Plot', ''), data.get('Genre', '')),
            'Pacing': self._estimate_pacing(data.get('Runtime', ''), data.get('Genre', '')),
            'Style_Tags': self._analyze_style(data.get('Genre', ''), data.get('Director', '')),
            'Themes': self._extract_themes(data.get('Plot', ''), data.get('Genre', '')),
            
            # Character & Narrative Analysis
            'Lead_Gender': self._analyze_lead_gender(data.get('Actors', ''), data.get('Plot', '')),
            'Narrative_Structure': self._analyze_narrative_structure(data.get('Plot', '')),
            'Ending_Type': self._analyze_ending_type(data.get('Plot', '')),
            'Character_Arcs': self._analyze_character_arcs(data.get('Plot', '')),
            
            # Technical & Production
            'Special_Effects': self._estimate_special_effects(data.get('Genre', ''), data.get('Year', '')),
            'Filming_Location_Diversity': self._estimate_location_diversity(data.get('Country', '')),
        }
        
        # Add genre flags for ML features
        movie_data.update(self._create_genre_flags(data.get('Genre', '')))
        
        return movie_data
    
    # Helper methods for data extraction and analysis
    def _safe_int(self, value: Any) -> int:
        """Safely convert to integer"""
        try:
            if isinstance(value, str):
                # Extract numbers from strings like "1995" or "1995–2000"
                numbers = ''.join(filter(str.isdigit, value.split('–')[0]))
                return int(numbers) if numbers else 0
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_runtime_minutes(self, runtime: str) -> int:
        """Extract runtime in minutes from string like '142 min'"""
        try:
            if 'min' in runtime:
                return int(runtime.split(' ')[0])
            return 0
        except (ValueError, IndexError):
            return 0
    
    def _extract_primary_genre(self, genres: str) -> str:
        """Extract primary genre from comma-separated list"""
        if not genres or genres == 'N/A':
            return 'Unknown'
        return genres.split(',')[0].strip()
    
    def _extract_secondary_genre(self, genres: str) -> str:
        """Extract secondary genre from comma-separated list"""
        if not genres or genres == 'N/A':
            return 'Unknown'
        genre_list = [g.strip() for g in genres.split(',')]
        return genre_list[1] if len(genre_list) > 1 else 'Unknown'
    
    def _has_oscar(self, awards: str) -> int:
        """Check if movie has Oscar wins/nominations"""
        awards_lower = awards.lower()
        return 1 if 'oscar' in awards_lower and ('won' in awards_lower or 'nominated' in awards_lower) else 0
    
    def _count_awards(self, awards: str) -> int:
        """Count number of awards/nominations"""
        if not awards or awards == 'N/A':
            return 0
        # Simple count - you could make this more sophisticated
        return awards.count('win') + awards.count('nomination')
    
    def _is_adaptation(self, text: str) -> str:
        """Determine if movie is an adaptation"""
        text_lower = text.lower()
        adaptation_keywords = [
            'based on', 'adapted from', 'novel', 'book', 'true story', 
            'biography', 'autobiography', 'remake', 'comic', 'graphic novel'
        ]
        for keyword in adaptation_keywords:
            if keyword in text_lower:
                return keyword.replace(' ', '_')
        return 'original'
    
    def _estimate_budget_level(self, box_office: str, production: str) -> str:
        """Estimate budget level based on box office and production company"""
        if box_office != 'N/A' and box_office:
            try:
                # Convert box office to number (e.g., "$150,000,000" -> 150000000)
                revenue = float(''.join(filter(str.isdigit, box_office)))
                if revenue > 200000000:
                    return 'blockbuster'
                elif revenue > 50000000:
                    return 'high'
                elif revenue > 10000000:
                    return 'medium'
            except (ValueError, TypeError):
                pass
        
        # Fallback based on production company
        big_studios = ['warner', 'disney', 'universal', 'paramount', 'sony', 'fox', 'mgm']
        if any(studio in production.lower() for studio in big_studios):
            return 'medium'
        
        return 'low'
    
    def _analyze_mood(self, plot: str, genres: str) -> str:
        """Analyze mood from plot and genres"""
        plot_lower = plot.lower()
        genres_lower = genres.lower()
        
        moods = []
        
        # Genre-based moods
        if any(g in genres_lower for g in ['comedy', 'rom-com']):
            moods.append('funny')
        if any(g in genres_lower for g in ['drama', 'romance']):
            moods.append('emotional')
        if any(g in genres_lower for g in ['thriller', 'horror', 'mystery']):
            moods.append('suspenseful')
        if any(g in genres_lower for g in ['action', 'adventure']):
            moods.append('exciting')
        if any(g in genres_lower for g in ['film-noir', 'crime']):
            moods.append('dark')
        
        # Plot-based moods
        plot_keywords = {
            'uplifting': ['hope', 'triumph', 'joy', 'success', 'love conquers'],
            'dark': ['death', 'murder', 'tragedy', 'betrayal', 'revenge'],
            'thoughtful': ['philosophy', 'meaning', 'existential', 'reflection'],
            'inspiring': ['overcome', 'achievement', 'courage', 'bravery']
        }
        
        for mood, keywords in plot_keywords.items():
            if any(keyword in plot_lower for keyword in keywords):
                moods.append(mood)
        
        return ', '.join(set(moods)) if moods else 'neutral'
    
    def _estimate_pacing(self, runtime: str, genres: str) -> str:
        """Estimate pacing based on runtime and genres"""
        runtime_mins = self._extract_runtime_minutes(runtime)
        genres_lower = genres.lower()
        
        if runtime_mins > 150:
            return 'epic'
        elif runtime_mins > 120:
            return 'slow'
        elif runtime_mins < 90:
            return 'fast'
        
        # Genre-based pacing
        if any(g in genres_lower for g in ['action', 'thriller', 'comedy']):
            return 'fast'
        elif any(g in genres_lower for g in ['drama', 'biography', 'history']):
            return 'medium'
        
        return 'medium'
    
    def _analyze_style(self, genres: str, director: str) -> str:
        """Analyze style based on genres and director"""
        styles = []
        genres_lower = genres.lower()
        director_lower = director.lower()
        
        # Genre-based styles
        if 'film-noir' in genres_lower:
            styles.append('noir')
        if 'epic' in genres_lower or 'war' in genres_lower:
            styles.append('epic')
        if 'fantasy' in genres_lower or 'sci-fi' in genres_lower:
            styles.append('surreal')
        if 'romance' in genres_lower or 'drama' in genres_lower:
            styles.append('intimate')
        
        # Director-based styles (simplified)
        auteur_directors = {
            'wes anderson': 'quirky',
            'david lynch': 'surreal', 
            'quentin tarantino': 'stylized',
            'christopher nolan': 'complex',
            'martin scorsese': 'gritty'
        }
        
        for director_pattern, style in auteur_directors.items():
            if director_pattern in director_lower:
                styles.append(style)
        
        return ', '.join(set(styles)) if styles else 'conventional'
    
    def _extract_themes(self, plot: str, genres: str) -> str:
        """Extract themes from plot and genres"""
        plot_lower = plot.lower()
        themes = []
        
        theme_keywords = {
            'redemption': ['redeem', 'second chance', 'forgiveness', 'atonement'],
            'betrayal': ['betray', 'traitor', 'backstab', 'double cross'],
            'coming-of-age': ['grow up', 'teenager', 'adolescent', 'maturity'],
            'justice': ['justice', 'court', 'law', 'righteous'],
            'family': ['family', 'parent', 'child', 'sibling'],
            'friendship': ['friend', 'buddy', 'companion', 'loyalty'],
            'love': ['love', 'romance', 'relationship', 'heart'],
            'revenge': ['revenge', 'vengeance', 'retaliation'],
            'survival': ['survive', 'wilderness', 'stranded', 'endure']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in plot_lower for keyword in keywords):
                themes.append(theme)
        
        return ', '.join(set(themes)) if themes else 'general'
    
    def _analyze_lead_gender(self, actors: str, plot: str) -> str:
        """Analyze lead gender from actors and plot"""
        # Simple analysis based on actor list order
        if actors and actors != 'N/A':
            first_actor = actors.split(',')[0].lower()
            # Very basic gender assumption based on common names
            # In production, you'd want a more sophisticated approach
            male_indicators = [' he ', ' his ', ' man ', ' male ']
            female_indicators = [' she ', ' her ', ' woman ', ' female ']
            
            plot_lower = plot.lower()
            male_count = sum(1 for indicator in male_indicators if indicator in plot_lower)
            female_count = sum(1 for indicator in female_indicators if indicator in plot_lower)
            
            if male_count > female_count:
                return 'male'
            elif female_count > male_count:
                return 'female'
            elif male_count == female_count and male_count > 0:
                return 'ensemble'
        
        return 'unknown'
    
    def _analyze_narrative_structure(self, plot: str) -> str:
        """Analyze narrative structure from plot"""
        plot_lower = plot.lower()
        
        if any(word in plot_lower for word in ['flashback', 'memory', 'remembering']):
            return 'non-linear'
        if any(word in plot_lower for word in ['dream', 'fantasy', 'imagination']):
            return 'surreal'
        
        return 'linear'
    
    def _analyze_ending_type(self, plot: str) -> str:
        """Analyze ending type from plot keywords"""
        plot_lower = plot.lower()
        
        if any(word in plot_lower for word in ['happy ending', 'triumph', 'success']):
            return 'happy'
        if any(word in plot_lower for word in ['tragedy', 'death', 'loss']):
            return 'tragic'
        if any(word in plot_lower for word in ['ambiguous', 'uncertain', 'open-ended']):
            return 'ambiguous'
        if any(word in plot_lower for word in ['bittersweet', 'mixed', 'complicated']):
            return 'bittersweet'
        
        return 'neutral'
    
    def _analyze_character_arcs(self, plot: str) -> str:
        """Analyze character arcs from plot"""
        plot_lower = plot.lower()
        arcs = []
        
        if any(word in plot_lower for word in ['redeem', 'change', 'become better']):
            arcs.append('redemption')
        if any(word in plot_lower for word in ['grow', 'mature', 'learn']):
            arcs.append('growth')
        if any(word in plot_lower for word in ['downfall', 'tragic', 'destruction']):
            arcs.append('tragedy')
        if any(word in plot_lower for word in ['discovery', 'find themselves', 'identity']):
            arcs.append('self-discovery')
        
        return ', '.join(arcs) if arcs else 'minimal'
    
    def _estimate_special_effects(self, genres: str, year: Any) -> str:
        """Estimate special effects level"""
        genres_lower = genres.lower()
        year_num = self._safe_int(year)
        
        if any(g in genres_lower for g in ['sci-fi', 'fantasy', 'action']):
            if year_num > 2000:
                return 'CGI_heavy'
            else:
                return 'practical_effects'
        
        return 'minimal'
    
    def _estimate_location_diversity(self, country: str) -> str:
        """Estimate filming location diversity"""
        if not country or country == 'N/A':
            return 'unknown'
        
        countries = [c.strip() for c in country.split(',')]
        if len(countries) > 2:
            return 'high'
        elif len(countries) > 1:
            return 'medium'
        else:
            return 'low'
    
    def _create_genre_flags(self, genres: str) -> Dict[str, int]:
        """Create genre flag columns for ML features"""
        genre_flags = {}
        all_genres = [
            'Drama', 'Comedy', 'Action', 'Adventure', 'Sci_Fi', 'Fantasy',
            'Horror', 'Thriller', 'Romance', 'Crime', 'Mystery', 'Animation',
            'Family', 'Biography', 'History', 'War', 'Musical', 'Sport',
            'Film_Noir', 'Western', 'Documentary'
        ]
        
        genres_lower = genres.lower()
        for genre in all_genres:
            genre_key = f'Is_{genre}'
            # Handle genre name variations
            genre_search = genre.lower().replace('_', '-')  # Search for "sci-fi" but store as "sci_fi"
            genre_flags[genre_key] = 1 if genre_search in genres_lower else 0
        
        return genre_flags
    
    def batch_search_movies(self, titles: List[str], delay: float = 0.2) -> pd.DataFrame:
        """Search multiple movies with delay to respect API limits"""
        movies_data = []
        
        for i, title in enumerate(titles):
            print(f"🌐 Fetching {i+1}/{len(titles)}: {title}")
            movie_data = self.search_movie(title)
            if movie_data:
                movies_data.append(movie_data)
            time.sleep(delay)  # Be nice to the API
        
        if movies_data:
            return pd.DataFrame(movies_data)
        else:
            return pd.DataFrame()