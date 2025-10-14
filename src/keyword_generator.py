import pandas as pd
import re
from typing import Dict, List, Any

class KeywordGenerator:
    def __init__(self):
        self.genre_keywords = {
            'Drama': ['emotional', 'character-driven', 'relationships', 'human condition'],
            'Comedy': ['funny', 'humorous', 'lighthearted', 'entertaining'],
            'Action': ['thrilling', 'adventure', 'stunts', 'combat', 'exciting'],
            'Romance': ['love', 'relationships', 'emotional', 'heartfelt'],
            'Thriller': ['suspenseful', 'tense', 'mystery', 'psychological'],
            'Crime': ['investigation', 'justice', 'criminal', 'law'],
            'Horror': ['scary', 'fear', 'supernatural', 'terror'],
            'Sci-Fi': ['futuristic', 'technology', 'space', 'speculative'],
            'Fantasy': ['magical', 'mythical', 'adventure', 'supernatural'],
            'Mystery': ['investigation', 'clues', 'suspense', 'puzzle'],
            'Animation': ['family', 'colorful', 'imaginative', 'creative'],
            'Documentary': ['real-life', 'educational', 'informative', 'factual'],
            'Biography': ['true story', 'historical figure', 'inspirational'],
            'History': ['period piece', 'historical', 'educational', 'era-specific'],
            'War': ['conflict', 'military', 'bravery', 'survival']
        }
        
        self.mood_keywords = {
            'uplifting': ['hope', 'triumph', 'inspiration', 'joy', 'success'],
            'dark': ['tragedy', 'death', 'betrayal', 'revenge', 'conflict'],
            'thoughtful': ['philosophical', 'reflective', 'meaning', 'existential'],
            'tense': ['suspense', 'anxiety', 'pressure', 'uncertainty'],
            'romantic': ['love', 'passion', 'heartfelt', 'emotional'],
            'funny': ['humor', 'comedy', 'wit', 'entertaining'],
            'epic': ['grand', 'sweeping', 'large-scale', 'journey']
        }

    def generate_keywords(self, movie_data: Dict[str, Any]) -> str:
        """Generate enhanced keywords from movie data with error handling"""
        try:
            keywords = set()
            
            # 1. Genre-based keywords
            genre_keywords = self._extract_genre_keywords(movie_data)
            keywords.update(genre_keywords)
            
            # 2. Plot analysis keywords
            plot_keywords = self._analyze_plot(movie_data.get('Plot', ''))
            keywords.update(plot_keywords)
            
            # 3. Mood and tone keywords
            mood_keywords = self._analyze_mood(movie_data)
            keywords.update(mood_keywords)
            
            # 4. Director/style keywords
            director_keywords = self._analyze_director_style(movie_data.get('Director', ''))
            keywords.update(director_keywords)
            
            # 5. Country/language keywords
            cultural_keywords = self._analyze_cultural_context(movie_data)
            keywords.update(cultural_keywords)
            
            # 6. Year/era keywords
            era_keywords = self._analyze_era(movie_data.get('Year', 0))
            keywords.update(era_keywords)
            
            # 7. Awards/recognition keywords
            award_keywords = self._analyze_awards(movie_data.get('Awards', ''))
            keywords.update(award_keywords)
            
            # 8. Additional features
            feature_keywords = self._analyze_features(movie_data)
            keywords.update(feature_keywords)
            
            # Filter out empty strings and join
            filtered_keywords = [kw for kw in keywords if kw and isinstance(kw, str)]
            
            return ', '.join(filtered_keywords) if filtered_keywords else 'general, entertainment'
            
        except Exception as e:
            print(f"⚠️  Error generating keywords for {movie_data.get('Title', 'Unknown')}: {e}")
            # Return basic fallback keywords
            return self._generate_fallback_keywords(movie_data)

    def _extract_genre_keywords(self, movie_data: Dict) -> List[str]:
        """Extract keywords based on genres"""
        genres = movie_data.get('Genre', '') or movie_data.get('Genres', '')
        if not genres or pd.isna(genres):
            return []
        
        # Ensure genres is a string
        genres_str = str(genres)
        keywords = []
        
        for genre, genre_words in self.genre_keywords.items():
            if genre.lower() in genres_str.lower():
                keywords.extend(genre_words)
        
        return keywords

    def _analyze_plot(self, plot_text: Any) -> List[str]:
        """Analyze plot text to extract keywords with safe handling"""
        if not plot_text or pd.isna(plot_text):
            return []
        
        # Ensure plot_text is a string
        plot_str = str(plot_text)
        if not plot_str.strip():
            return []
        
        try:
            plot_lower = plot_str.lower()
            keywords = []
            
            # Plot themes
            themes = {
                'redemption': ['redeem', 'second chance', 'forgiveness', 'atonement'],
                'betrayal': ['betray', 'traitor', 'backstab', 'double cross'],
                'coming-of-age': ['grow up', 'teenager', 'adolescent', 'maturity', 'young adult'],
                'justice': ['justice', 'court', 'law', 'righteous', 'legal'],
                'family': ['family', 'parent', 'child', 'sibling', 'mother', 'father'],
                'friendship': ['friend', 'buddy', 'companion', 'loyalty'],
                'love': ['love', 'romance', 'relationship', 'heart', 'passion'],
                'revenge': ['revenge', 'vengeance', 'retaliation', 'payback'],
                'survival': ['survive', 'wilderness', 'stranded', 'endure', 'alive'],
                'identity': ['identity', 'self-discovery', 'who am i', 'purpose'],
                'power': ['power', 'control', 'authority', 'dominance'],
                'freedom': ['freedom', 'liberty', 'escape', 'liberation']
            }
            
            for theme, indicators in themes.items():
                if any(indicator in plot_lower for indicator in indicators):
                    keywords.append(theme)
            
            # Plot structure keywords
            if any(word in plot_lower for word in ['flashback', 'memory', 'past']):
                keywords.append('non-linear')
            if any(word in plot_lower for word in ['dream', 'fantasy', 'imagination']):
                keywords.append('surreal')
            if any(word in plot_lower for word in ['mystery', 'secret', 'hidden']):
                keywords.append('mysterious')
            if any(word in plot_lower for word in ['journey', 'travel', 'adventure']):
                keywords.append('journey')
                
            return list(set(keywords))
            
        except Exception as e:
            print(f"⚠️  Error analyzing plot: {e}")
            return []

    def _analyze_mood(self, movie_data: Dict) -> List[str]:
        """Analyze movie mood from various data points"""
        moods = set()
        
        # From genres
        genres = movie_data.get('Genre', '') or movie_data.get('Genres', '')
        if genres and not pd.isna(genres):
            genres_str = str(genres).lower()
            if any(g in genres_str for g in ['comedy', 'rom-com']):
                moods.add('funny')
            if any(g in genres_str for g in ['drama', 'romance']):
                moods.add('emotional')
            if any(g in genres_str for g in ['thriller', 'horror', 'mystery']):
                moods.add('suspenseful')
            if any(g in genres_str for g in ['action', 'adventure']):
                moods.add('exciting')
            if any(g in genres_str for g in ['film-noir', 'crime']):
                moods.add('dark')
        
        # From plot
        plot = movie_data.get('Plot', '')
        if plot and not pd.isna(plot):
            plot_str = str(plot).lower()
            if any(word in plot_str for word in ['hope', 'triumph', 'success', 'joy']):
                moods.add('uplifting')
            if any(word in plot_str for word in ['death', 'tragedy', 'loss', 'betrayal']):
                moods.add('dark')
            if any(word in plot_str for word in ['philosophy', 'meaning', 'existential']):
                moods.add('thoughtful')
        
        return list(moods)

    def _analyze_director_style(self, director: Any) -> List[str]:
        """Analyze director's style"""
        if not director or pd.isna(director):
            return []
        
        director_str = str(director).lower()
        styles = []
        
        # Director style mappings
        director_styles = {
            'wes anderson': ['quirky', 'symmetrical', 'colorful', 'detailed'],
            'david lynch': ['surreal', 'dreamlike', 'psychological', 'unconventional'],
            'quentin tarantino': ['stylized', 'dialogue-heavy', 'violent', 'non-linear'],
            'christopher nolan': ['complex', 'mind-bending', 'epic', 'technical'],
            'martin scorsese': ['gritty', 'character-driven', 'urban', 'intense'],
            'steven spielberg': ['epic', 'emotional', 'adventure', 'family-friendly'],
            'alfred hitchcock': ['suspense', 'psychological', 'masterful', 'tense'],
            'stanley kubrick': ['perfectionist', 'symbolic', 'technical', 'cold']
        }
        
        for dir_pattern, style_words in director_styles.items():
            if dir_pattern in director_str:
                styles.extend(style_words)
                break
        
        return styles

    def _analyze_cultural_context(self, movie_data: Dict) -> List[str]:
        """Analyze cultural and geographical context"""
        cultural_keys = []
        
        # Country analysis
        country = movie_data.get('Country', '')
        if country and not pd.isna(country):
            country_str = str(country).lower()
            if 'india' in country_str or 'hindi' in country_str:
                cultural_keys.extend(['bollywood', 'indian cinema', 'south asian'])
            if 'japan' in country_str:
                cultural_keys.extend(['japanese cinema', 'east asian'])
            if 'korea' in country_str:
                cultural_keys.extend(['korean cinema', 'east asian'])
            if 'france' in country_str:
                cultural_keys.extend(['french cinema', 'european'])
            if 'germany' in country_str:
                cultural_keys.extend(['german cinema', 'european'])
            if 'spain' in country_str:
                cultural_keys.extend(['spanish cinema', 'european'])
            if 'mexico' in country_str:
                cultural_keys.extend(['mexican cinema', 'latin american'])
            if 'china' in country_str:
                cultural_keys.extend(['chinese cinema', 'east asian'])
        
        # Language analysis
        language = movie_data.get('Language', '') or movie_data.get('Primary_Language', '')
        if language and not pd.isna(language):
            lang_str = str(language).lower()
            if any(lang in lang_str for lang in ['hindi', 'tamil', 'telugu', 'bengali']):
                cultural_keys.extend(['indian', 'south asian'])
            if 'french' in lang_str:
                cultural_keys.append('french language')
            if 'german' in lang_str:
                cultural_keys.append('german language')
            if 'spanish' in lang_str:
                cultural_keys.append('spanish language')
            if 'japanese' in lang_str:
                cultural_keys.append('japanese language')
            if 'korean' in lang_str:
                cultural_keys.append('korean language')
        
        return list(set(cultural_keys))

    def _analyze_era(self, year: Any) -> List[str]:
        """Analyze the era of the movie"""
        if not year or pd.isna(year) or year == 0:
            return []
        
        try:
            year_int = int(year)
            if year_int < 1940:
                return ['classic cinema', 'vintage', 'golden age']
            elif year_int < 1970:
                return ['mid-century', 'retro', '60s cinema']
            elif year_int < 1990:
                return ['70s-80s', 'retro', 'modern classic']
            elif year_int < 2010:
                return ['90s-2000s', 'contemporary']
            else:
                return ['recent', 'modern', '21st century']
        except (ValueError, TypeError):
            return []

    def _analyze_awards(self, awards: Any) -> List[str]:
        """Analyze awards and recognition"""
        if not awards or pd.isna(awards):
            return []
        
        awards_str = str(awards).lower()
        award_keys = []
        
        if 'oscar' in awards_str:
            award_keys.append('oscar-winning')
        if 'cannes' in awards_str:
            award_keys.append('cannes-recognized')
        if 'sundance' in awards_str:
            award_keys.append('sundance')
        if 'bafta' in awards_str:
            award_keys.append('bafta-winning')
        if any(word in awards_str for word in ['critics', 'critical acclaim']):
            award_keys.append('critically acclaimed')
        
        return award_keys

    def _analyze_features(self, movie_data: Dict) -> List[str]:
        """Analyze additional features"""
        features = []
        
        # Runtime-based features
        runtime = movie_data.get('Runtime_mins', 0)
        if runtime > 150:
            features.append('epic length')
        elif runtime < 90:
            features.append('short runtime')
        
        # Rating-based features
        rating = movie_data.get('IMDb_Rating', 0) or movie_data.get('imdbRating', 0)
        if rating >= 8.0:
            features.append('highly rated')
        elif rating >= 7.0:
            features.append('well rated')
        
        # Box office success
        box_office = movie_data.get('BoxOffice', '')
        if box_office and box_office != 'N/A':
            features.append('box office success')
        
        return features

    def _generate_fallback_keywords(self, movie_data: Dict) -> str:
        """Generate basic fallback keywords when analysis fails"""
        fallback_keywords = []
        
        # Basic genre keywords
        genres = movie_data.get('Genre', '') or movie_data.get('Genres', '')
        if genres and not pd.isna(genres):
            genre_list = str(genres).split(',')
            fallback_keywords.extend([genre.strip().lower() for genre in genre_list[:2]])
        
        # Basic year context
        year = movie_data.get('Year', 0)
        if year and year > 0:
            if year >= 2010:
                fallback_keywords.append('recent')
            elif year >= 2000:
                fallback_keywords.append('2000s')
            else:
                fallback_keywords.append('classic')
        
        # Country context
        country = movie_data.get('Country', '')
        if country and not pd.isna(country):
            if 'India' in str(country):
                fallback_keywords.append('indian cinema')
            elif any(c in str(country) for c in ['USA', 'United States']):
                fallback_keywords.append('american')
            elif 'UK' in str(country) or 'United Kingdom' in str(country):
                fallback_keywords.append('british')
        
        return ', '.join(fallback_keywords) if fallback_keywords else 'entertainment, general'