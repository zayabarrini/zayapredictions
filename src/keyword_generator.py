# Add this to src/utils.py or create a new file src/keyword_generator.py

import re
from typing import Dict, List

class KeywordGenerator:
    """Generate keywords from movie metadata when TMDB fails"""
    
    def __init__(self):
        self.genre_keywords = {
            'Drama': ['emotional', 'character-driven', 'relationships', 'conflict', 'human condition'],
            'Comedy': ['humor', 'funny', 'lighthearted', 'satire', 'comic relief'],
            'Action': ['thrilling', 'adventure', 'stunts', 'combat', 'suspense'],
            'Romance': ['love story', 'relationships', 'passion', 'heartfelt', 'emotional'],
            'Crime': ['investigation', 'justice', 'moral ambiguity', 'thriller', 'suspense'],
            'History': ['period piece', 'historical context', 'biographical', 'era-specific'],
            'Documentary': ['real-life', 'factual', 'educational', 'investigative', 'informative'],
            'Horror': ['fear', 'suspense', 'supernatural', 'thriller', 'psychological'],
            'Sci-Fi': ['futuristic', 'technology', 'speculative', 'space', 'innovation'],
            'Fantasy': ['magical', 'mythical', 'imaginary worlds', 'supernatural', 'adventure']
        }
        
        self.theme_keywords = {
            'female': ['women empowerment', 'female perspective', 'gender issues', 'feminist'],
            'lgbt': ['queer themes', 'LGBTQ+', 'identity', 'inclusion', 'diversity'],
            'social': ['social justice', 'inequality', 'class struggle', 'political'],
            'family': ['family dynamics', 'generational', 'relationships', 'domestic'],
            'cultural': ['cultural identity', 'tradition', 'heritage', 'ethnicity'],
            'political': ['government', 'power', 'corruption', 'activism'],
            'psychological': ['mental health', 'inner conflict', 'emotional journey'],
            'coming-of-age': ['youth', 'maturation', 'self-discovery', 'adolescence']
        }
    
    def generate_keywords(self, movie_data: Dict) -> str:
        """Generate comprehensive keywords from available movie data"""
        keywords = []
        
        # Add genres
        if 'Genre' in movie_data and movie_data['Genre']:
            genres = [g.strip() for g in movie_data['Genre'].split(',')]
            for genre in genres:
                if genre in self.genre_keywords:
                    keywords.extend(self.genre_keywords[genre])
        
        # Add from plot analysis
        if 'Plot' in movie_data and movie_data['Plot']:
            plot_keywords = self._analyze_plot(movie_data['Plot'])
            keywords.extend(plot_keywords)
        
        # Add from CSV data
        if 'Keywords_From_CSV' in movie_data and movie_data['Keywords_From_CSV']:
            csv_keywords = [k.strip() for k in movie_data['Keywords_From_CSV'].split(',')]
            keywords.extend(csv_keywords)
        
        # Add from themes alignment
        if 'Hours_Themes_Alignment_From_CSV' in movie_data and movie_data['Hours_Themes_Alignment_From_CSV']:
            theme_keywords = self._analyze_themes(movie_data['Hours_Themes_Alignment_From_CSV'])
            keywords.extend(theme_keywords)
        
        # Add from female critiques
        if 'Female_Critiques_From_CSV' in movie_data and movie_data['Female_Critiques_From_CSV']:
            female_keywords = self._analyze_female_perspective(movie_data['Female_Critiques_From_CSV'])
            keywords.extend(female_keywords)
        
        # Add country/cultural context
        if 'Country' in movie_data and movie_data['Country']:
            country_keywords = self._get_country_keywords(movie_data['Country'])
            keywords.extend(country_keywords)
        
        # Remove duplicates and clean
        unique_keywords = list(set([k.lower() for k in keywords if k]))
        return ', '.join(unique_keywords[:15])  # Limit to 15 most relevant
    
    def _analyze_plot(self, plot: str) -> List[str]:
        """Extract keywords from plot summary"""
        if not plot:
            return []
        
        plot_lower = plot.lower()
        keywords = []
        
        # Common plot elements
        plot_elements = {
            'friendship': ['friend', 'companion', 'buddy'],
            'betrayal': ['betray', 'traitor', 'backstab'],
            'redemption': ['redeem', 'forgiveness', 'second chance'],
            'revenge': ['revenge', 'vengeance', 'retaliation'],
            'justice': ['justice', 'court', 'law'],
            'survival': ['survive', 'endure', 'struggle'],
            'discovery': ['discover', 'find', 'uncover'],
            'transformation': ['transform', 'change', 'evolve']
        }
        
        for keyword, triggers in plot_elements.items():
            if any(trigger in plot_lower for trigger in triggers):
                keywords.append(keyword)
        
        return keywords
    
    def _analyze_themes(self, themes_text: str) -> List[str]:
        """Extract keywords from themes alignment text"""
        if not themes_text:
            return []
        
        themes_lower = themes_text.lower()
        keywords = []
        
        for theme, triggers in self.theme_keywords.items():
            if any(trigger in themes_lower for trigger in triggers):
                keywords.extend(self.theme_keywords[theme])
        
        return keywords
    
    def _analyze_female_perspective(self, critiques_text: str) -> List[str]:
        """Extract keywords from female critiques"""
        if not critiques_text:
            return []
        
        critiques_lower = critiques_text.lower()
        keywords = []
        
        female_indicators = [
            'female', 'woman', 'women', 'feminist', 'gender', 'patriarchy',
            'mother', 'daughter', 'sister', 'wife', 'girl', 'matriarchy'
        ]
        
        strength_indicators = [
            'strong', 'empower', 'agency', 'independent', 'leadership',
            'voice', 'autonomy', 'resilient', 'courage'
        ]
        
        if any(indicator in critiques_lower for indicator in female_indicators):
            keywords.extend(['female perspective', 'women-centric'])
        
        if any(indicator in critiques_lower for indicator in strength_indicators):
            keywords.extend(['female empowerment', 'strong women'])
        
        return keywords
    
    def _get_country_keywords(self, country: str) -> List[str]:
        """Get cultural keywords based on country"""
        country_keywords = {
            'USA': ['american', 'hollywood', 'western culture'],
            'UK': ['british', 'english', 'european'],
            'France': ['french', 'european', 'art house'],
            'Japan': ['japanese', 'asian', 'eastern philosophy'],
            'India': ['indian', 'bollywood', 'south asian'],
            'China': ['chinese', 'asian', 'eastern culture'],
            'Brazil': ['brazilian', 'latin american', 'tropical'],
            'Mexico': ['mexican', 'latin american', 'hispanic'],
            'Germany': ['german', 'european', 'central europe'],
            'Italy': ['italian', 'european', 'mediterranean'],
            'Russia': ['russian', 'eastern european', 'slavic'],
            'South Korea': ['korean', 'asian', 'eastern'],
            'Spain': ['spanish', 'european', 'mediterranean'],
            'Canada': ['canadian', 'north american', 'multicultural'],
            'Australia': ['australian', 'oceanic', 'outback']
        }
        
        for country_pattern, keywords in country_keywords.items():
            if country_pattern.lower() in country.lower():
                return keywords
        
        return ['international', 'global cinema']