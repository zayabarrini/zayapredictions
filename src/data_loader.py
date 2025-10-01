import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, List

class DataLoader:
    def __init__(self):
        self.feature_columns = [
            'IMDb_Rating', 'Runtime_mins', 'Year', 'Num_Votes'
        ]
        
    def load_ratings(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess ratings CSV with your specific format"""
        df = pd.read_csv(filepath)
        
        # Clean column names - handle your specific format
        df.columns = [col.strip().replace(' ', '_').replace('/', '_') for col in df.columns]
        
        # Map your column names to standardized names
        column_mapping = {
            'Your_Rating': 'Your_Rating',
            'IMDb_Rating': 'IMDb_Rating', 
            'Runtime_(mins)': 'Runtime_mins',
            'Year': 'Year',
            'Num_Votes': 'Num_Votes',
            'Title': 'Title',
            'Genres': 'Genres',
            'Directors': 'Directors',
            'Country_of_origin': 'Country'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Clean and convert data types
        df['Your_Rating'] = pd.to_numeric(df['Your_Rating'], errors='coerce')
        df['IMDb_Rating'] = pd.to_numeric(df['IMDb_Rating'], errors='coerce')
        df['Runtime_mins'] = pd.to_numeric(df['Runtime_mins'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Num_Votes'] = pd.to_numeric(df['Num_Votes'], errors='coerce')
        
        # Extract additional features from your data
        df = self._extract_additional_features(df)
        
        return df
    
    def _extract_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional features from your detailed CSV"""
        # Extract tags from your special columns
        df['Has_LGBT'] = df.apply(lambda x: self._has_tag(x, 'LGBT'), axis=1)
        df['Has_Cinematography'] = df.apply(lambda x: self._has_tag(x, 'Cinematography'), axis=1)
        df['Has_Screenplay'] = df.apply(lambda x: self._has_tag(x, 'Screenplay'), axis=1)
        df['Has_Plot_Twist'] = df.apply(lambda x: self._has_tag(x, 'Plot, Twist, Drama'), axis=1)
        
        # Extract genre features
        df['Genre_Count'] = df['Genres'].str.count(',') + 1
        df['Is_Drama'] = df['Genres'].str.contains('Drama', na=False).astype(int)
        df['Is_Comedy'] = df['Genres'].str.contains('Comedy', na=False).astype(int)
        df['Is_Crime'] = df['Genres'].str.contains('Crime', na=False).astype(int)
        df['Is_History'] = df['Genres'].str.contains('History', na=False).astype(int)
        df['Is_Romance'] = df['Genres'].str.contains('Romance', na=False).astype(int)
        
        # Add these to feature columns
        self.feature_columns.extend([
            'Has_LGBT', 'Has_Cinematography', 'Has_Screenplay', 'Has_Plot_Twist',
            'Genre_Count', 'Is_Drama', 'Is_Comedy', 'Is_Crime', 'Is_History', 'Is_Romance'
        ])
        
        return df
    
    def _has_tag(self, row, tag: str) -> int:
        """Check if row has specific tag in any column"""
        for col in row.index:
            if pd.notna(row[col]) and tag in str(row[col]):
                return 1
        return 0
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        # Filter out movies without ratings
        rated_movies = df[df['Your_Rating'].notna()].copy()
        
        # Fill missing values
        for col in self.feature_columns:
            if col in rated_movies.columns:
                if rated_movies[col].dtype in ['int64', 'float64']:
                    rated_movies[col] = rated_movies[col].fillna(0)
                else:
                    rated_movies[col] = rated_movies[col].fillna('')
        
        # Prepare features - only use columns that exist
        available_features = [col for col in self.feature_columns if col in rated_movies.columns]
        X = rated_movies[available_features].values
        y = rated_movies['Your_Rating'].values
        
        return X, y
    
    def load_movies_to_rate(self, filepath: str) -> pd.DataFrame:
        """Load movies that need predictions from your specific format"""
        df = pd.read_csv(filepath)
        
        # Debug: print available columns
        print(f"Available columns in movies_to_rate: {df.columns.tolist()}")
        
        # Handle different possible column names
        film_column = None
        possible_film_columns = ['Film (Year)', 'Film_(Year)', 'Film_Year', 'Film']
        
        for col in possible_film_columns:
            if col in df.columns:
                film_column = col
                break
        
        if film_column is None:
            raise ValueError(f"Could not find film column. Available columns: {df.columns.tolist()}")
        
        print(f"Using film column: {film_column}")
        
        # Extract title and year from the Film column
        def extract_title_year(film_str):
            if pd.isna(film_str):
                return pd.Series(['Unknown', 0])
            
            # Pattern to match "Title (Year)"
            match = re.match(r'(.+?)\s*\((\d{4})\)', str(film_str))
            if match:
                title = match.group(1).strip()
                year = int(match.group(2))
                return pd.Series([title, year])
            else:
                # If no year in parentheses, try to extract from string
                year_match = re.search(r'(\d{4})', str(film_str))
                year = int(year_match.group(1)) if year_match else 0
                title = re.sub(r'\s*\(\d{4}\)', '', str(film_str)).strip()
                return pd.Series([title, year])
        
        # Apply extraction
        df[['Title', 'Year']] = df[film_column].apply(extract_title_year)
        
        # Extract additional features from the detailed columns
        df['Keywords'] = df['Keywords / Tags'].fillna('') if 'Keywords / Tags' in df.columns else ''
        df['Director'] = df['Director'].fillna('')
        
        # Extract country from Country / Language column
        if 'Country / Language' in df.columns:
            df['Country'] = df['Country / Language'].str.split('/').str[0].str.strip().fillna('')
        else:
            df['Country'] = ''
        
        # Extract narrative type
        if 'M/F Narrative' in df.columns:
            df['Narrative_Type'] = df['M/F Narrative'].fillna('')
        else:
            df['Narrative_Type'] = ''
        
        # Extract awards information
        if 'Main Awards & Recognition' in df.columns:
            df['Has_Oscar'] = df['Main Awards & Recognition'].str.contains('Oscar|Oscars', case=False, na=False).astype(int)
            df['Awards_Text'] = df['Main Awards & Recognition'].fillna('')
        else:
            df['Has_Oscar'] = 0
            df['Awards_Text'] = ''
        
        # Extract context for additional features
        if 'Notable Critiques / Context' in df.columns:
            df['Has_Female_Strength'] = df['Notable Critiques / Context'].str.contains(
                'Female Strength|Proactive Female|Female Agency|Complex Women', case=False, na=False
            ).astype(int)
            df['Has_Male_Gaze'] = df['Notable Critiques / Context'].str.contains(
                'Male Gazey|Problematic', case=False, na=False
            ).astype(int)
        else:
            df['Has_Female_Strength'] = 0
            df['Has_Male_Gaze'] = 0
        
        print(f"Successfully processed {len(df)} movies to rate")
        print(f"Sample titles: {df['Title'].head(3).tolist()}")
        
        return df

    def enhance_movie_features(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Add missing features to movies dataframe to match training features"""
        # Make a copy to avoid modifying original
        enhanced_df = movies_df.copy()
        
        # Ensure all required feature columns exist
        for feature in self.feature_columns:
            if feature not in enhanced_df.columns:
                enhanced_df[feature] = 0  # Default value
        
        # Extract features from available data
        if 'Genres' in enhanced_df.columns:
            enhanced_df['Genre_Count'] = enhanced_df['Genres'].str.count(',') + 1
            enhanced_df['Is_Drama'] = enhanced_df['Genres'].str.contains('Drama', na=False).astype(int)
            enhanced_df['Is_Comedy'] = enhanced_df['Genres'].str.contains('Comedy', na=False).astype(int)
            enhanced_df['Is_Crime'] = enhanced_df['Genres'].str.contains('Crime', na=False).astype(int)
            enhanced_df['Is_History'] = enhanced_df['Genres'].str.contains('History', na=False).astype(int)
            enhanced_df['Is_Romance'] = enhanced_df['Genres'].str.contains('Romance', na=False).astype(int)
        
        # Extract tags from plot, keywords, and other text fields
        text_columns = ['Plot', 'Keywords', 'Genres', 'Director', 'Country']
        
        for idx, movie in enhanced_df.iterrows():
            # Combine all text fields for tag detection
            combined_text = ' '.join([
                str(movie.get(col, '')) for col in text_columns 
                if col in movie and pd.notna(movie[col])
            ]).lower()
            
            # Detect LGBT themes
            lgbt_keywords = ['lgbt', 'gay', 'lesbian', 'transgender', 'queer', 'homosexual']
            enhanced_df.at[idx, 'Has_LGBT'] = 1 if any(keyword in combined_text for keyword in lgbt_keywords) else 0
            
            # Detect cinematography focus
            cinema_keywords = ['cinematography', 'visually stunning', 'beautiful shot', 'photography', 'aesthetic']
            enhanced_df.at[idx, 'Has_Cinematography'] = 1 if any(keyword in combined_text for keyword in cinema_keywords) else 0
            
            # Detect screenplay quality
            screenplay_keywords = ['screenplay', 'writing', 'dialogue', 'script', 'well-written']
            enhanced_df.at[idx, 'Has_Screenplay'] = 1 if any(keyword in combined_text for keyword in screenplay_keywords) else 0
            
            # Detect plot twists
            twist_keywords = ['plot twist', 'twist', 'surprise ending', 'unexpected']
            enhanced_df.at[idx, 'Has_Plot_Twist'] = 1 if any(keyword in combined_text for keyword in twist_keywords) else 0
        
        return enhanced_df
