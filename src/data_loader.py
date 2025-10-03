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
        """Load and preprocess ratings CSV with IMDb export format"""
        df = pd.read_csv(filepath)
        
        print(f"📋 Original columns in file: {df.columns.tolist()}")
        
        # Clean column names - handle IMDb export format
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Map IMDb export column names to standardized names
        column_mapping = {
            'Your_Rating': 'Your_Rating',
            'IMDb_Rating': 'IMDb_Rating', 
            'Runtime_(mins)': 'Runtime_mins',
            'Year': 'Year',
            'Num_Votes': 'Num_Votes',
            'Title': 'Title',
            'Genres': 'Genres',
            'Directors': 'Directors',
            'Release_Date': 'Release_Date',
            'URL': 'URL',
            'Title_Type': 'Title_Type',
            'Description': 'Description'
        }
        
        # Apply mapping for existing columns
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Clean and convert data types
        numeric_columns = ['Your_Rating', 'IMDb_Rating', 'Runtime_mins', 'Year', 'Num_Votes']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract additional features from your data
        df = self._extract_additional_features(df)
        
        print(f"✅ Processed {len(df)} movies with columns: {df.columns.tolist()}")
        
        return df
    
    def _extract_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional features from your detailed CSV"""
        # Extract tags from your special columns
        df['Has_LGBT'] = df.apply(lambda x: self._has_tag(x, 'LGBT'), axis=1)
        df['Has_Cinematography'] = df.apply(lambda x: self._has_tag(x, 'Cinematography'), axis=1)
        df['Has_Screenplay'] = df.apply(lambda x: self._has_tag(x, 'Screenplay'), axis=1)
        df['Has_Plot_Twist'] = df.apply(lambda x: self._has_tag(x, 'Plot, Twist, Drama'), axis=1)
        
        # Extract genre features
        if 'Genres' in df.columns:
            df['Genre_Count'] = df['Genres'].str.count(',') + 1
            df['Is_Drama'] = df['Genres'].str.contains('Drama', na=False).astype(int)
            df['Is_Comedy'] = df['Genres'].str.contains('Comedy', na=False).astype(int)
            df['Is_Crime'] = df['Genres'].str.contains('Crime', na=False).astype(int)
            df['Is_History'] = df['Genres'].str.contains('History', na=False).astype(int)
            df['Is_Romance'] = df['Genres'].str.contains('Romance', na=False).astype(int)
            df['Is_Action'] = df['Genres'].str.contains('Action', na=False).astype(int)
            df['Is_Adventure'] = df['Genres'].str.contains('Adventure', na=False).astype(int)
            df['Is_Sci-Fi'] = df['Genres'].str.contains('Sci-Fi', na=False).astype(int)
        else:
            # Set default values if Genres column doesn't exist
            df['Genre_Count'] = 1
            for genre in ['Drama', 'Comedy', 'Crime', 'History', 'Romance', 'Action', 'Adventure', 'Sci-Fi']:
                df[f'Is_{genre.replace("-", "_")}'] = 0
        
        # Add these to feature columns
        self.feature_columns.extend([
            'Has_LGBT', 'Has_Cinematography', 'Has_Screenplay', 'Has_Plot_Twist',
            'Genre_Count', 'Is_Drama', 'Is_Comedy', 'Is_Crime', 'Is_History', 
            'Is_Romance', 'Is_Action', 'Is_Adventure', 'Is_Sci_Fi'
        ])
        
        return df
    
    def _has_tag(self, row, tag: str) -> int:
        """Check if row has specific tag in any column"""
        for col in row.index:
            if pd.notna(row[col]) and tag in str(row[col]):
                return 1
        return 0
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training, return feature names"""
        # Filter out movies without ratings
        rated_movies = df[df['Your_Rating'].notna()].copy()
        
        print(f"🎯 Using {len(rated_movies)} rated movies for training")
        
        # Fill missing values
        for col in self.feature_columns:
            if col in rated_movies.columns:
                if rated_movies[col].dtype in ['int64', 'float64']:
                    rated_movies[col] = rated_movies[col].fillna(0)
                else:
                    rated_movies[col] = rated_movies[col].fillna('')
        
        # Prepare features - only use columns that exist and are numeric
        available_features = []
        for col in self.feature_columns:
            if col in rated_movies.columns:
                # Ensure the column is numeric for training
                if rated_movies[col].dtype in ['int64', 'float64']:
                    available_features.append(col)
                else:
                    print(f"⚠️  Skipping non-numeric feature: {col}")
        
        print(f"🔧 Using {len(available_features)} numeric features: {available_features}")
        
        X = rated_movies[available_features].values
        y = rated_movies['Your_Rating'].values
        
        return X, y, available_features
    
    def load_movies_to_rate(self, filepath: str) -> pd.DataFrame:
        """Load movies that need predictions - supports multiple formats"""
        df = pd.read_csv(filepath)
        
        print(f"📋 Columns in movies file: {df.columns.tolist()}")
        
        # Handle different file formats
        if 'Film (Year)' in df.columns:
            return self._load_custom_format(df)
        elif 'Title' in df.columns:
            return self._load_imdb_format(df)
        else:
            raise ValueError("Unsupported CSV format. Expected 'Title' column or 'Film (Year)' column")
    
    def _load_imdb_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load movies from standard IMDb export format"""
        print("📁 Detected IMDb export format")
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Debug: show what columns we have and first few rows
        print(f"🔍 Available columns: {df.columns.tolist()}")
        print(f"🔍 First few rows data types:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")
        
        # Map to standardized names - FIXED: Use correct column mapping
        column_mapping = {
            'Title': 'Title',
            'Original_Title': 'Original_Title',  # Add this
            'Year': 'Year',
            'Directors': 'Director',
            'Genres': 'Genres',
            'IMDb_Rating': 'IMDb_Rating',
            'Runtime_(mins)': 'Runtime_mins',
            'Num_Votes': 'Num_Votes',
            'Release_Date': 'Release_Date',
            'URL': 'URL',
            'Title_Type': 'Title_Type',
            'Description': 'Description',
            'Const': 'Const',
            'Your_Rating': 'Your_Rating'  # In case some are already rated
        }
        
        # Apply mapping for existing columns
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Debug: Check what we have after renaming
        print(f"🔍 After renaming - Title column sample: {df['Title'].head(3).tolist() if 'Title' in df.columns else 'NO TITLE COLUMN'}")
        
        # Ensure we have the Title column
        if 'Title' not in df.columns:
            print("❌ ERROR: No Title column found after mapping!")
            print(f"   Available columns: {df.columns.tolist()}")
            raise ValueError("CSV must contain a Title column")
        
        # Clean and convert data types with proper error handling
        def safe_convert_to_string(series):
            """Safely convert series to string, handling any values"""
            return series.fillna('').astype(str).str.strip()
        
        def safe_convert_to_numeric(series, errors='coerce'):
            """Safely convert series to numeric"""
            return pd.to_numeric(series, errors=errors)
        
        # Clean title and other string columns
        string_columns = ['Title', 'Original_Title', 'Director', 'Genres', 'Description']
        for col in string_columns:
            if col in df.columns:
                df[col] = safe_convert_to_string(df[col])
                print(f"📝 {col}: {df[col].head(2).tolist()}")
        
        # Convert numeric columns - be more careful about which columns to convert
        numeric_mappings = [
            ('Year', 'Year'),
            ('Runtime_mins', 'Runtime_mins'),
            ('IMDb_Rating', 'IMDb_Rating'), 
            ('Num_Votes', 'Num_Votes'),
            ('Your_Rating', 'Your_Rating')
        ]
        
        for source_col, target_col in numeric_mappings:
            if source_col in df.columns:
                # First check if the column contains mostly numeric data
                sample_values = df[source_col].head(3).tolist()
                print(f"🔢 Converting {source_col} - sample values: {sample_values}")
                
                df[target_col] = safe_convert_to_numeric(df[source_col])
                valid_count = df[target_col].notna().sum()
                print(f"   Converted {source_col} to numeric: {valid_count}/{len(df)} valid values")
        
        # Set default values for missing columns
        if 'Director' not in df.columns:
            df['Director'] = 'Unknown'
            print("⚠️  Director column not found, setting to 'Unknown'")
        
        if 'Country' not in df.columns:
            df['Country'] = 'Unknown'
        
        if 'Genres' not in df.columns:
            df['Genres'] = 'Unknown'
            print("⚠️  Genres column not found, setting to 'Unknown'")
        
        # Extract additional features
        df['Keywords'] = df.get('Genres', 'Unknown')
        
        # Add feature detection columns with default values
        feature_columns = ['Has_Female_Strength', 'Has_Male_Gaze', 'Has_Oscar']
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Remove any rows with invalid titles
        initial_count = len(df)
        df = df[
            df['Title'].notna() & 
            (df['Title'] != '') & 
            (df['Title'] != 'nan') &
            (~df['Title'].str.isnumeric())  # Remove titles that are just numbers
        ]
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"🔄 Removed {initial_count - final_count} rows with invalid titles")
        
        print(f"✅ Processed {len(df)} movies from IMDb format")
        print(f"📝 Final sample titles: {df['Title'].head(5).tolist()}")
        print(f"📊 Final data types:")
        for col in ['Title', 'Year', 'Runtime_mins', 'IMDb_Rating', 'Director']:
            if col in df.columns:
                print(f"   {col}: {df[col].dtype}")
        
        return df
    
    def _load_custom_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load movies from custom format with Film (Year) column - improved version"""
        print("📁 Detected custom format with 'Film (Year)' column")
        
        # Debug: print available columns
        print(f"Available columns in movies_to_rate: {df.columns.tolist()}")
        
        # First, try to use existing Title and Year columns if they exist
        if 'Title' in df.columns and 'Year' in df.columns:
            print("✅ Using existing Title and Year columns")
            # Clean the data
            df['Title'] = df['Title'].fillna('').astype(str)
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            
            # Use Directors from the CSV if available
            if 'Directors' in df.columns:
                df['Director'] = df['Directors'].fillna('').astype(str)
            elif 'Director' in df.columns:
                df['Director'] = df['Director'].fillna('').astype(str)
            else:
                df['Director'] = 'Unknown'
                
        else:
            # Fallback to Film (Year) extraction
            film_column = None
            possible_film_columns = ['Film (Year)', 'Film_(Year)', 'Film_Year', 'Film']
            
            for col in possible_film_columns:
                if col in df.columns:
                    film_column = col
                    break
            
            if film_column is None:
                raise ValueError(f"Could not find film or title column. Available columns: {df.columns.tolist()}")
            
            print(f"Using film column: {film_column}")
            
            # Convert to string and handle NaN values
            df[film_column] = df[film_column].fillna('Unknown').astype(str)
            
            # Extract title and year from the Film column
            titles = []
            years = []
            
            for film_str in df[film_column]:
                if pd.isna(film_str) or film_str in ['Unknown', 'nan', 'N/A (Film not found)']:
                    titles.append('Unknown')
                    years.append(0)
                    continue
                    
                film_str = str(film_str).strip()
                
                # Pattern to match "Title (Year)"
                match = re.match(r'(.+?)\s*\((\d{4})\)', film_str)
                if match:
                    title = match.group(1).strip()
                    year_str = match.group(2)
                    try:
                        year = int(year_str)
                    except (ValueError, TypeError):
                        year = 0
                    titles.append(title)
                    years.append(year)
                else:
                    # If no year in parentheses, check if it's just a title
                    if film_str and film_str != 'Unknown':
                        titles.append(film_str)
                        years.append(0)
                    else:
                        titles.append('Unknown')
                        years.append(0)
            
            df['Title'] = titles
            df['Year'] = years
            
            # Extract Director
            if 'Director' in df.columns:
                df['Director'] = df['Director'].fillna('').astype(str)
            else:
                df['Director'] = 'Unknown'
        
        # Now use the original CSV data that we want to preserve
        # Extract additional features from the detailed columns
        
        # Keywords from CSV (preserve the rich data you have)
        if 'Keywords / Tags' in df.columns:
            df['Keywords_From_CSV'] = df['Keywords / Tags'].fillna('').astype(str)
        else:
            df['Keywords_From_CSV'] = ''
        
        # Female critiques and themes
        if 'Female Critiques' in df.columns:
            df['Female_Critiques_From_CSV'] = df['Female Critiques'].fillna('').astype(str)
        else:
            df['Female_Critiques_From_CSV'] = ''
        
        if "Reasons for Alignment with The Hours' Themes" in df.columns:
            df['Hours_Themes_Alignment_From_CSV'] = df["Reasons for Alignment with The Hours' Themes"].fillna('').astype(str)
        else:
            df['Hours_Themes_Alignment_From_CSV'] = ''
        
        # Country information
        if 'Country / Primary Language' in df.columns:
            country_data = df['Country / Primary Language'].fillna('').astype(str)
            df['Country_From_CSV'] = country_data.str.split('/').str[0].str.strip()
        else:
            df['Country_From_CSV'] = ''
        
        # Awards information
        if 'Main Awards & Recognition' in df.columns:
            awards_data = df['Main Awards & Recognition'].fillna('').astype(str)
            df['Awards_From_CSV'] = awards_data
            df['Has_Oscar_From_CSV'] = awards_data.str.contains('Oscar|Oscars', case=False, na=False).astype(int)
        else:
            df['Awards_From_CSV'] = ''
            df['Has_Oscar_From_CSV'] = 0
        
        # Narrative type
        if 'Male / Female Narrative' in df.columns:
            df['Narrative_Type_From_CSV'] = df['Male / Female Narrative'].fillna('').astype(str)
        else:
            df['Narrative_Type_From_CSV'] = ''
        
        # Also preserve the original IMDb data if available
        if 'IMDb Rating' in df.columns:
            df['IMDb_Rating_From_CSV'] = pd.to_numeric(df['IMDb Rating'], errors='coerce').fillna(0)
        else:
            df['IMDb_Rating_From_CSV'] = 0
        
        if 'Runtime (mins)' in df.columns:
            df['Runtime_mins_From_CSV'] = pd.to_numeric(df['Runtime (mins)'], errors='coerce').fillna(0)
        else:
            df['Runtime_mins_From_CSV'] = 0
        
        if 'Genres' in df.columns:
            df['Genres_From_CSV'] = df['Genres'].fillna('').astype(str)
        else:
            df['Genres_From_CSV'] = ''
        
        if 'Num Votes' in df.columns:
            df['Num_Votes_From_CSV'] = pd.to_numeric(df['Num Votes'], errors='coerce').fillna(0)
        else:
            df['Num_Votes_From_CSV'] = 0
        
        print(f"✅ Successfully processed {len(df)} movies from custom format")
        print(f"📝 Titles: {df['Title'].tolist()}")
        print(f"📅 Years: {df['Year'].tolist()}")
        print(f"🎭 Directors: {df['Director'].tolist()}")
        
        return df
    
    def remove_duplicate_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate movies from dataframe"""
        if movies_df.empty:
            return movies_df
        
        initial_count = len(movies_df)
        
        # Create a clean title column for better matching
        movies_df['Clean_Title'] = movies_df['Title'].str.lower().str.strip()
        
        # Remove exact duplicates based on title and year
        movies_df = movies_df.drop_duplicates(subset=['Clean_Title', 'Year'], keep='first')
        
        # For movies with same title but different years, keep the one with valid year
        movies_df = movies_df.sort_values(['Clean_Title', 'Year'], 
                                       ascending=[True, False], 
                                       na_position='last')
        movies_df = movies_df.drop_duplicates(subset=['Clean_Title'], keep='first')
        
        # Remove the temporary column
        movies_df = movies_df.drop('Clean_Title', axis=1)
        
        final_count = len(movies_df)
        
        if initial_count != final_count:
            print(f"🔄 Removed {initial_count - final_count} duplicate movie entries")
        
        return movies_df
    
    def enhance_movie_features(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Add missing features to movies dataframe to match training features"""
        # Remove duplicates first
        movies_df = self.remove_duplicate_movies(movies_df)
        
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
            enhanced_df['Is_Action'] = enhanced_df['Genres'].str.contains('Action', na=False).astype(int)
            enhanced_df['Is_Adventure'] = enhanced_df['Genres'].str.contains('Adventure', na=False).astype(int)
            enhanced_df['Is_Sci_Fi'] = enhanced_df['Genres'].str.contains('Sci-Fi', na=False).astype(int)
        
        # Extract tags from plot, keywords, and other text fields
        text_columns = ['Plot', 'Keywords', 'Genres', 'Director', 'Country', 'Description']
        
        for idx, movie in enhanced_df.iterrows():
            # Combine all text fields for tag detection
            combined_text = ' '.join([
                str(movie.get(col, '')) for col in text_columns 
                if col in movie and pd.notna(movie[col])
            ]).lower()
            
            # Detect LGBT themes
            lgbt_keywords = ['lgbt', 'gay', 'lesbian', 'transgender', 'queer', 'homosexual', 'same-sex']
            enhanced_df.at[idx, 'Has_LGBT'] = 1 if any(keyword in combined_text for keyword in lgbt_keywords) else 0
            
            # Detect cinematography focus
            cinema_keywords = ['cinematography', 'visually stunning', 'beautiful shot', 'photography', 'aesthetic', 'visual masterpiece']
            enhanced_df.at[idx, 'Has_Cinematography'] = 1 if any(keyword in combined_text for keyword in cinema_keywords) else 0
            
            # Detect screenplay quality
            screenplay_keywords = ['screenplay', 'writing', 'dialogue', 'script', 'well-written', 'scriptwriting']
            enhanced_df.at[idx, 'Has_Screenplay'] = 1 if any(keyword in combined_text for keyword in screenplay_keywords) else 0
            
            # Detect plot twists
            twist_keywords = ['plot twist', 'twist', 'surprise ending', 'unexpected', 'shocking revelation']
            enhanced_df.at[idx, 'Has_Plot_Twist'] = 1 if any(keyword in combined_text for keyword in twist_keywords) else 0
        
        return enhanced_df
    
    def remove_already_rated_movies(self, movies_to_rate: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Remove movies from movies_to_rate that are already in ratings.csv"""
        if movies_to_rate.empty or ratings_df.empty:
            return movies_to_rate
        
        initial_count = len(movies_to_rate)
        
        # Create clean title columns for matching
        movies_to_rate['Clean_Title'] = movies_to_rate['Title'].str.lower().str.strip()
        ratings_df['Clean_Title'] = ratings_df['Title'].str.lower().str.strip()
        
        # Remove common suffixes and punctuation for better matching
        movies_to_rate['Clean_Title'] = movies_to_rate['Clean_Title'].str.replace(r'[^\w\s]', '', regex=True)
        ratings_df['Clean_Title'] = ratings_df['Clean_Title'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Get list of already rated movie titles
        rated_titles = set(ratings_df['Clean_Title'].dropna().tolist())
        
        # Filter out movies that are already rated
        filtered_movies = movies_to_rate[~movies_to_rate['Clean_Title'].isin(rated_titles)].copy()
        
        # Also check by year for more precise matching (if year is available)
        if 'Year' in movies_to_rate.columns and 'Year' in ratings_df.columns:
            rated_movies_with_year = ratings_df[['Clean_Title', 'Year']].dropna().drop_duplicates()
            rated_movies_with_year = rated_movies_with_year[rated_movies_with_year['Year'] > 0]
            
            # Create a set of (title, year) tuples for exact matching
            rated_titles_with_year = set(zip(rated_movies_with_year['Clean_Title'], rated_movies_with_year['Year']))
            
            # Remove movies with exact title and year match
            movies_to_rate_clean = filtered_movies.dropna(subset=['Year'])
            movies_to_rate_clean = movies_to_rate_clean[movies_to_rate_clean['Year'] > 0]
            
            exact_matches_mask = movies_to_rate_clean.apply(
                lambda row: (row['Clean_Title'], row['Year']) in rated_titles_with_year, axis=1
            )
            
            exact_match_indices = movies_to_rate_clean[exact_matches_mask].index
            filtered_movies = filtered_movies.drop(exact_match_indices)
        
        # Remove temporary column
        filtered_movies = filtered_movies.drop('Clean_Title', axis=1)
        
        final_count = len(filtered_movies)
        
        if initial_count != final_count:
            removed_count = initial_count - final_count
            print(f"🔄 Removed {removed_count} movies that are already in your ratings")
            if removed_count > 0:
                removed_titles = movies_to_rate[movies_to_rate['Clean_Title'].isin(rated_titles)]['Title'].tolist()
                print(f"📋 Removed titles: {removed_titles[:5]}{'...' if len(removed_titles) > 5 else ''}")
        
        return filtered_movies