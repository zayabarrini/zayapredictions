#!/usr/bin/env python3
"""
Main Movie Recommendation System
Now supports different input CSV files and prevents duplicates
"""

import time
import pandas as pd
import os
import numpy as np
import argparse
import sys
from src.data_loader import DataLoader
from src.recommender import MovieRecommender
from src.omdb_client import OMDbClient
from src.utils import (display_recommendations, plot_recommendations, 
                      export_recommendations_to_csv, create_detailed_analysis_csv,
                      ensure_directories)
from config.config import Config
from src.tmdb_client import TMDBClient
from src.keyword_generator import KeywordGenerator

def debug_tmdb_api():
        """Debug TMDB API configuration"""
        from src.tmdb_client import TMDBClient
        import requests
        
        print("🔧 Debugging TMDB API...")
        print(f"TMDB API Key configured: {bool(Config.TMDB_API_KEY and Config.TMDB_API_KEY != 'your_tmdb_api_key_here')}")
        
        if Config.TMDB_API_KEY and Config.TMDB_API_KEY != 'your_tmdb_api_key_here':
            # Test the API key with a simple request
            test_url = f"https://api.themoviedb.org/3/movie/550?api_key={Config.TMDB_API_KEY}"
            try:
                response = requests.get(test_url, timeout=10)
                if response.status_code == 200:
                    print("✅ TMDB API key is valid!")
                    data = response.json()
                    print(f"✅ Test movie: {data.get('title', 'Unknown')}")
                    return True
                elif response.status_code == 401:
                    print("❌ TMDB API key is invalid (401 Unauthorized)")
                    return False
                else:
                    print(f"❌ TMDB API returned status code: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ TMDB API test failed: {e}")
                return False
        else:
            print("❌ TMDB API key not configured in .env file")
            return False
        
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--ratings', '-r', default='data/ratings.csv',
                      help='Path to your ratings CSV file (default: data/ratings.csv)')
    parser.add_argument('--movies', '-m', default='data/movies_to_rate.csv',
                      help='Path to movies to rate CSV file (default: data/movies_to_rate.csv)')
    parser.add_argument('--output-prefix', '-o', 
                      help='Custom prefix for output files (uses input file names by default)')
    parser.add_argument('--no-plots', action='store_true',
                      help='Skip generating plots')
    parser.add_argument('--top-n', type=int, default=15,
                      help='Number of top recommendations to display (default: 15)')
    
    return parser.parse_args()

def get_file_basename(filepath):
    """Extract base name from filepath for use in output naming"""
    filename = os.path.basename(filepath)
    return os.path.splitext(filename)[0]

def remove_duplicate_movies(movies_df):
    """Remove duplicate movies from the dataframe"""
    initial_count = len(movies_df)
    
    # Remove exact duplicates
    movies_df = movies_df.drop_duplicates(subset=['Title', 'Year'], keep='first')
    
    # Remove duplicates where Year might be missing or different
    movies_df = movies_df.sort_values(['Title', 'Year'], na_position='first')
    movies_df = movies_df.drop_duplicates(subset=['Title'], keep='first')
    
    final_count = len(movies_df)
    
    if initial_count != final_count:
        print(f"🔄 Removed {initial_count - final_count} duplicate movies")
    
    return movies_df

def check_for_auto_recommendations(movies_file):
    """Check if auto-generated recommendations exist and use them"""
    auto_file = "data/auto_generated_movies_to_rate.csv"
    
    if os.path.exists(auto_file) and movies_file == "data/movies_to_rate.csv":
        print("🤖 Found auto-generated recommendations!")
        choice = input("Use auto-generated recommendations? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            return auto_file
    
    return movies_file

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Generate output prefix from input files if not provided
    if args.output_prefix is None:
        ratings_name = get_file_basename(args.ratings)
        movies_name = get_file_basename(args.movies)
        output_prefix = f"{ratings_name}_{movies_name}"
    else:
        output_prefix = args.output_prefix
    
    print("🎬 Movie Recommendation System - Custom Format")
    print("=" * 50)
    print(f"📁 Input files:")
    print(f"   Ratings: {args.ratings}")
    print(f"   Movies to rate: {args.movies}")
    print(f"   Output prefix: {output_prefix}")
    print("=" * 50)
    
    # Ensure all output directories exist
    ensure_directories()
    
    # Initialize components
    data_loader = DataLoader()
    recommender = MovieRecommender()
    omdb_client = OMDbClient()
    
    # Load your ratings
    try:
        if not os.path.exists(args.ratings):
            print(f"❌ Error: Ratings file not found: {args.ratings}")
            print("💡 Please check the file path or use --ratings option")
            return
        
        ratings_df = data_loader.load_ratings(args.ratings)
        print(f"📊 Loaded {len(ratings_df)} movies from ratings file")
        print(f"⭐ Found {ratings_df['Your_Rating'].notna().sum()} rated movies")
        
        # Show some stats about your ratings
        print(f"\n📈 Your Rating Stats:")
        print(f"  Average rating: {ratings_df['Your_Rating'].mean():.2f}")
        print(f"  Rating range: {ratings_df['Your_Rating'].min()} - {ratings_df['Your_Rating'].max()}")
        
        # Show top genres
        genre_counts = ratings_df['Genres'].value_counts().head(3)
        if not genre_counts.empty:
            print(f"  Common genres: {genre_counts.to_dict()}")
        
    except FileNotFoundError:
        print(f"❌ Error: {args.ratings} not found")
        return
    except Exception as e:
        print(f"❌ Error loading ratings: {e}")
        return
    
    # Prepare training data
    try:
        X, y, feature_names = data_loader.prepare_features(ratings_df)
        print(f"\n🤖 Training model with {len(X)} samples and {X.shape[1]} features...")
        
        if len(X) == 0:
            print("❌ No training data available. Please check your ratings file.")
            return
            
    except Exception as e:
        print(f"❌ Error preparing features: {e}")
        return

    # Train model
    try:
        metrics = recommender.train(X, y, feature_names)
        print(f"✅ Model trained successfully!")
        print(f"   Train R²: {metrics['train_score']:.3f}")
        print(f"   Test R²: {metrics['test_score']:.3f}")
        print(f"   MAE: {metrics['mae']:.3f}")
        print(f"   Training samples: {metrics['training_samples']}")
        print(f"   Test samples: {metrics['test_samples']}")
        print(f"   Features used: {len(metrics['features_used'])}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    # Show feature importance
    try:
        feature_importance = recommender.get_feature_importance()  # REMOVE the argument
        print("\n🔍 Top Feature Importance:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
    except Exception as e:
        print(f"⚠️  Could not calculate feature importance: {e}")
    
    # Check for auto-recommendations
    movies_file = check_for_auto_recommendations(args.movies)
    
    # Load movies to rate
    try:
        if not os.path.exists(movies_file):
            print(f"❌ Error: Movies file not found: {movies_file}")
            print("💡 Please check the file path or use --movies option")
            return
        
        movies_to_rate = data_loader.load_movies_to_rate(movies_file)
        print(f"\n🎭 Loaded {len(movies_to_rate)} movies to get recommendations for")
        print(f"   Sample movies: {movies_to_rate['Title'].head(3).tolist()}")
        
        # 🆕 REMOVE MOVIES THAT ARE ALREADY IN RATINGS
        print("🔍 Checking for movies already in your ratings...")
        initial_movie_count = len(movies_to_rate)
        movies_to_rate = data_loader.remove_already_rated_movies(movies_to_rate, ratings_df)
        
        if len(movies_to_rate) == 0:
            print("❌ All movies in the input file are already in your ratings.csv")
            print("💡 Please add new movies to rate in your movies_to_rate.csv file")
            return
        elif len(movies_to_rate) < initial_movie_count:
            print(f"✅ Filtered to {len(movies_to_rate)} new movies to rate")
        
    except FileNotFoundError:
        print(f"❌ Error: {movies_file} not found")
        return
    except Exception as e:
        print(f"❌ Error loading movies to rate: {e}")
        return
    
    # Get movie details from OMDb
    print("🌐 Fetching movie details from OMDb...")
    try:
        movies_data = []
        
        for _, movie_row in movies_to_rate.iterrows():
            title = movie_row['Title']
            year = movie_row.get('Year', None)
            
            # Prepare fallback data from CSV
            fallback_data = {
                'Director': movie_row.get('Director', ''),
                'Country_From_CSV': movie_row.get('Country_From_CSV', ''),
                'Description': movie_row.get('Description', ''),
                'Runtime_mins_From_CSV': movie_row.get('Runtime_mins_From_CSV', 0),
                'Genres_From_CSV': movie_row.get('Genres_From_CSV', ''),
                'IMDb_Rating_From_CSV': movie_row.get('IMDb_Rating_From_CSV', 0),
                'Num_Votes_From_CSV': movie_row.get('Num_Votes_From_CSV', 0),
                'Const': movie_row.get('Const', ''),
                'Release_Date': movie_row.get('Release_Date', '')
            }
            
            print(f"🌐 Fetching {len(movies_data) + 1}/{len(movies_to_rate)}: {title}")
            movie_data = omdb_client.search_movie(title, year, fallback_data)
            if movie_data:
                # Add the original CSV data to the movie data
                movie_data.update({
                    'Keywords_From_CSV': movie_row.get('Keywords_From_CSV', ''),
                    'Female_Critiques_From_CSV': movie_row.get('Female_Critiques_From_CSV', ''),
                    'Hours_Themes_Alignment_From_CSV': movie_row.get('Hours_Themes_Alignment_From_CSV', ''),
                    'Awards_From_CSV': movie_row.get('Awards_From_CSV', ''),
                    'Narrative_Type_From_CSV': movie_row.get('Narrative_Type_From_CSV', ''),
                    'Has_Oscar_From_CSV': movie_row.get('Has_Oscar_From_CSV', 0)
                })
                movies_data.append(movie_data)
            time.sleep(0.2)  # Be nice to the API
        
        if movies_data:
            movie_details = pd.DataFrame(movies_data)
            print(f"✅ Found details for {len(movie_details)} movies")
        else:
            print("❌ No movie details found from OMDb")
            return
            
    except Exception as e:
        print(f"❌ Error fetching movie details: {e}")
        return
    
    # Enhanced keyword generation (always available, doesn't need API)
    print("🎬 Generating enhanced keywords from available data...")
    keyword_generator = KeywordGenerator()

    enhanced_keywords = []
    for _, movie in movie_details.iterrows():
        keywords = keyword_generator.generate_keywords(movie.to_dict())
        enhanced_keywords.append(keywords)

    movie_details['Enhanced_Keywords'] = enhanced_keywords
    print(f"✅ Generated enhanced keywords for {len(movie_details)} movies")

    if hasattr(Config, 'TMDB_API_KEY') and Config.TMDB_API_KEY != 'your_tmdb_api_key_here':
        if debug_tmdb_api():
            print("🎬 Enhancing with TMDB keywords and tags...")
            tmdb_client = TMDBClient()
            
            tmdb_data_list = []
            found_count = 0
            
            for idx, movie in movie_details.iterrows():
                if idx % 10 == 0:  # Progress indicator
                    print(f"🎬 TMDB progress: {idx+1}/{len(movie_details)}")
                    
                # Try with original title if available
                original_title = movie.get('Original_Title', '') if 'Original_Title' in movie else ''
                
                tmdb_data = tmdb_client.search_movie(
                    title=movie['Title'], 
                    year=movie.get('Year'), 
                    original_title=original_title
                )
                
                if tmdb_data:
                    tmdb_data_list.append(tmdb_data)
                    found_count += 1
                else:
                    # Add empty TMDB data structure to maintain alignment
                    tmdb_data_list.append({})
                
                time.sleep(0.15)  # Slightly faster but still respectful
            
            if found_count > 0:
                tmdb_df = pd.DataFrame(tmdb_data_list)
                # Merge TMDB data - use indices to maintain order
                movie_details = pd.concat([movie_details, tmdb_df], axis=1)
                print(f"✅ Added TMDB data for {found_count}/{len(movie_details)} movies")
            else:
                print("❌ No TMDB data found for any movies")
        else:
            print("❌ TMDB API key is invalid. Skipping TMDB enhancement.")
        
    else:
        print("ℹ️  TMDB API key not configured or invalid. Skipping keyword enhancement.")
    
    # Get recommendations
    print("\n🎯 Generating recommendations...")
    try:
        recommendations = recommender.predict_ratings(movie_details)
        
        # Remove duplicates from final recommendations
        initial_recommendations = len(recommendations)
        recommendations = remove_duplicate_movies(recommendations)
        if initial_recommendations != len(recommendations):
            print(f"🔄 Removed {initial_recommendations - len(recommendations)} duplicates from final recommendations")
        
        # Debug: Show what columns we actually have
        print(f"📋 Columns in recommendations: {recommendations.columns.tolist()}")
        
        # Display results
        display_recommendations(recommendations, top_n=args.top_n)
        
        # Export to CSV with custom naming
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Export basic recommendations
        csv_filename = f"outputs/recommendations/{output_prefix}_recommendations_{timestamp}.csv"
        csv_file = export_recommendations_to_csv(recommendations, metrics, csv_filename)
        
        # Export detailed analysis
        analysis_filename = f"outputs/recommendations/{output_prefix}_analysis_{timestamp}.csv"
        analysis_file = create_detailed_analysis_csv(recommendations, ratings_df, analysis_filename)
        
        # Generate and save plots
        if not args.no_plots:
            try:
                total_movies = len(recommendations)
                print(f"\n📊 Generating plots for {total_movies} movies...")
                
                if total_movies <= 4:
                    # Very small dataset - use simple plot
                    from src.utils import create_simple_plot_for_small_dataset
                    create_simple_plot_for_small_dataset(recommendations, timestamp, output_prefix)
                elif total_movies <= 15:
                    # Small dataset - use standard plots but adjust parameters
                    plot_recommendations(recommendations, top_n=total_movies, save_plots=True, 
                                      plot_prefix=output_prefix)
                else:
                    # Larger dataset - use standard plots
                    plot_recommendations(recommendations, top_n=args.top_n, save_plots=True, 
                                      plot_prefix=output_prefix)
                    
            except Exception as e:
                print(f"⚠️  Could not generate plots: {e}")
        else:
            print("📊 Skipping plots as requested")
        
        # Save recommendations in original format (backward compatibility)
        recommendations.to_csv('data/latest_recommendations.csv', index=False)
        print(f"\n💾 Backward compatibility: Recommendations also saved to data/latest_recommendations.csv")
        
        # Show why movies are recommended
        print("\n🎯 Why these recommendations?")
        print("The system learned from your high ratings for:")
        high_rated = ratings_df[ratings_df['Your_Rating'] >= 8]
        if not high_rated.empty:
            common_tags = []
            if high_rated['Has_LGBT'].sum() > len(high_rated) * 0.3:
                common_tags.append("LGBT themes")
            if high_rated['Has_Cinematography'].sum() > len(high_rated) * 0.3:
                common_tags.append("cinematography")
            if high_rated['Has_Screenplay'].sum() > len(high_rated) * 0.3:
                common_tags.append("screenplay")
            if high_rated['Has_Plot_Twist'].sum() > len(high_rated) * 0.3:
                common_tags.append("plot twists")
                
            if common_tags:
                print(f"  • Movies with: {', '.join(common_tags)}")
            print(f"  • Average runtime: {high_rated['Runtime_mins'].mean():.0f} mins")
            
            # Show top genres in high-rated movies
            top_genres = high_rated['Genres'].value_counts().head(3)
            if not top_genres.empty:
                print(f"  • Preferred genres: {', '.join(top_genres.index.astype(str))}")
        
        print(f"\n🎉 All outputs saved with prefix: '{output_prefix}'")
        print(f"   📊 CSV files: outputs/recommendations/")
        print(f"   📈 Plots: outputs/plots/")
        print(f"\n💡 To run with different files: python main.py --ratings your_ratings.csv --movies your_movies.csv")
        
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        
    

if __name__ == "__main__":
    main()