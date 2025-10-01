from datetime import datetime  # Add this import at the top of your file
import pandas as pd
import os
import numpy as np
from src.data_loader import DataLoader
from src.recommender import MovieRecommender
from src.omdb_client import OMDbClient
from src.utils import (display_recommendations, plot_recommendations, 
                      export_recommendations_to_csv, create_detailed_analysis_csv,
                      ensure_directories)
from config.config import Config

def main():
    print("🎬 Movie Recommendation System - Custom Format")
    print("=" * 50)
    
    # Ensure all output directories exist
    ensure_directories()
    
    # Initialize components
    data_loader = DataLoader()
    recommender = MovieRecommender()
    omdb_client = OMDbClient()
    
    # Load your ratings
    try:
        ratings_df = data_loader.load_ratings('data/ratings.csv')
        print(f"Loaded {len(ratings_df)} movies from ratings file")
        print(f"Found {ratings_df['Your_Rating'].notna().sum()} rated movies")
        
        # Show some stats about your ratings
        print(f"\nYour Rating Stats:")
        print(f"  Average rating: {ratings_df['Your_Rating'].mean():.2f}")
        print(f"  Rating range: {ratings_df['Your_Rating'].min()} - {ratings_df['Your_Rating'].max()}")
        print(f"  Common genres: {ratings_df['Genres'].value_counts().head(3).to_dict()}")
        
    except FileNotFoundError:
        print("Error: ratings.csv not found in data/ directory")
        return
    except Exception as e:
        print(f"Error loading ratings: {e}")
        return
    
    # Prepare training data
    try:
        X, y = data_loader.prepare_features(ratings_df)
        print(f"\nTraining model with {len(X)} samples and {X.shape[1]} features...")
        
        if len(X) == 0:
            print("No training data available. Please check your ratings file.")
            return
            
    except Exception as e:
        print(f"Error preparing features: {e}")
        return
    
    # Train model
    try:
        metrics = recommender.train(X, y)
        print(f"Model trained successfully!")
        print(f"Train R²: {metrics['train_score']:.3f}")
        print(f"Test R²: {metrics['test_score']:.3f}")
        print(f"MAE: {metrics['mae']:.3f}")
        print(f"Training samples: {metrics['training_samples']}")
        print(f"Test samples: {metrics['test_samples']}")
        
    except Exception as e:
        print(f"Error training model: {e}")
        return
    
    # Show feature importance
    try:
        available_features = [col for col in recommender.feature_columns if col in ratings_df.columns]
        feature_importance = recommender.get_feature_importance(available_features)
        print("\nTop Feature Importance:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    except Exception as e:
        print(f"Could not calculate feature importance: {e}")
    
    # Load movies to rate
    try:
        movies_to_rate = data_loader.load_movies_to_rate('data/movies_to_rate.csv')
        print(f"\nLoaded {len(movies_to_rate)} movies to get recommendations for")
        print("Sample movies:", movies_to_rate['Title'].head(3).tolist())
    except FileNotFoundError:
        print("Error: movies_to_rate.csv not found in data/ directory")
        return
    except Exception as e:
        print(f"Error loading movies to rate: {e}")
        return
    
    # Get movie details from OMDb
    print("\nFetching movie details from OMDb...")
    try:
        movie_details = omdb_client.batch_search_movies(movies_to_rate['Title'].tolist())
        print(f"Found details for {len(movie_details)} movies")
        
        if len(movie_details) == 0:
            print("No movie details found from OMDb")
            return
            
        # Merge with our additional features from movies_to_rate
        movie_details = movie_details.merge(
            movies_to_rate[['Title', 'Director', 'Country', 'Keywords', 
                          'Has_Female_Strength', 'Has_Male_Gaze', 'Has_Oscar']], 
            on='Title', 
            how='left'
        )
        
        # Enhance features to match training data format
        print("Enhancing features to match training format...")
        movie_details = data_loader.enhance_movie_features(movie_details)
        
        print(f"Final feature count: {len([col for col in recommender.feature_columns if col in movie_details.columns])}")
        
    except Exception as e:
        print(f"Error fetching movie details: {e}")
        return
    
    # Get recommendations
    print("Generating recommendations...")
    try:
        recommendations = recommender.predict_ratings(movie_details)
        
        # Debug: Show what columns we actually have
        print(f"📋 Columns in recommendations: {recommendations.columns.tolist()}")
        
        # Display results
        display_recommendations(recommendations, top_n=15)
        
        # Export to CSV
        csv_file = export_recommendations_to_csv(recommendations, metrics)
        
        # Export detailed analysis
        analysis_file = create_detailed_analysis_csv(recommendations, ratings_df)
               
        # Generate and save plots with size awareness
        try:
            total_movies = len(recommendations)
            print(f"📊 Generating plots for {total_movies} movies...")
            
            if total_movies <= 4:
                # Very small dataset - use simple plot
                from src.utils import create_simple_plot_for_small_dataset
                create_simple_plot_for_small_dataset(recommendations, 
                                                datetime.now().strftime("%Y%m%d_%H%M%S"),
                                                "movie_recommendations")
            elif total_movies <= 15:
                # Small dataset - use standard plots but adjust parameters
                plot_recommendations(recommendations, top_n=total_movies, save_plots=True, 
                                plot_prefix="movie_recommendations")
            else:
                # Larger dataset - use standard plots
                plot_recommendations(recommendations, top_n=15, save_plots=True, 
                                plot_prefix="movie_recommendations")
                
        except Exception as e:
            print(f"Could not generate plots: {e}")
            import traceback
            traceback.print_exc()
        
        # Save recommendations in original format (backward compatibility)
        recommendations.to_csv('data/recommendations.csv', index=False)
        print(f"\n📁 Backward compatibility: Recommendations also saved to data/recommendations.csv")
        
        # Show why movies are recommended
        print("\n🎯 Why these recommendations?")
        print("The system learned from your high ratings for:")
        high_rated = ratings_df[ratings_df['Your_Rating'] >= 9]
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
        
        print(f"\n🎉 All outputs saved in 'outputs/' folder:")
        print(f"   📊 CSV files: outputs/recommendations/")
        print(f"   📈 Plots: outputs/plots/")
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()