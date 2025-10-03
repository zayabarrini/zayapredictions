#!/usr/bin/env python3
"""
Automated Movie Recommendation Generator
Learns from your ratings and generates new movie recommendations
"""

import pandas as pd
import os
import sys
from src.data_loader import DataLoader
from src.omdb_client import OMDbClient
from src.recommendation_generator import RecommendationGenerator
from src.utils import ensure_directories

def main():
    print("🤖 Automated Movie Recommendation Generator")
    print("=" * 50)
    
    # Ensure directories exist
    ensure_directories()
    
    # Initialize components
    data_loader = DataLoader()
    omdb_client = OMDbClient()
    recommendation_generator = RecommendationGenerator(omdb_client)
    
    # Load your ratings
    try:
        ratings_df = data_loader.load_ratings('data/ratings.csv')
        print(f"📊 Loaded {len(ratings_df)} movies from your ratings")
        print(f"⭐ Found {ratings_df['Your_Rating'].notna().sum()} rated movies")
        
    except FileNotFoundError:
        print("❌ Error: ratings.csv not found in data/ directory")
        print("💡 Please export your IMDb ratings to data/ratings.csv first")
        return
    
    # Generate recommendations
    print("\n🎬 Generating personalized movie recommendations...")
    recommendations_df = recommendation_generator.generate_recommendation_list(
        ratings_df, 
        num_recommendations=50  # Adjust as needed
    )
    
    if recommendations_df.empty:
        print("❌ No recommendations could be generated")
        return
    
    print(f"✅ Generated {len(recommendations_df)} potential recommendations")
    
    # Save as movies_to_rate.csv format
    output_file = "data/auto_generated_movies_to_rate.csv"
    recommendation_generator.save_recommendations_for_rating(recommendations_df, output_file)
    
    # Show top recommendations
    print("\n🏆 TOP 10 AUTO-GENERATED RECOMMENDATIONS:")
    print("=" * 60)
    
    for idx, (_, movie) in enumerate(recommendations_df.head(10).iterrows(), 1):
        print(f"\n{idx}. {movie['Title']} ({movie.get('Year', 'N/A')})")
        print(f"   🎭 Genres: {movie.get('Genres', 'N/A')}")
        print(f"   👨‍💼 Director: {movie.get('Director', 'N/A')}")
        print(f"   ⭐ IMDb: {movie.get('IMDb_Rating', 'N/A')}")
        print(f"   ⏱️  Runtime: {movie.get('Runtime_mins', 'N/A')} min")
        print(f"   💫 Match Score: {movie.get('Match_Score', 0):.3f}")
        print(f"   📝 Reason: {movie.get('Why_Recommended', 'N/A')}")
        print("-" * 50)
    
    print(f"\n🎉 Next steps:")
    print(f"1. Review the generated file: {output_file}")
    print(f"2. Run the main recommendation system:")
    print(f"   pipenv run python main.py")
    print(f"3. The system will use your auto-generated recommendations!")
    
    # Save detailed recommendations for reference
    detailed_file = "outputs/recommendations/auto_generated_detailed.csv"
    os.makedirs('outputs/recommendations', exist_ok=True)
    recommendations_df.to_csv(detailed_file, index=False)
    print(f"📁 Detailed recommendations saved to: {detailed_file}")

if __name__ == "__main__":
    main()