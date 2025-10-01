import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List
from datetime import datetime

def ensure_directories():
    """Ensure all output directories exist"""
    directories = [
        'outputs',
        'outputs/recommendations', 
        'outputs/plots',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("📁 Output directories created/verified")

def display_recommendations(recommendations_df: pd.DataFrame, top_n: int = 10):
    """Display recommendations in a formatted way"""
    display_cols = ['Title', 'Year', 'Predicted_Rating', 'Recommendation_Score', 
                   'IMDb_Rating', 'Runtime_mins', 'Genres', 'Director']
    
    # Select available columns
    available_cols = [col for col in display_cols if col in recommendations_df.columns]
    
    print("=" * 80)
    print(f"TOP {top_n} RECOMMENDED MOVIES FOR YOU")
    print("=" * 80)
    
    for idx, row in recommendations_df.head(top_n).iterrows():
        print(f"\n🎬 {idx + 1}. {row['Title']} ({row.get('Year', 'N/A')})")
        print(f"   ⭐ Predicted Your Rating: {row['Predicted_Rating']:.1f}/10")
        print(f"   💫 Recommendation Score: {row['Recommendation_Score']:.3f}")
        print(f"   🎭 IMDb Rating: {row.get('IMDb_Rating', 'N/A')}")
        print(f"   ⏱️  Runtime: {row.get('Runtime_mins', 'N/A')} min")
        print(f"   🎭 Genres: {row.get('Genres', 'N/A')}")
        if 'Director' in row and pd.notna(row['Director']):
            print(f"   👨‍💼 Director: {row['Director']}")
        print("-" * 60)

def plot_recommendations(recommendations_df: pd.DataFrame, top_n: int = 10, 
                        save_plots: bool = True, plot_prefix: str = ""):
    """Create visualization of recommendations and save to plots folder"""
    
    # Ensure plots directory exists
    plots_dir = 'outputs/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{plot_prefix}_" if plot_prefix else ""
    
    # Adjust top_n based on available data
    total_movies = len(recommendations_df)
    effective_top_n = min(top_n, total_movies)
    
    print(f"📊 Creating plots for {total_movies} movies (showing top {effective_top_n})")
    
    plt.style.use('default')
    
    # Create subplots based on available data
    if total_movies >= 5:
        # Enough data for 2x2 grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        subplots_created = 4
    elif total_movies >= 2:
        # Few movies - use 2x1 layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        subplots_created = 2
    else:
        # Very few movies - single plot
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        subplots_created = 1
    
    # Top recommended movies
    top_movies = recommendations_df.head(effective_top_n)
    
    # Plot 1: Recommendation scores (horizontal bar chart) - Always create this one
    y_pos = range(len(top_movies))
    
    if len(top_movies) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_movies)))
        bars = ax1.barh(y_pos, top_movies['Recommendation_Score'].values, color=colors)
        ax1.set_yticks(y_pos)
        
        # Adjust label length based on number of movies
        if len(top_movies) <= 10:
            # Show full titles for small datasets
            labels = [movie['Title'] for _, movie in top_movies.iterrows()]
        else:
            # Truncate titles for larger datasets
            labels = [f"{movie['Title'][:20]}..." for _, movie in top_movies.iterrows()]
        
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('Recommendation Score')
        ax1.set_title(f'Top {effective_top_n} Recommended Movies\n(Higher = Better Match for You)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars (only if there's space)
        if len(top_movies) <= 15:
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # Plot 2: Predicted vs IMDb ratings (scatter plot) - Only if we have enough data
    if subplots_created >= 2 and len(top_movies) >= 3:
        scatter = ax2.scatter(top_movies['IMDb_Rating'], top_movies['Predicted_Rating'], 
                             c=top_movies['Recommendation_Score'], cmap='viridis', 
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax2, label='Recommendation Score')
        
        # Add trend line only if we have enough points
        if len(top_movies) > 2:
            z = np.polyfit(top_movies['IMDb_Rating'], top_movies['Predicted_Rating'], 1)
            p = np.poly1d(z)
            ax2.plot(top_movies['IMDb_Rating'], p(top_movies['IMDb_Rating']), "r--", 
                    alpha=0.8, label='Trend line')
            ax2.legend()
        
        # Add movie labels only for small datasets
        if len(top_movies) <= 20:
            for idx, row in top_movies.iterrows():
                ax2.annotate(row['Title'][:12], 
                            (row['IMDb_Rating'], row['Predicted_Rating']),
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.8)
        
        ax2.set_xlabel('IMDb Rating')
        ax2.set_ylabel('Predicted Your Rating')
        ax2.set_title('Your Predicted Rating vs IMDb Rating\n(Dots colored by recommendation score)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    elif subplots_created >= 2:
        # Not enough data for scatter plot - show message
        ax2.text(0.5, 0.5, 'Not enough data for scatter plot\n(Need at least 3 movies)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Predicted vs IMDb Rating', fontsize=14, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # Plot 3: Runtime distribution - Only if we have the data and subplots
    if subplots_created >= 3 and 'Runtime_mins' in recommendations_df.columns:
        runtime_data = recommendations_df['Runtime_mins'].dropna()
        if len(runtime_data) >= 5:  # Need reasonable amount of data for histogram
            ax3.hist(runtime_data, bins=min(10, len(runtime_data)//2), 
                    alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.axvline(runtime_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {runtime_data.mean():.0f} min')
            ax3.set_xlabel('Runtime (mins)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Runtime Distribution', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Not enough runtime data\nfor distribution analysis', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Runtime Distribution', fontsize=14, fontweight='bold')
            ax3.set_xticks([])
            ax3.set_yticks([])
    elif subplots_created >= 3:
        ax3.text(0.5, 0.5, 'Runtime data not available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Runtime Distribution', fontsize=14, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # Plot 4: Year distribution - Only if we have the data and subplots
    if subplots_created >= 4 and 'Year' in recommendations_df.columns:
        year_data = recommendations_df['Year'].dropna()
        if len(year_data) >= 5:  # Need reasonable amount of data for histogram
            ax4.hist(year_data, bins=min(10, len(year_data)//2), 
                    alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(year_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {year_data.mean():.0f}')
            ax4.set_xlabel('Release Year')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Release Year Distribution', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Not enough year data\nfor distribution analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Release Year Distribution', fontsize=14, fontweight='bold')
            ax4.set_xticks([])
            ax4.set_yticks([])
    elif subplots_created >= 4:
        ax4.text(0.5, 0.5, 'Year data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Release Year Distribution', fontsize=14, fontweight='bold')
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    plt.tight_layout()
    
    # Save the plot
    if save_plots:
        plot_filename = f"{plots_dir}/{prefix}recommendation_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Main plot saved to: {plot_filename}")
    
    plt.show()
    
    # Create additional plots only if we have enough data
    if len(recommendations_df) >= 3:
        create_additional_plots(recommendations_df, effective_top_n, timestamp, prefix)
    else:
        print("📊 Skipping additional plots - not enough data")

def create_additional_plots(recommendations_df: pd.DataFrame, top_n: int, 
                          timestamp: str, prefix: str):
    """Create additional specialized plots (only for sufficient data)"""
    plots_dir = 'outputs/plots'
    
    # Only create feature plot if we have at least 5 movies and some features
    if len(recommendations_df) >= 5:
        feature_columns = ['Has_LGBT', 'Has_Cinematography', 'Has_Screenplay', 
                          'Has_Plot_Twist', 'Is_Drama', 'Is_Comedy', 'Is_Crime']
        
        existing_features = [col for col in feature_columns if col in recommendations_df.columns]
        
        if existing_features and recommendations_df[existing_features].sum().sum() > 0:
            plt.figure(figsize=(12, 6))
            feature_counts = recommendations_df[existing_features].sum().sort_values(ascending=True)
            
            # Only show features that actually appear in the data
            feature_counts = feature_counts[feature_counts > 0]
            
            if len(feature_counts) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(feature_counts)))
                bars = plt.barh(range(len(feature_counts)), feature_counts.values, color=colors)
                
                plt.yticks(range(len(feature_counts)), feature_counts.index)
                plt.xlabel('Number of Movies with Feature')
                plt.title(f'Feature Distribution in {len(recommendations_df)} Recommended Movies', 
                         fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                            f'{int(width)}', ha='left', va='center')
                
                plt.tight_layout()
                feature_plot_filename = f"{plots_dir}/{prefix}feature_distribution_{timestamp}.png"
                plt.savefig(feature_plot_filename, dpi=300, bbox_inches='tight')
                print(f"📊 Feature distribution plot saved to: {feature_plot_filename}")
                plt.show()
                return
    
    print("📊 Skipping feature distribution plot - insufficient feature data")

def create_simple_plot_for_small_dataset(recommendations_df: pd.DataFrame, 
                                       timestamp: str, prefix: str):
    """Create a simple combined plot for very small datasets (1-4 movies)"""
    plots_dir = 'outputs/plots'
    
    plt.figure(figsize=(12, 8))
    
    # Simple bar chart showing recommendation scores
    movies = recommendations_df.head(10)  # Show all available
    y_pos = range(len(movies))
    
    bars = plt.barh(y_pos, movies['Recommendation_Score'].values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(movies))))
    
    plt.yticks(y_pos, [movie['Title'] for _, movie in movies.iterrows()])
    plt.xlabel('Recommendation Score')
    plt.title(f'Movie Recommendations\n(Total: {len(movies)} movies)', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add detailed information to each bar
    for i, (_, movie) in enumerate(movies.iterrows()):
        plt.text(movie['Recommendation_Score'] + 0.01, i, 
                f"Pred: {movie['Predicted_Rating']:.1f} | IMDb: {movie.get('IMDb_Rating', 'N/A')}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    simple_plot_filename = f"{plots_dir}/{prefix}simple_recommendations_{timestamp}.png"
    plt.savefig(simple_plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Simple plot for small dataset saved to: {simple_plot_filename}")
    plt.show()

def export_recommendations_to_csv(recommendations_df: pd.DataFrame, 
                                 model_metrics: dict = None,
                                 filename: str = None):
    """Export recommendations to a detailed CSV file in outputs folder"""
    
    # Ensure recommendations directory exists
    recommendations_dir = 'outputs/recommendations'
    os.makedirs(recommendations_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{recommendations_dir}/movie_recommendations_{timestamp}.csv"
    else:
        # Ensure filename is in outputs directory
        if not filename.startswith('outputs/'):
            filename = f"{recommendations_dir}/{filename}"
    
    # Create a copy for export
    export_df = recommendations_df.copy()
    
    # Add ranking
    export_df['Rank'] = range(1, len(export_df) + 1)
    
    # Define desired columns in order of importance
    desired_columns = [
        'Rank', 'Title', 'Year', 'Predicted_Rating', 'Recommendation_Score',
        'IMDb_Rating', 'Runtime_mins', 'Genres', 'Director', 'Country',
        'Plot', 'Has_LGBT', 'Has_Cinematography', 'Has_Screenplay', 
        'Has_Plot_Twist', 'Has_Female_Strength', 'Has_Male_Gaze', 'Has_Oscar',
        'Genre_Count', 'Is_Drama', 'Is_Comedy', 'Is_Crime', 'Is_History', 'Is_Romance'
    ]
    
    # Only include columns that actually exist in the dataframe
    available_columns = [col for col in desired_columns if col in export_df.columns]
    
    print(f"📊 Available columns for CSV export: {available_columns}")
    
    # Reorder dataframe with only available columns
    export_df = export_df[available_columns]
    
    # Round numerical columns for cleaner output
    numerical_columns = ['Predicted_Rating', 'Recommendation_Score', 'IMDb_Rating']
    for col in numerical_columns:
        if col in export_df.columns:
            export_df[col] = export_df[col].round(3)
    
    # Save to CSV
    export_df.to_csv(filename, index=False)
    
    # Create a summary file with model metrics
    if model_metrics:
        summary_filename = filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w') as f:
            f.write("MOVIE RECOMMENDATION SYSTEM - RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total movies analyzed: {len(recommendations_df)}\n")
            f.write(f"Model performance:\n")
            f.write(f"  - Train R²: {model_metrics.get('train_score', 'N/A'):.3f}\n")
            f.write(f"  - Test R²: {model_metrics.get('test_score', 'N/A'):.3f}\n")
            f.write(f"  - MAE: {model_metrics.get('mae', 'N/A'):.3f}\n")
            f.write(f"  - Training samples: {model_metrics.get('training_samples', 'N/A')}\n")
            
            # Top 5 recommendations
            f.write("\nTOP 5 RECOMMENDATIONS:\n")
            for idx, row in recommendations_df.head().iterrows():
                f.write(f"{idx + 1}. {row['Title']} ({row.get('Year', 'N/A')}) - "
                       f"Predicted: {row['Predicted_Rating']:.1f}/10 - "
                       f"Score: {row['Recommendation_Score']:.3f}\n")
    
    print(f"📊 Recommendations exported to: {filename}")
    if model_metrics:
        print(f"📋 Summary exported to: {summary_filename}")
    
    return filename

def create_detailed_analysis_csv(recommendations_df: pd.DataFrame, 
                                original_ratings_df: pd.DataFrame,
                                filename: str = None):
    """Create a more detailed analysis CSV with comparison to your rating patterns"""
    
    # Ensure recommendations directory exists
    recommendations_dir = 'outputs/recommendations'
    os.makedirs(recommendations_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{recommendations_dir}/movie_analysis_{timestamp}.csv"
    else:
        if not filename.startswith('outputs/'):
            filename = f"{recommendations_dir}/{filename}"
    
    # Calculate your rating patterns from original data
    your_stats = {
        'avg_rating': original_ratings_df['Your_Rating'].mean(),
        'avg_runtime': original_ratings_df['Runtime_mins'].mean(),
        'avg_imdb_rating': original_ratings_df['IMDb_Rating'].mean(),
        'common_genres': original_ratings_df['Genres'].value_counts().head(5).to_dict()
    }
    
    # Create analysis dataframe
    analysis_df = recommendations_df.copy()
    
    # Add comparison metrics (only if the required columns exist)
    if 'Runtime_mins' in analysis_df.columns:
        analysis_df['Runtime_Vs_Your_Avg'] = analysis_df['Runtime_mins'] - your_stats['avg_runtime']
        analysis_df['Good_Runtime_Match'] = abs(analysis_df['Runtime_Vs_Your_Avg']) <= 30
    
    if 'IMDb_Rating' in analysis_df.columns:
        analysis_df['IMDb_Vs_Your_Avg'] = analysis_df['IMDb_Rating'] - your_stats['avg_imdb_rating']
        analysis_df['Good_IMDb_Match'] = abs(analysis_df['IMDb_Vs_Your_Avg']) <= 1.0
    
    if 'Predicted_Rating' in analysis_df.columns:
        analysis_df['Predicted_Vs_Your_Avg'] = analysis_df['Predicted_Rating'] - your_stats['avg_rating']
    
    if 'Recommendation_Score' in analysis_df.columns:
        analysis_df['High_Confidence'] = analysis_df['Recommendation_Score'] >= 0.6
    
    # Add ranking
    analysis_df['Rank'] = range(1, len(analysis_df) + 1)
    
    # Define desired columns for analysis
    desired_analysis_columns = [
        'Rank', 'Title', 'Year', 'Predicted_Rating', 'Recommendation_Score',
        'IMDb_Rating', 'Runtime_mins', 'Genres', 'Director',
        'Runtime_Vs_Your_Avg', 'IMDb_Vs_Your_Avg', 'Predicted_Vs_Your_Avg',
        'Good_Runtime_Match', 'Good_IMDb_Match', 'High_Confidence',
        'Has_LGBT', 'Has_Cinematography', 'Has_Screenplay', 'Has_Plot_Twist'
    ]
    
    # Only include columns that actually exist
    available_analysis_columns = [col for col in desired_analysis_columns if col in analysis_df.columns]
    
    print(f"📈 Available columns for analysis CSV: {available_analysis_columns}")
    
    analysis_df = analysis_df[available_analysis_columns]
    
    # Round numerical columns
    numerical_columns = ['Predicted_Rating', 'Recommendation_Score', 'IMDb_Rating', 
                        'Runtime_Vs_Your_Avg', 'IMDb_Vs_Your_Avg', 'Predicted_Vs_Your_Avg']
    for col in numerical_columns:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].round(3)
    
    analysis_df.to_csv(filename, index=False)
    print(f"📈 Detailed analysis exported to: {filename}")
    
    return filename