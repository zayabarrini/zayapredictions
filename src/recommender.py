import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from typing import Tuple, Dict, List
from config.config import Config

class MovieRecommender:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=Config.RANDOM_STATE,
            max_depth=10
        )
        self.is_trained = False
        self.feature_columns = [
            'IMDb_Rating', 'Runtime_mins', 'Year', 'Num_Votes',
            'Has_LGBT', 'Has_Cinematography', 'Has_Screenplay', 'Has_Plot_Twist',
            'Genre_Count', 'Is_Drama', 'Is_Comedy', 'Is_Crime', 'Is_History', 
            'Is_Romance', 'Is_Action', 'Is_Adventure', 'Is_Sci_Fi'
        ]
        self.actual_training_features = []  # Track features used in training
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict:
        """Train the recommendation model with specific features"""
        if len(X) == 0:
            raise ValueError("No training data available")
            
        # Store the actual features used for training
        if feature_names:
            self.actual_training_features = feature_names
        else:
            # If no feature names provided, use indices
            self.actual_training_features = [f"feature_{i}" for i in range(X.shape[1])]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test) if len(X_test) > 0 else 0
        y_pred = self.model.predict(X_test) if len(X_test) > 0 else []
        mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else 0
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'mae': mae,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X.shape[1],
            'features_used': self.actual_training_features
        }
    
    def predict_ratings(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Predict ratings for new movies with feature alignment"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print(f"🔧 Model was trained with {len(self.actual_training_features)} features: {self.actual_training_features}")
        
        # Prepare features - only use the exact features that were used in training
        available_features = [col for col in self.actual_training_features if col in movies_df.columns]
        missing_features = [col for col in self.actual_training_features if col not in movies_df.columns]
        
        print(f"✅ Available features for prediction: {len(available_features)}")
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            print("   These features will be set to 0")
        
        # Fill missing values and ensure all training features exist
        for feature in self.actual_training_features:
            if feature not in movies_df.columns:
                movies_df[feature] = 0  # Default value for missing features
            else:
                # Fill NaN values
                if movies_df[feature].dtype in ['int64', 'float64']:
                    movies_df[feature] = movies_df[feature].fillna(0)
                else:
                    movies_df[feature] = movies_df[feature].fillna('')
        
        # Use exactly the same features as training, in the same order
        X_new = movies_df[self.actual_training_features].values
        
        print(f"🎯 Making predictions with {X_new.shape[1]} features")
        
        # Make predictions
        predictions = self.model.predict(X_new)
        
        # Create results dataframe
        results_df = movies_df.copy()
        results_df['Predicted_Rating'] = predictions
        results_df['Recommendation_Score'] = self._calculate_recommendation_score(results_df)
        
        return results_df.sort_values('Recommendation_Score', ascending=False)
    
    def _calculate_recommendation_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate recommendation score based on multiple factors"""
        # Normalize predicted rating (0-10 scale)
        rating_score = df['Predicted_Rating'] / 10.0
        
        # Consider IMDb rating (weighted less)
        imdb_score = df['IMDb_Rating'] / 10.0 if 'IMDb_Rating' in df.columns else 0.5
        
        # Consider number of votes (popularity)
        if 'Num_Votes' in df.columns and df['Num_Votes'].max() > 0:
            vote_score = np.log1p(df['Num_Votes']) / np.log1p(df['Num_Votes'].max())
        else:
            vote_score = 0.5
        
        # Boost for your preferred tags
        tag_boost = 0
        tag_weights = {
            'Has_LGBT': 0.15,
            'Has_Cinematography': 0.12,
            'Has_Screenplay': 0.12,
            'Has_Plot_Twist': 0.08,
            'Is_Drama': 0.10,
            'Is_History': 0.08
        }
        
        for tag, weight in tag_weights.items():
            if tag in df.columns:
                tag_boost += df[tag] * weight
        
        # Penalty for male gaze content if you prefer female-positive narratives
        penalty = 0
        if 'Has_Male_Gaze' in df.columns:
            penalty = df['Has_Male_Gaze'] * 0.1
        
        # Combined score (you can adjust weights)
        combined_score = (
            0.5 * rating_score +    # Your predicted preference
            0.2 * imdb_score +     # General quality
            0.1 * vote_score +     # Popularity
            0.3 * tag_boost -      # Your specific interests
            0.1 * penalty          # Content you might dislike
        )
        
        # Ensure scores are within reasonable bounds
        combined_score = np.clip(combined_score, 0, 1)
        
        return combined_score
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.actual_training_features,
            'importance': self.model.feature_importances_
        })
        
        return importance_df.sort_values('importance', ascending=False)