# Set your OMDB API key first
export OMDB_API_KEY="your_api_key_here"

# Search for top movies from your actress list
python movie_searcher.py --actresses actresses.csv --min-rating 7.5 --min-votes 10000 --top-n 10 --output top_actress_movies.csv

# Search for a specific actor
python movie_searcher.py --actor "Tom Hanks" --min-rating 8.0 --top-n 5

# Search by director
python movie_searcher.py --director "Christopher Nolan" --min-rating 8.0 --top-n 10

# Search by genre
python movie_searcher.py --genre "Sci-Fi" --min-rating 7.5 --min-votes 5000 --top-n 15

# Combine criteria
python movie_searcher.py --actresses actresses.csv --genre "Drama" --min-rating 7.0 --top-n 20


Features
Customizable search: Actors, directors, genres, or combinations

Quality filtering: Minimum rating and vote thresholds

Duplicate removal: Automatically removes duplicate movies

Rate limiting: Respects API limits

CSV export: Save results for later analysis

Detailed output: Shows comprehensive movie information