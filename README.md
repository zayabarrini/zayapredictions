```movie_recommender/
├── Pipfile
├── config/
│   └── config.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── recommender.py
│   ├── omdb_client.py
│   └── utils.py
├── data/
│   ├── ratings.csv
│   └── movies_to_rate.csv
└── main.py
```

# Basic usage (default files)
python main.py

# Use different ratings file
python main.py --ratings data/my_ratings.csv

# Use different movies file
python main.py --movies data/my_movies.csv

# Use both different files
python main.py --ratings data/action_ratings.csv --movies data/action_movies.csv

# Custom output prefix
python main.py --ratings data/drama_ratings.csv --output-prefix drama_analysis

# Skip plots for faster processing
python main.py --no-plots

# Show more recommendations
python main.py --top-n 20

# Combine options
python main.py --ratings data/comedy_ratings.csv --movies data/comedy_movies.csv --output-prefix comedy --top-n 25 --no-plots


python3 main.py --movies data/Mubi.csv
python3 main.py --movies data/The_Beauty_Of.csv
python3 main.py --movies data/The_Hours_in_Countries.csv
python3 main.py --movies data/Top_rated_IMdb_M-Oscars.csv

python3 main.py --movies data/todo/language/all-languages2.csv

python3 main.py --movies data/todo/language/ru.csv

python3 main.py --movies data/todo/International-Movie-Oscar-Submissions.csv


See also: [Cinema Analysis](https://github.com/zayabarrini/cinema)