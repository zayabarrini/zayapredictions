```
movie_recommender/
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

```
international-payments/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── constants.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── payment_processor.py
│   │   ├── security_manager.py
│   │   ├── models.py
│   │   └── exceptions.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── payment_agent.py
│   │   ├── fraud_detector.py
│   │   └── route_optimizer.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── exchange_service.py
│   │   ├── compliance_service.py
│   │   ├── analytics_service.py
│   │   └── notification_service.py
│   ├── regional/
│   │   ├── __init__.py
│   │   ├── compliance_rules.py
│   │   ├── currency_rules.py
│   │   └── localization.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── middleware.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── validators.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── api.md
│   ├── security.md
│   └── deployment.md
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── scripts/
│   ├── deploy.sh
│   ├── migrate.py
│   └── health_check.py
└── configs/
    ├── dev.yaml
    ├── staging.yaml
    └── prod.yaml
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