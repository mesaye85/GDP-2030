# GDP Prediction 2030

This project aims to predict GDP values for various countries in 2030 using machine learning techniques.

## Setup

1. Make sure you have Python 3.10 or higher installed
2. Install Poetry (dependency management tool)
3. Install dependencies:
   ```bash
   poetry install
   ```

## Running the Application

1. Activate the virtual environment:
   ```bash
   poetry shell
   ```

2. Run the Flask application:
   ```bash
   python main.py
   ```

## Project Structure

- `main.py`: Main Flask application
- `python.py`: Core prediction logic and data processing
- `models/`: Directory containing trained models
- `data/`: Directory containing datasets

## Dependencies

- Flask: Web framework
- Pandas: Data manipulation
- Scikit-learn: Machine learning
- TensorFlow: Deep learning
- Matplotlib: Data visualization

## Development

To run tests:
```bash
poetry run pytest
```

To format code:
```bash
poetry run black .
```

To check code style:
```bash
poetry run flake8
```