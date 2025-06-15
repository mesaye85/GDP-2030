# GDP-2030

Simple Flask application that predicts GDP using a trained model. The project manages dependencies using Poetry.

## Setup

1. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
2. (Optional) Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Running the App

Start the Flask server:
```bash
python main.py
```

The server listens on `0.0.0.0` using the `PORT` environment variable if set, otherwise port `5001`.
