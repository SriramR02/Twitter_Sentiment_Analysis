# Twitter Sentiment Analysis

 

This project performs sentiment analysis on tweets using **Natural Language Processing (NLP)** and **Machine Learning (ML)** models. The application is built with **Streamlit** for the frontend and **FastAPI** for the backend.

---

## Table of Contents
- Introduction
- Features
- Installation
- Usage
- API Endpoints
- Technologies Used
- Contributing
- License

---

## Introduction
Twitter Sentiment Analysis is a project that analyzes the sentiment of tweets. It classifies tweets into positive, negative, or neutral sentiments using NLP techniques and ML models. The project includes a web interface built with Streamlit and a backend API built with FastAPI.

---

## Features
- Analyze sentiment of tweets in real-time
- Visualize sentiment distribution
- Interactive web interface
- RESTful API for sentiment analysis

---

## Installation

1. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage
1. Start the FastAPI backend server:
    ```bash
    uvicorn app.main:app --reload
    ```

2. Start the Streamlit frontend:
    ```bash
    streamlit run app/app.py
    ```

3. Open your browser and go to `http://localhost:8501` to access the Streamlit interface.

---

## API Endpoints
- `POST /predict`: Analyze the sentiment of a tweet.
    - Request body:
        ```json
        {
            "tweet": "Your tweet text here"
        }
        ```
    - Response:
        ```json
        {
            "sentiment": "positive"  or "neutral"  or "negative"
        }
        ```

---

## Technologies Used
- **NLP**: Natural Language Processing techniques for text analysis
- **ML**: Machine Learning models for sentiment classification
- **Streamlit**: Web framework for the frontend
- **FastAPI**: Web framework for the backend API

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
