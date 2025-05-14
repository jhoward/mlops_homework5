from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import os

# Get absolute path to the data file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "6000_all_categories_questions_with_excerpts.csv")

class Retriever:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Retriever, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if self._initialized:
            return
            
        self.model = SentenceTransformer(model_name)
        self.nn_model = None
        self.df = None
        self.embeddings = None
        # Automatically load data on initialization
        self.load_data()
        self._initialized = True

    def load_data(self, data_path=DATA_PATH):
        """Load and prepare the data"""
        self.df = pd.read_csv(data_path)
        # Encode all excerpts
        self.embeddings = self.model.encode(self.df['wikipedia_excerpt'].tolist(), show_progress_bar=False)
        # Initialize and fit the nearest neighbors model
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn_model.fit(self.embeddings)

    def get_similar_responses(self, question: str, k: int = 5) -> list:
        """
        Retrieve similar responses for a given question
        
        Args:
            question (str): The input question
            k (int): Number of similar responses to return
            
        Returns:
            list: List of dictionaries containing similar responses with their scores
        """
        # Step 1: Convert question to embedding
        query_embedding = self.model.encode([question])
        
        # Step 2 & 3: Compute similarity and get top k results
        distances, indices = self.nn_model.kneighbors(query_embedding, n_neighbors=k)
        
        # Step 4 & 5: Get raw text and format output
        similar_responses = []
        for dist, idx in zip(distances[0], indices[0]):
            response = {
                'prompt': self.df.iloc[idx]['prompt'],
                'excerpt': self.df.iloc[idx]['wikipedia_excerpt'],
                'answer': self.df.iloc[idx]['answer'],
                'similarity_score': float(1 - dist)  # Convert distance to similarity score
            }
            similar_responses.append(response)
            
        return similar_responses
