#!/usr/bin/env python3
"""
Job Classification Machine Learning Models
==========================================

In this module i am implementing machine learning models for job classification:
- Job category prediction (Data Engineer, Data Scientist, etc.)
- Seniority level prediction (Entry, Mid, Senior)
- Salary prediction using regression

Now i am using standard ML libraries (scikit-learn, pandas) with original implementations.
All model architectures and training logic are original work for this project.

References:
- scikit-learn Documentation: https://scikit-learn.org/stable/
- pandas Documentation: https://pandas.pydata.org/docs/
- Nicholas Renotte - Machine Learning with Python: https://www.youtube.com/c/NicholasRenotte
- spaCy Documentation: https://spacy.io/
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import pickle

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

# NLP Libraries
import re
from collections import Counter
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobClassifier:
    """
    Now i am creating Machine Learning models for job classification and analysis
    Now i am handling job categorization, seniority prediction, and salary estimation
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Now i am initializing NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Now i am initializing models
        self.job_category_model = None
        self.seniority_model = None
        self.employment_type_model = None
        self.salary_model = None
        
        # Now i am creating feature engineering components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoders = {}
        self.scalers = {}
        
        # Now i am creating job categories mapping
        self.job_categories = {
            'data_engineer': ['data engineer', 'data engineering', 'etl', 'pipeline', 'warehouse'],
            'data_analyst': ['data analyst', 'analytics', 'reporting', 'dashboard', 'bi'],
            'data_scientist': ['data scientist', 'machine learning', 'ml', 'ai', 'statistics'],
            'software_engineer': ['software engineer', 'developer', 'programming', 'coding'],
            'business_analyst': ['business analyst', 'ba', 'requirements', 'process'],
            'product_manager': ['product manager', 'pm', 'product owner', 'strategy'],
            'project_manager': ['project manager', 'scrum master', 'agile', 'delivery']
        }
        
        # Now i am creating seniority levels
        self.seniority_levels = ['entry level', 'associate', 'mid-senior level', 'senior', 'director', 'executive']
        
        # Now i am creating employment types
        self.employment_types = ['full-time', 'part-time', 'contract', 'temporary', 'internship', 'co-op']
    
    def load_job_data(self, data_file: str = None) -> pd.DataFrame:
        """Now i am loading job data from JSON files"""
        try:
            if data_file:
                data_path = self.data_dir / data_file
            else:
                # Now i am finding the most recent job data file
                json_files = list(self.data_dir.glob("all_jobs_final_*.json"))
                if not json_files:
                    json_files = list(self.data_dir.glob("extracted_text_*.json"))
                
                if not json_files:
                    raise FileNotFoundError("No job data files found")
                
                data_path = max(json_files, key=lambda x: x.stat().st_mtime)
            
            logger.info(f"Loading job data from: {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Now i am converting to DataFrame
            jobs_list = []
            for job_key, job_data in data.items():
                job_data['job_id'] = job_key
                jobs_list.append(job_data)
            
            df = pd.DataFrame(jobs_list)
            logger.info(f"Loaded {len(df)} jobs")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading job data: {e}")
            return pd.DataFrame()
    
    def preprocess_text(self, text: str) -> str:
        """Now i am preprocessing job text for ML features"""
        if not text or pd.isna(text):
            return ""
        
        # Now i am converting to lowercase
        text = text.lower()
        
        # Now i am removing special characters but keep important ones
        text = re.sub(r'[^\w\s\$\-\.\,\:\;\(\)\[\]\/]', ' ', text)
        
        # Now i am removing extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Now i am removing common stop words and noise
        noise_words = ['job', 'position', 'role', 'hiring', 'company', 'inc', 'corp', 'llc']
        words = text.split()
        words = [word for word in words if word not in noise_words and len(word) > 2]
        
        return ' '.join(words)
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Now i am extracting and engineering features from job data"""
        logger.info("Extracting features from job data...")
        
        # Now i am creating a copy for feature engineering
        features_df = df.copy()
        
        # Text Features (Now i am extracting and engineering features from job data)
        features_df['role_processed'] = features_df['role'].apply(self.preprocess_text)
        features_df['company_processed'] = features_df['company'].apply(self.preprocess_text)
        features_df['location_processed'] = features_df['location'].apply(self.preprocess_text)
        
        # Now i am combining all text for TF-IDF
        features_df['combined_text'] = (
            features_df['role_processed'] + ' ' +
            features_df['company_processed'] + ' ' +
            features_df['location_processed'] + ' ' +
            features_df['Job_Function'].fillna('') + ' ' +
            features_df['Industries'].fillna('')
        )
        
        # Categorical Features (Now i am extracting and engineering features from job data)
        # Now i am improving seniority classification
        features_df['seniority_encoded'] = features_df['Seniority_level'].fillna('unknown')
        
        # Now i am applying rule-based seniority classification for unknown cases
        def classify_seniority_from_text(text):
            if pd.isna(text) or text == 'unknown':
                return 'unknown'
            text_lower = str(text).lower()
            
            # Now i am creating senior level indicators
            senior_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'manager', 'director', 'sr.', 'sr ']
            if any(keyword in text_lower for keyword in senior_keywords):
                return 'senior'
            
            # Now i am creating entry level indicators
            entry_keywords = ['junior', 'entry', 'associate', 'trainee', 'intern', 'graduate', 'jr.', 'jr ']
            if any(keyword in text_lower for keyword in entry_keywords):
                return 'entry'
            
            # Mid-l evel indicators
            mid_keywords = ['mid', 'intermediate', 'level 2', 'level ii']
            if any(keyword in text_lower for keyword in mid_keywords):
                return 'mid-senior'
            
            return 'unknown'
        
        # Now i am applying improved seniority classification
        features_df['seniority_encoded'] = features_df['combined_text'].apply(classify_seniority_from_text)
        features_df['employment_encoded'] = features_df['Employment_type'].fillna('unknown')
        features_df['function_encoded'] = features_df['Job_Function'].fillna('unknown')
        features_df['industry_encoded'] = features_df['Industries'].fillna('unknown')
        
        # Location Features (Now i am extracting and engineering features from job data)
        features_df['is_remote'] = features_df['location'].str.contains('remote', case=False, na=False).astype(int)
        features_df['is_hybrid'] = features_df['location'].str.contains('hybrid', case=False, na=False).astype(int)
        
        # Pay Features (if available) (Now i am extracting and engineering features from job data)
        features_df['has_pay_info'] = features_df['pay_range'].notna().astype(int)
        
        # Now i am extracting numeric pay values for salary prediction
        features_df['min_salary'] = features_df['pay_range'].apply(self.extract_min_salary)
        features_df['max_salary'] = features_df['pay_range'].apply(self.extract_max_salary)
        features_df['avg_salary'] = (features_df['min_salary'] + features_df['max_salary']) / 2
        
        # Text Length Features (Now i am extracting and engineering features from job data)
        features_df['role_length'] = features_df['role'].str.len().fillna(0)
        features_df['company_length'] = features_df['company'].str.len().fillna(0)
        
        logger.info(f"Feature extraction complete. Shape: {features_df.shape}")
        return features_df
    
    def extract_min_salary(self, pay_range: str) -> float:
        """Extract minimum salary from pay range string"""
        if not pay_range or pd.isna(pay_range):
            return np.nan
        
        # Now i am extracting numbers from salary range
        numbers = re.findall(r'\$?(\d{1,3}(?:,\d{3})*)', pay_range)
        if numbers:
            # Now i am converting to integers (remove commas)
            salaries = [int(num.replace(',', '')) for num in numbers]
            return min(salaries)
        return np.nan
    
    def extract_max_salary(self, pay_range: str) -> float:
        """Now i am extracting maximum salary from pay range string"""
        if not pay_range or pd.isna(pay_range):
            return np.nan
        
        # Now i am extracting numbers from salary range
        numbers = re.findall(r'\$?(\d{1,3}(?:,\d{3})*)', pay_range)
        if numbers:
            # Now i am converting to integers (remove commas)
            salaries = [int(num.replace(',', '')) for num in numbers]
            return max(salaries)
        return np.nan
    
    def classify_job_category(self, text: str) -> str:
        """Now i am classifying job into predefined categories based on text"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        # Now i am scoring each category
        category_scores = {}
        for category, keywords in self.job_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        # Return category with highest score (Now i am classifying job into predefined categories based on text)
        if category_scores and max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        return 'unknown'
    
    def train_job_category_model(self, df: pd.DataFrame) -> None:
        """Now i am training model to classify job categories"""
        logger.info("Training job category classification model...")
        
        # Now i am preparing features
        X_text = df['combined_text'].fillna('')
        y_category = df['role'].apply(self.classify_job_category)
        
        # Now i am splitting data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_category, test_size=0.2, random_state=42, stratify=y_category
        )
        
        # Now i am creating pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Now i am training model
        pipeline.fit(X_train, y_train)
        
        # Now i am evaluating model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Job category model accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Now i am saving model
        self.job_category_model = pipeline
        joblib.dump(pipeline, self.models_dir / 'job_category_model.pkl')
        
        logger.info("Job category model saved successfully")
    
    def train_seniority_model(self, df: pd.DataFrame) -> None:
        """Now i am training model to predict seniority levels"""
        logger.info("Training seniority level prediction model...")
        
        # Now i am preparing features
        X_text = df['combined_text'].fillna('')
        y_seniority = df['seniority_encoded']
        
        # Now i am filtering out unknown seniority levels
        mask = y_seniority != 'unknown'
        X_text = X_text[mask]
        y_seniority = y_seniority[mask]
        
        if len(X_text) == 0:
            logger.warning("No seniority data available for training")
            return
        
        # If we have very few samples, now i am using a simple rule-based approach
        if len(X_text) < 5:
            logger.warning("Very few seniority samples available - using rule-based approach")
            self.seniority_model = "rule_based_few_samples"
            return
        
        # Now i am checking if we have enough samples for stratified split
        min_samples_per_class = 2
        class_counts = y_seniority.value_counts()
        can_stratify = all(count >= min_samples_per_class for count in class_counts)
        
        if can_stratify and len(y_seniority) >= 10:
            # Now i am using stratified split for balanced classes
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y_seniority, test_size=0.2, random_state=42, stratify=y_seniority
            )
        else:
            # Now i am using simple random split for imbalanced or small datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y_seniority, test_size=0.2, random_state=42
            )
        
        # Now i am creating pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Now i am training model
        pipeline.fit(X_train, y_train)
        
        # Now i am evaluating model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Seniority model accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Now i am saving model
        self.seniority_model = pipeline
        joblib.dump(pipeline, self.models_dir / 'seniority_model.pkl')
        
        logger.info("Seniority model saved successfully")
    
    def train_salary_model(self, df: pd.DataFrame) -> None:
        """Now i am training model to predict salary ranges"""
        logger.info("Training salary prediction model...")
        
        # Now i am filtering data with salary information
        salary_df = df.dropna(subset=['avg_salary'])
        
        if len(salary_df) == 0:
            logger.warning("No salary data available for training")
            return
        
        # Now i am preparing features
        X_text = salary_df['combined_text'].fillna('')
        y_salary = salary_df['avg_salary']
        
        # Now i am splitting data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_salary, test_size=0.2, random_state=42
        )
        
        # Now i am creating pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
            ('regressor', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        # Now i am training model
        pipeline.fit(X_train, y_train)
        
        # Now i am evaluating model
        y_pred = pipeline.predict(X_test)
        
        # Now i am calculating metrics
        mae = np.mean(np.abs(y_pred - y_test))
        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        
        logger.info(f"Salary model MAE: ${mae:,.0f}")
        logger.info(f"Salary model RMSE: ${rmse:,.0f}")
        
        # Now i am saving model
        self.salary_model = pipeline
        joblib.dump(pipeline, self.models_dir / 'salary_model.pkl')
        
        logger.info("Salary model saved successfully")
    
    def train_all_models(self, data_file: str = None) -> None:
        """Now i am training all ML models"""
        logger.info("Starting ML model training pipeline...")
        
        # Now i am loading data
        df = self.load_job_data(data_file)
        if df.empty:
            logger.error("No data available for training")
            return
        
        # Now i am extracting features
        features_df = self.extract_features(df)
        
        # Now i am training models
        self.train_job_category_model(features_df)
        self.train_seniority_model(features_df)
        self.train_salary_model(features_df)
        
        logger.info("All models trained successfully!")
    
    def predict_job_category(self, job_text: str) -> str:
        """Now i am predicting job category for new job"""
        if not self.job_category_model:
            return "Model not trained"
        
        # Now i am handling rule-based models
        if isinstance(self.job_category_model, str):
            if self.job_category_model == "rule_based_small_dataset":
                return self._rule_based_job_category_prediction(job_text)
            else:
                return "Model not trained"
        
        prediction = self.job_category_model.predict([job_text])
        return prediction[0]
    
    def predict_seniority(self, job_text: str) -> str:
        """Now i am predicting seniority level for new job"""
        if not self.seniority_model:
            return "Model not trained"
        
        # Now i am handling rule-based models
        if isinstance(self.seniority_model, str):
            if self.seniority_model == "rule_based_small_dataset":
                return self._rule_based_seniority_prediction(job_text)
            else:
                return "Model not trained"
        
        prediction = self.seniority_model.predict([job_text])
        return prediction[0]
    
    def _rule_based_job_category_prediction(self, job_text: str) -> str:
        """Now i am rule-based job category prediction for small datasets"""
        job_text_lower = job_text.lower()
        
        # Now i am checking each category
        for category, keywords in self.job_categories.items():
            if any(keyword in job_text_lower for keyword in keywords):
                return category.replace('_', ' ').title()
        
        # Now i am defaulting to data engineer if no match found
        return "Data Engineer"
    
    def _rule_based_seniority_prediction(self, job_text: str) -> str:
        """Now i am rule-based seniority prediction for small datasets"""
        job_text_lower = job_text.lower()
        
        # Now i am creating senior level indicators
        senior_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'manager', 'director']
        if any(keyword in job_text_lower for keyword in senior_keywords):
            return 'senior'
        
        # Now i am creating entry level indicators  
        entry_keywords = ['junior', 'entry', 'associate', 'trainee', 'intern', 'graduate']
        if any(keyword in job_text_lower for keyword in entry_keywords):
            return 'entry'
        
        # Now i am creating mid-level indicators
        mid_keywords = ['mid', 'intermediate', 'experienced', 'specialist']
        if any(keyword in job_text_lower for keyword in mid_keywords):
            return 'mid-level'
        
        # Now i am defaulting to mid-level for unknown cases
        return 'mid-level'
    
    def predict_salary(self, job_text: str) -> float:
        """Predict salary for new job"""
        if not self.salary_model:
            return np.nan
        
        # Now i am handling rule-based models
        if isinstance(self.salary_model, str):
            if self.salary_model == "rule_based_small_dataset":
                return self._rule_based_salary_prediction(job_text)
            else:
                return np.nan
        
        prediction = self.salary_model.predict([job_text])
        return prediction[0]
    
    def _rule_based_salary_prediction(self, job_text: str) -> float:
        """Now i am rule-based salary prediction for small datasets"""
        job_text_lower = job_text.lower()
        
        # Now i am creating senior level indicators - higher salary
        senior_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'manager', 'director']
        if any(keyword in job_text_lower for keyword in senior_keywords):
            return 120000.0  # Senior level salary
        
        # Now i am creating entry level indicators - lower salary
        entry_keywords = ['junior', 'entry', 'associate', 'trainee', 'intern', 'graduate']
        if any(keyword in job_text_lower for keyword in entry_keywords):
            return 70000.0  # Entry level salary
        
        # Now i am creating data engineer specific - mid-level salary
        if 'data engineer' in job_text_lower:
            return 95000.0
        
        # Now i am defaulting to mid-level salary
        return 85000.0

def main():
    """Main function to train ML models"""
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 50)
    print("Job Classification and Analysis")
    print()
    
    # Now i am initializing classifier
    classifier = JobClassifier()
    
    # Now i am training all models
    classifier.train_all_models()
    
    print("\n ML Model Training Complete!")
    print("Models saved to: models/")
    print("\nNext steps:")
    print("1. Evaluate model performance")
    print("2. Integrate with job application system")
    print("3. Set up automated predictions")

if __name__ == "__main__":
    main()
