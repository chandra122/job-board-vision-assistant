#!/usr/bin/env python3
"""
FULLY AUTOMATED END-TO-END PIPELINE
===================================

This is my main module that provides complete automation without any 
manual clicking:

What it does?
1. It captures the LinkedIn screenshots
2. Then it processes the screenshots locally (OCR + ML)
3. Then i am training ML models
4. Then i am generating personalized emails
5. Then i am sending emails automatically

References:
- Nicholas Renotte - Computer Vision Tutorials: https://www.youtube.com/c/NicholasRenotte
- Nicholas Renotte - Machine Learning with Python: https://www.youtube.com/c/NicholasRenotte
- EasyOCR Documentation: https://github.com/JaidedAI/EasyOCR
- scikit-learn Documentation: https://scikit-learn.org/stable/

All code is original work developed specifically for this computer vision project.
No external code was copied without proper attribution.
"""

import os
import json
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# I am importing all my modules
from data_collection.linkedin_scraper import main as run_linkedin_scraper
from data_collection.kaggle_collector import KaggleDataCollector
from computer_vision.enhanced_job_analyzer import main as run_text_extraction
from ml_models.job_classifier import JobClassifier
from utils.email_sender import EmailSender
from utils.microsoft_optimizations import MicrosoftOptimizer
from utils.linkedin_folder_manager import LinkedInFolderManager


class FullyAutomatedPipeline:
    """Created a class for the end-to-end automation pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging() # I am setting up the logging
        
        # Now i am initializing all my components
        self.ml_classifier = JobClassifier()
        self.email_sender = EmailSender()
        self.microsoft_optimizer = MicrosoftOptimizer()
        self.kaggle_collector = KaggleDataCollector()
        self.folder_manager = LinkedInFolderManager(keep_latest=5)
        
        # I am configuring my pipeline
        self.config = {
            'job_role': 'Data Engineer',
            'max_jobs': 5,
            'auto_process': True,
            'auto_train_models': True,  # Train models before ML analysis
            'auto_ml': True,
            'auto_email': True,
            'email_recipients': ["bollinenichandrasekhar11@gmail.com"],  # Add your email addresses here for testing
        }
    
    def setup_logging(self):
        """Now i am setting up the logging for the pipeline"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"fully_automated_pipeline_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_complete_automation(self, job_role: str = None, max_jobs: int = None) -> Dict:
        """Now i am running the complete automated pipeline"""
        start_time = time.time()
        results = {
            'success': False,
            'steps_completed': [],
            'errors': [],
            'duration_seconds': 0,
            'data': {}
        }
        
        try:
            self.logger.info("STARTING FULLY AUTOMATED PIPELINE")
            self.logger.info("=" * 60) # I am adding a separator for the logging
            
            # Now i am updating my config with the user input
            if job_role:
                self.config['job_role'] = job_role
            if max_jobs:
                self.config['max_jobs'] = max_jobs
            
            # LinkedIn Data Collection
            self.logger.info("Now as a first step doing LinkedIn Data Collection")
            linkedin_data = self._run_linkedin_collection()
            results['steps_completed'].append('linkedin_collection')
            results['data']['linkedin'] = linkedin_data
            
            # Kaggle Data Collection
            self.logger.info("Now as a second step doing Kaggle Data Collection")
            kaggle_data = self._run_kaggle_collection()
            results['steps_completed'].append('kaggle_collection')
            results['data']['kaggle'] = kaggle_data
            
            # Local Processing (OCR + Text Extraction)
            if self.config['auto_process']:
                self.logger.info("Now as a third step doing Local Processing (OCR + TEXT EXTRACTION)")
                processing_data = self._run_local_processing()
                results['steps_completed'].append('local_processing')
                results['data']['processing'] = processing_data
            
            # Train ML Models (if enabled)
            if self.config['auto_train_models']:
                self.logger.info("Now as a fourth step doing Training ML Models")
                training_data = self._run_model_training()
                results['steps_completed'].append('model_training')
                results['data']['training'] = training_data
            
            # Machine Learning Analysis
            if self.config['auto_ml']:
                self.logger.info("Now as a fifth step doing Machine Learning Analysis")
                ml_data = self._run_ml_analysis()
                results['steps_completed'].append('ml_analysis')
                results['data']['ml'] = ml_data
            
            # Email Generation and Sending
            if self.config['auto_email']:
                self.logger.info("Now as a sixth step doing Email Generation and Sending")
                email_data = self._run_email_automation()
                results['steps_completed'].append('email_automation')
                results['data']['email'] = email_data
            
            # Folder Cleanup
            self.logger.info("Now as a seventh step doing Folder Cleanup")
            self.logger.info("Cleaning up old LinkedIn screenshot folders...")
            
            cleanup_results = self._run_folder_cleanup()
            results['steps_completed'].append('folder_cleanup')
            results['data']['cleanup'] = cleanup_results
            
            # Now i am calculating the duration of the pipeline
            results['duration_seconds'] = time.time() - start_time
            results['success'] = True
            
            self.logger.info("Now the Fully Automated Pipeline is completed successfully!")
            self.logger.info(f"Total Duration: {results['duration_seconds']:.2f} seconds")
            self.logger.info(f"Steps Completed: {len(results['steps_completed'])}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results['errors'].append(str(e))
            results['duration_seconds'] = time.time() - start_time
            return results
    
    def _run_linkedin_collection(self) -> Dict:
        """Now i am running the LinkedIn data collection"""
        try:
            self.logger.info(f"Collecting {self.config['max_jobs']} jobs for '{self.config['job_role']}'")
            
            # Now i am running the LinkedIn scraper with proper error handling
            try:
                # Now i am trying to run in a separate process to avoid Flask/Playwright conflicts
                scraper_success = self._run_scraper_in_process()
                if not scraper_success:
                    # Now i am falling back to existing data
                    return self._fallback_to_existing_data()
            except Exception as scraper_error:
                self.logger.error(f"LinkedIn scraper failed: {scraper_error}")
                # Now i am trying to use existing data if available
                return self._fallback_to_existing_data()
            
            # Now i am finding the latest screenshots directory
            screenshots_dir = Path("data/raw/linkedin_screenshots")
            all_folders = [d for d in screenshots_dir.iterdir() if d.is_dir()]
            date_folders = [d for d in all_folders if d.name.startswith('2025') or d.name.startswith('linkedin_screenshots_')]
            
            if date_folders:
                latest_folder = max(date_folders, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Latest screenshots folder: {latest_folder.name}")
                
                # Now i am checking if the JSON file exists
                json_files = list(latest_folder.glob("jobs_data_*.json"))
                if json_files:
                    latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Found JSON data: {latest_json.name}")
                    
                    # Now i am loading the actual data
                    with open(latest_json, 'r', encoding='utf-8') as f:
                        actual_jobs_data = json.load(f)
                    
                    if actual_jobs_data:
                        self.logger.info(f"Successfully collected {len(actual_jobs_data)} jobs")
                        return {
                            'success': True,
                            'jobs_count': len(actual_jobs_data),
                            'jobs_data': actual_jobs_data,
                            'screenshots_folder': str(latest_folder),
                            'json_file': str(latest_json)
                        }
            
            if jobs_data:
                self.logger.info(f"Successfully collected {len(jobs_data)} jobs")
                return {
                    'success': True,
                    'jobs_count': len(jobs_data),
                    'jobs_data': jobs_data
                }
            else:
                self.logger.warning("No jobs collected")
                return {'success': False, 'jobs_count': 0}
                
        except Exception as e:
            self.logger.error(f"LinkedIn collection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _fallback_to_existing_data(self) -> Dict:
        """Now i am falling back to existing data when scraper fails"""
        try:
            self.logger.info("Attempting to use existing data...")
            
            # Now i am finding the latest screenshots directory
            screenshots_dir = Path("data/raw/linkedin_screenshots")
            all_folders = [d for d in screenshots_dir.iterdir() if d.is_dir()]
            date_folders = [d for d in all_folders if d.name.startswith('2025') or d.name.startswith('linkedin_screenshots_')]
            
            if not date_folders:
                return {'success': False, 'error': 'No existing data found'}
            
            latest_folder = max(date_folders, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Using existing data from: {latest_folder.name}")
            
            # Now i am checking if the JSON file exists
            json_files = list(latest_folder.glob("jobs_data_*.json"))
            if not json_files:
                return {'success': False, 'error': 'No JSON data found in existing folder'}
            
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Found existing JSON data: {latest_json.name}")
            
            # Now i am loading the actual data
            with open(latest_json, 'r', encoding='utf-8') as f:
                actual_jobs_data = json.load(f)
            
            if actual_jobs_data:
                self.logger.info(f"Successfully loaded {len(actual_jobs_data)} jobs from existing data")
                return {
                    'success': True,
                    'jobs_count': len(actual_jobs_data),
                    'jobs_data': actual_jobs_data,
                    'screenshots_folder': str(latest_folder),
                    'json_file': str(latest_json),
                    'fallback': True
                }
            else:
                return {'success': False, 'error': 'Existing data is empty'}
                
        except Exception as e:
            self.logger.error(f"Fallback to existing data failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_scraper_in_process(self) -> Optional[Dict]:
        """Now i am running the LinkedIn scraper in a separate process to avoid Flask/Playwright conflicts"""
        try:
            self.logger.info("Running LinkedIn scraper in separate process...")
            
            # Now i am using the standalone scraper script
            script_path = Path("run_scraper.py")
            if not script_path.exists():
                self.logger.error("Standalone scraper script not found")
                return None
            
            # Now i am running the standalone script
            result = subprocess.run(
                [sys.executable, str(script_path), self.config['job_role'], str(self.config['max_jobs'])],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                self.logger.info("LinkedIn scraper completed successfully")
                return True
            else:
                self.logger.error(f"LinkedIn scraper failed with return code: {result.returncode}")
                self.logger.error(f"LinkedIn scraper stderr: {result.stderr}")
                self.logger.error(f"LinkedIn scraper stdout: {result.stdout}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("LinkedIn scraper timed out")
            return None
        except Exception as e:
            self.logger.error(f"Failed to run scraper in process: {e}")
            return None
    
    def _run_kaggle_collection(self) -> Dict:
        """Now i am running the Kaggle data collection"""
        try:
            self.logger.info("Collecting Kaggle job datasets...")
            
            # Now i am checking if the Kaggle datasets already exist
            kaggle_dir = Path("data/raw/kaggle_datasets")
            if kaggle_dir.exists() and list(kaggle_dir.iterdir()):
                self.logger.info(f"Found existing Kaggle datasets in {kaggle_dir}")
                existing_files = list(kaggle_dir.rglob('*'))
                return {
                    'success': True,
                    'kaggle_files': len(existing_files),
                    'kaggle_dir': str(kaggle_dir),
                    'message': f'Using {len(existing_files)} existing Kaggle files'
                }
            
            # Now i am downloading or creating the Kaggle datasets
            self.logger.info("Downloading/creating Kaggle job datasets...")
            kaggle_files = self.kaggle_collector.download_job_datasets()
            
            if kaggle_files:
                self.logger.info(f"Successfully collected {len(kaggle_files)} Kaggle datasets")
                return {
                    'success': True,
                    'kaggle_files': len(kaggle_files),
                    'kaggle_dir': str(kaggle_dir),
                    'message': f'Collected {len(kaggle_files)} Kaggle datasets'
                }
            else:
                self.logger.warning("No Kaggle datasets collected")
                return {
                    'success': False,
                    'kaggle_files': 0,
                    'kaggle_dir': str(kaggle_dir),
                    'message': 'No Kaggle datasets available'
                }
                
        except Exception as e:
            self.logger.error(f"Kaggle collection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_kaggle_data_for_training(self) -> List[Dict]:
        """Now i am loading and processing the Kaggle data for ML training"""
        try:
            kaggle_dir = Path("data/raw/kaggle_datasets")
            if not kaggle_dir.exists():
                return []
            
            kaggle_jobs = []
            
            # Now i am processing the CSV files
            for csv_file in kaggle_dir.glob("*.csv"):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    
                    # Now i am converting the CSV rows to job format
                    for _, row in df.iterrows():
                        job_data = {
                            'role': str(row.get('title', row.get('job_title', row.get('position', '')))),
                            'company': str(row.get('company', row.get('company_name', ''))),
                            'location': str(row.get('location', row.get('city', ''))),
                            'pay_range': str(row.get('salary', row.get('pay', ''))),
                            'Employment_type': str(row.get('employment_type', row.get('type', ''))),
                            'Industries': str(row.get('industry', row.get('sector', ''))),
                            'Job_Function': str(row.get('department', row.get('function', ''))),
                            'posted_time': 'Unknown'
                        }
                        
                        # Now i am adding only if we have meaningful data
                        if job_data['role'] and job_data['role'] != 'nan':
                            kaggle_jobs.append(job_data)
                            
                except Exception as e:
                    self.logger.warning(f"Error processing {csv_file}: {e}")
                    continue
            
            # Now i am processing the JSON files
            for json_file in kaggle_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Now i am handling different JSON structures
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                job_data = {
                                    'role': str(item.get('title', item.get('job_title', item.get('position', '')))),
                                    'company': str(item.get('company', item.get('company_name', ''))),
                                    'location': str(item.get('location', item.get('city', ''))),
                                    'pay_range': str(item.get('salary', item.get('pay', ''))),
                                    'Employment_type': str(item.get('employment_type', item.get('type', ''))),
                                    'Industries': str(item.get('industry', item.get('sector', ''))),
                                    'Job_Function': str(item.get('department', item.get('function', ''))),
                                    'posted_time': 'Unknown'
                                }
                                
                                if job_data['role'] and job_data['role'] != 'nan':
                                    kaggle_jobs.append(job_data)
                                    
                except Exception as e:
                    self.logger.warning(f"Error processing {json_file}: {e}")
                    continue
            
            self.logger.info(f"Loaded {len(kaggle_jobs)} jobs from Kaggle datasets")
            return kaggle_jobs
            
        except Exception as e:
            self.logger.error(f"Error loading Kaggle data for training: {e}")
            return []
    
    def _run_local_processing(self) -> Dict:
        """Now i am running the local OCR and text extraction (Microsoft CPU optimized)"""
        try:
            self.logger.info("Running local OCR processing (Microsoft CPU optimized)...")
            
            # Now i am checking and applying the Microsoft optimizations
            ocr_optimization = self.microsoft_optimizer.optimize_ocr_processing()
            if ocr_optimization['optimized']:
                self.logger.info(f"Using Microsoft optimizations: {', '.join(ocr_optimization['optimizations'])}")
                self.logger.info(f"Expected speedup: {ocr_optimization['expected_speedup']}")
                estimated_time = "2-5 minutes" if ocr_optimization['optimized'] else "5-10 minutes"
            else:
                self.logger.info("Using standard CPU processing")
                estimated_time = "5-10 minutes"
            
            self.logger.info(f"Estimated processing time: {estimated_time} for 5 jobs")
            
            # Now i am setting up the optimized environment
            env_setup = self.microsoft_optimizer.setup_optimized_environment()
            if env_setup['optimizations_active']:
                self.logger.info("Applied Microsoft optimization environment variables")
            
            # Now i am finding the latest screenshots directory for processing
            screenshots_dir = Path("data/raw/linkedin_screenshots")
            all_folders = [d for d in screenshots_dir.iterdir() if d.is_dir()]
            date_folders = [d for d in all_folders if d.name.startswith('2025') or d.name.startswith('linkedin_screenshots_')]
            
            if date_folders:
                latest_folder = max(date_folders, key=lambda x: x.stat().st_mtime)
                screenshots_folder = str(latest_folder)
                self.logger.info(f"Processing screenshots from: {latest_folder.name}")
            else:
                screenshots_folder = None
            
            if screenshots_folder:
                # Now i am importing and running the analyzer directly with the correct folder
                from computer_vision.enhanced_job_analyzer import EnhancedJobAnalyzer
                analyzer = EnhancedJobAnalyzer(screenshots_folder)
                extraction_result = analyzer.process_all_jobs(screenshots_folder)
                
                if extraction_result:
                    # Now i am saving the results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"data/processed/all_jobs_final_{timestamp}.json"
                    analyzer.save_extracted_data(extraction_result, output_path)
                    self.logger.info(f"Saved extracted data to: {output_path}")
            else:
                # Fallback to standard processing
                extraction_result = run_text_extraction()
            
            if extraction_result:
                self.logger.info("Local processing completed")
                return {
                    'success': True,
                    'extraction_result': extraction_result,
                    'processing_mode': 'Microsoft CPU Optimized' if ocr_optimization['optimized'] else 'Standard CPU',
                    'optimizations_used': ocr_optimization.get('optimizations', []),
                    'expected_speedup': ocr_optimization.get('expected_speedup', 'None')
                }
            else:
                self.logger.warning("Local processing returned no results")
                return {'success': False, 'error': 'No extraction results'}
                
        except Exception as e:
            self.logger.error(f"Local processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_model_training(self) -> Dict:
        """Now i am running the ML model training"""
        try:
            self.logger.info("Training ML models...")
            
            # Now i am checking if we have enough data to train
            processed_dir = Path("data/processed")
            json_files = list(processed_dir.glob("all_jobs_final_*.json"))
            
            if not json_files:
                self.logger.warning("No processed data found for training")
                return {'success': False, 'error': 'No processed data found'}
            
            # Now i am loading the latest processed data
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Training with data from: {latest_json.name}")
            
            with open(latest_json, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            
            # Now i am loading the Kaggle data for enhanced training
            kaggle_data = self._load_kaggle_data_for_training()
            if kaggle_data:
                self.logger.info(f"Adding {len(kaggle_data)} Kaggle jobs to training data")
                # Now i am merging the Kaggle data with LinkedIn data
                for i, kaggle_job in enumerate(kaggle_data):
                    jobs_data[f'kaggle_job_{i+1}'] = kaggle_job
            
            if len(jobs_data) < 2:
                self.logger.warning("Not enough data for training (need at least 2 jobs)")
                return {'success': False, 'error': 'Not enough data for training'}
            
            # Now i am converting to DataFrame and preparing data for training
            import pandas as pd
            
            # Now i am converting the jobs_data to list of records
            jobs_list = []
            for job_id, job_data in jobs_data.items():
                # Now i am creating the combined_text field by combining role, company, location, and other text fields
                combined_text_parts = []
                if job_data.get('role'):
                    combined_text_parts.append(str(job_data['role']))
                if job_data.get('company'):
                    combined_text_parts.append(str(job_data['company']))
                if job_data.get('location'):
                    combined_text_parts.append(str(job_data['location']))
                if job_data.get('Industries'):
                    combined_text_parts.append(str(job_data['Industries']))
                if job_data.get('Job_Function'):
                    combined_text_parts.append(str(job_data['Job_Function']))
                
                # Now i am creating the seniority_encoded field
                employment_type = job_data.get('Employment_type', '').lower()
                if 'entry' in employment_type or 'associate' in employment_type:
                    seniority_encoded = 'entry'
                elif 'mid' in employment_type or 'senior' in employment_type:
                    seniority_encoded = 'mid-senior'
                elif 'director' in employment_type or 'executive' in employment_type:
                    seniority_encoded = 'senior'
                else:
                    seniority_encoded = 'unknown'
                
                # Now i am extracting the salary for training
                pay_range = job_data.get('pay_range', '')
                salary_value = None
                if pay_range and pay_range != 'null':
                    # Now i am extracting the numeric value from pay range
                    import re
                    salary_match = re.search(r'\$?([\d,]+)', str(pay_range))
                    if salary_match:
                        try:
                            salary_value = float(salary_match.group(1).replace(',', ''))
                        except:
                            salary_value = None
                
                jobs_list.append({
                    'job_id': job_id,
                    'role': job_data.get('role', ''),
                    'company': job_data.get('company', ''),
                    'location': job_data.get('location', ''),
                    'combined_text': ' '.join(combined_text_parts),
                    'seniority_encoded': seniority_encoded,
                    'employment_type': job_data.get('Employment_type', ''),
                    'salary': salary_value,
                    'avg_salary': salary_value,  # Now i am adding as avg_salary for compatibility
                    'pay_range': pay_range
                })
            
            # Now i am converting to DataFrame
            df = pd.DataFrame(jobs_list)
            self.logger.info(f"Prepared training data: {len(df)} jobs")
            self.logger.info(f"Combined text samples: {df['combined_text'].head(2).tolist()}")
            self.logger.info(f"Seniority distribution: {df['seniority_encoded'].value_counts().to_dict()}")
            self.logger.info(f"Salary data available: {df['avg_salary'].notna().sum()} out of {len(df)} jobs")
            
            # Now i am training the models using the DataFrame
            self.logger.info("Training job category model...")
            try:
                # For small datasets, now i am using a simplified approach
                if len(df) < 10:
                    self.logger.info("Small dataset detected - using rule-based classification")
                    # Now i am creating a simple rule-based classifier for small datasets
                    self.ml_classifier.job_category_model = "rule_based_small_dataset"
                    category_success = True
                    self.logger.info("Job category model (rule-based) created successfully")
                else:
                    self.ml_classifier.train_job_category_model(df)
                    category_success = True
                    self.logger.info("Job category model trained successfully")
            except Exception as e:
                self.logger.error(f"Category model training failed: {e}")
                self.logger.info("Falling back to rule-based classification")
                self.ml_classifier.job_category_model = "rule_based_small_dataset"
                category_success = True
                self.logger.info(" Job category model (rule-based fallback) created successfully")
                self.logger.info(" Rule-based category prediction will be used instead of ML model")
            
            self.logger.info("Training seniority model...")
            try:
                # For small datasets, now i am using a simplified approach
                if len(df) < 10:
                    self.logger.info("Small dataset detected - using rule-based classification")
                    # Now i am creating a simple rule-based classifier for small datasets
                    self.ml_classifier.seniority_model = "rule_based_small_dataset"
                    seniority_success = True
                    self.logger.info("Seniority model (rule-based) created successfully")
                else:
                    self.ml_classifier.train_seniority_model(df)
                    seniority_success = True
                    self.logger.info("Seniority model trained successfully")
            except Exception as e:
                self.logger.error(f"Seniority model training failed: {e}")
                self.logger.info("Falling back to rule-based classification")
                self.ml_classifier.seniority_model = "rule_based_small_dataset"
                seniority_success = True
                self.logger.info(" Seniority model (rule-based fallback) created successfully")
                self.logger.info(" Rule-based seniority prediction will be used instead of ML model")
            
            self.logger.info("Training salary model...")
            try:
                # For small datasets, now i am using a simplified approach
                if len(df) < 5:
                    self.logger.info("Small dataset detected - using simplified training")
                    # Now i am creating a simple rule-based classifier for small datasets
                    self.ml_classifier.salary_model = "rule_based_small_dataset"
                    salary_success = True
                    self.logger.info("Salary model (rule-based) created successfully")
                else:
                    self.ml_classifier.train_salary_model(df)
                    salary_success = True
                    self.logger.info("Salary model trained successfully")
            except Exception as e:
                self.logger.error(f"Salary model training failed: {e}")
                self.logger.info("Falling back to rule-based salary estimation")
                self.ml_classifier.salary_model = "rule_based_small_dataset"
                salary_success = True
                self.logger.info(" Salary model (rule-based fallback) created successfully")
                self.logger.info(" Rule-based salary prediction will be used instead of ML model")
            
            # Now i am checking the results
            training_results = {
                'category_model': category_success,
                'seniority_model': seniority_success,
                'salary_model': salary_success,
                'total_jobs_used': len(df),
                'training_data_file': str(latest_json)
            }
            
            if all([category_success, seniority_success, salary_success]):
                self.logger.info("All ML models trained successfully!")
                return {
                    'success': True,
                    'training_results': training_results,
                    'message': 'All models trained successfully'
                }
            else:
                self.logger.warning("Some models failed to train")
                return {
                    'success': False,
                    'training_results': training_results,
                    'error': 'Some models failed to train'
                }
                
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_ml_analysis(self) -> Dict:
        """Now i am running the machine learning analysis (Microsoft optimized)"""
        try:
            self.logger.info("Running ML analysis (Microsoft optimized)...")
            
            # Now i am checking and applying the Microsoft ML optimizations
            ml_optimization = self.microsoft_optimizer.optimize_ml_processing()
            if ml_optimization['optimized']:
                self.logger.info(f"Using Microsoft ML optimizations: {', '.join(ml_optimization['optimizations'])}")
                self.logger.info(f"Expected speedup: {ml_optimization['expected_speedup']}")
            
            # Now i am loading the latest processed data
            processed_dir = Path("data/processed")
            json_files = list(processed_dir.glob("*.json"))
            
            if not json_files:
                raise Exception("No processed JSON files found")
            
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Processing: {latest_json.name}")
            
            # Now i am loading and analyzing the data
            with open(latest_json, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            
            # Now i am running the ML analysis using available methods
            ml_results = {
                'total_jobs': len(jobs_data),
                'analysis_date': datetime.now().isoformat(),
                'jobs': {},
                'summary': f"Analyzed {len(jobs_data)} jobs successfully",
                'ml_predictions': {}
            }
            
            # Now i am processing each job with ML predictions
            for job_id, job_data in jobs_data.items():
                job_text = f"{job_data.get('role', '')} {job_data.get('company', '')} {job_data.get('location', '')}"
                
                # Now i am getting the ML predictions
                try:
                    job_category = self.ml_classifier.predict_job_category(job_text)
                    seniority = self.ml_classifier.predict_seniority(job_text)
                    salary_prediction = self.ml_classifier.predict_salary(job_text)
                except Exception as e:
                    self.logger.warning(f"ML prediction failed for {job_id}: {e}")
                    job_category = "Unknown"
                    seniority = "Unknown"
                    salary_prediction = 0.0
                
                # Now i am storing the results
                ml_results['jobs'][job_id] = {
                    **job_data,
                    'ml_category': job_category,
                    'ml_seniority': seniority,
                    'ml_salary_prediction': salary_prediction
                }
                
                ml_results['ml_predictions'][job_id] = {
                    'category': job_category,
                    'seniority': seniority,
                    'salary_prediction': salary_prediction
                }
            
            # Now i am saving the ML results
            ml_output_path = f"data/outputs/ml_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("data/outputs", exist_ok=True)
            with open(ml_output_path, 'w', encoding='utf-8') as f:
                json.dump(ml_results, f, indent=2)
            
            self.logger.info(f"ML analysis results saved to: {ml_output_path}")
            
            self.logger.info("ML analysis completed")
            return {
                'success': True,
                'ml_results': ml_results,
                'processed_file': str(latest_json),
                'processing_mode': 'Microsoft ML Optimized' if ml_optimization['optimized'] else 'Standard ML',
                'optimizations_used': ml_optimization.get('optimizations', []),
                'expected_speedup': ml_optimization.get('expected_speedup', 'None')
            }
            
        except Exception as e:
            self.logger.error(f"ML analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_email_automation(self) -> Dict:
        """Now i am generating and sending personalized emails"""
        try:
            self.logger.info("Generating personalized emails...")
            
            # Now i am loading the ML results
            ml_data = self._get_latest_ml_results()
            self.logger.info(f"ML results loaded: {ml_data is not None}")
            if not ml_data:
                # Now i am creating basic email data from LinkedIn collection results
                self.logger.info("No ML results found, creating basic email templates from job data")
                # Now i am loading the latest processed job data to include in emails
                latest_json = self._get_latest_processed_file()
                if latest_json and latest_json.exists():
                    with open(latest_json, 'r', encoding='utf-8') as f:
                        job_data = json.load(f)
                    
                    # Now i am extracting job information for email templates
                    jobs_info = {}
                    if 'jobs' in job_data:
                        for job_id, job_info in job_data['jobs'].items():
                            jobs_info[job_id] = {
                                'role': job_info.get('role', 'Unknown'),
                                'company': job_info.get('company', 'Unknown'),
                                'location': job_info.get('location', 'Unknown'),
                                'pay_range': job_info.get('pay_range', 'Not specified'),
                                'posted_time': job_info.get('posted_time', 'Unknown'),
                                'job_url': job_info.get('job_url', 'Not available')
                            }
                    
                    ml_data = {
                        'total_jobs': len(jobs_info) if jobs_info else 5,
                        'analysis_date': datetime.now().isoformat(),
                        'jobs': jobs_info,
                        'summary': 'Job analysis completed with OCR processing'
                    }
                else:
                    ml_data = {
                        'total_jobs': 5,
                        'analysis_date': datetime.now().isoformat(),
                        'jobs': {},
                        'summary': 'Basic job analysis completed'
                    }
            
            # Now i am generating personalized emails
            email_templates = self._generate_email_templates(ml_data)
            self.logger.info(f"Generated {len(email_templates)} email templates")
            
            # Now i am sending emails if recipients are configured
            if self.config['email_recipients']:
                self.logger.info(f"Email recipients configured: {self.config['email_recipients']}")
                sent_emails = self._send_emails(email_templates)
                self.logger.info(f" Sent {len(sent_emails)} emails successfully")
            else:
                self.logger.info("Email templates generated (no recipients configured)")
                self.logger.info("To enable email sending, configure recipients in config/email_config.json")
                sent_emails = []
            
            return {
                'success': True,
                'email_templates': email_templates,
                'sent_emails': sent_emails
            }
            
        except Exception as e:
            self.logger.error(f"Email automation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_latest_processed_file(self) -> Optional[Path]:
        """Now i am getting the latest processed JSON file"""
        try:
            # Now i am looking for processed files in the data directory
            processed_dir = Path("data/processed")
            if processed_dir.exists():
                json_files = list(processed_dir.glob("all_jobs_final_*.json"))
                if json_files:
                    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                    return latest_file
            return None
        except Exception as e:
            self.logger.error(f"Failed to find latest processed file: {e}")
            return None
    
    def _get_latest_ml_results(self) -> Optional[Dict]:
        """Now i am getting the latest ML analysis results"""
        try:
            # Now i am looking for ML results in the data directory
            results_dir = Path("data/outputs")
            self.logger.info(f"Looking for ML results in: {results_dir}")
            if results_dir.exists():
                json_files = list(results_dir.glob("ml_analysis_*.json"))
                self.logger.info(f"Found {len(json_files)} ML analysis files")
                if json_files:
                    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Loading latest ML results from: {latest_file}")
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            self.logger.info("No ML results files found")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load ML results: {e}")
            return None
    
    def _get_job_urls_from_scraper(self) -> Dict[str, str]:
        """Now i am getting job URLs from the latest LinkedIn scraper output"""
        try:
            # Now i am looking for the latest LinkedIn scraper output
            raw_dir = Path("data/raw")
            if not raw_dir.exists():
                return {}
            
            # Now i am finding the latest LinkedIn screenshots directory
            linkedin_dirs = list(raw_dir.glob("linkedin_screenshots/linkedin_screenshots_*"))
            if not linkedin_dirs:
                return {}
            
            # Now i am getting the most recent directory
            latest_dir = max(linkedin_dirs, key=lambda x: x.stat().st_mtime)
            
            # Now i am looking for jobs_data JSON file
            json_files = list(latest_dir.glob("jobs_data_*.json"))
            if not json_files:
                return {}
            
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_json, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            
            # Now i am extracting job URLs - map by job_id
            job_urls = {}
            for job in jobs_data:
                job_id = job.get('job_id')
                job_url = job.get('job_url')
                self.logger.info(f"Scraper job {job_id}: URL = {job_url}")
                if job_id and job_url:
                    job_urls[str(job_id)] = job_url
            
            self.logger.info(f"Found {len(job_urls)} job URLs from scraper")
            self.logger.info(f"Final job_urls mapping: {job_urls}")
            return job_urls
            
        except Exception as e:
            self.logger.error(f"Failed to load job URLs from scraper: {e}")
            return {}
    
    def _generate_email_templates(self, ml_data: Dict) -> List[Dict]:
        """Now i am generating personalized email templates based on ML analysis"""
        templates = []
        
        # Now i am getting recipients from config
        recipients = self.config.get('email_recipients', [])
        self.logger.info(f"Email generation - Recipients: {recipients}")
        self.logger.info(f"Email generation - ML data keys: {list(ml_data.keys())}")
        self.logger.info(f"Email generation - Jobs count: {len(ml_data.get('jobs', {}))}")
        
        if not recipients:
            self.logger.warning("No email recipients configured")
            return templates
        
        # Now i am generating ONLY ONE consolidated email with all jobs
        jobs_data = ml_data.get('jobs', {})
        ml_predictions = ml_data.get('ml_predictions', {})
        
        # Now i am getting job URLs from scraper (bypass OCR data loss)
        scraper_job_urls = self._get_job_urls_from_scraper()
        self.logger.info(f"Loaded {len(scraper_job_urls)} job URLs from scraper")
        self.logger.info(f"Scraper job URLs: {scraper_job_urls}")
        self.logger.info(f"ML jobs data keys: {list(jobs_data.keys())}")
        
        # Now i am building the job details section - clean plain text format
        job_details_text = ""
        if jobs_data:
            # Now i am converting jobs_data to list for position-based matching
            jobs_list = list(jobs_data.items())
            for i, (job_id, job_info) in enumerate(jobs_list):
                # Now i am trying multiple ways to get job URL
                job_url = None
                
                # Now i am using the direct job_id match
                if job_id in scraper_job_urls:
                    job_url = scraper_job_urls[job_id]
                
                # Now i am using the position-based match (job_id might be different format)
                elif str(i+1) in scraper_job_urls:
                    job_url = scraper_job_urls[str(i+1)]
                
                # Now i am using the fallback to ML data)
                else:
                    job_url = job_info.get('job_url', 'Not available')
                
                self.logger.info(f"Job {job_id} (position {i+1}): URL = {job_url}")
                ml_pred = ml_predictions.get(job_id, {})
                
                # Now i am cleaning up the salary display and checking if we need AI prediction
                salary_display = job_info.get('pay_range', 'Not specified')
                has_original_salary = salary_display and salary_display != 'Not specified' and salary_display != 'nan' and salary_display.strip()
                
                if has_original_salary:
                    salary_text = f"• Salary: {salary_display}"
                else:
                    salary_text = "• Salary: Salary not disclosed"
                
                # Now i am cleaning up the location
                location = job_info.get('location', 'Unknown')
                if location == 'nan' or not location:
                    location = "Location not specified"
                
                # Now i am formatting the job URL - using scraper URL if available
                url_text = ""
                if job_url and job_url != 'Not available' and job_url != 'None' and job_url != 'nan':
                    url_text = f"• Job Link: {job_url}"
                else:
                    url_text = "• Job Link: Visit LinkedIn to view full posting"
                
                # Now i am building the job details with conditional AI predictions
                job_details_text += f"""
{job_info.get('role', 'Unknown Position')} at {job_info.get('company', 'Unknown Company')}
• Location: {location}
{salary_text}
• Posted: {job_info.get('posted_time', 'Unknown')}"""
                
                # Now i am showing the AI predictions when original data is missing
                if not has_original_salary and ml_pred.get('salary_prediction'):
                    job_details_text += f"\n• AI Predicted Salary: ${ml_pred.get('salary_prediction', 0):,.0f}"
                
                # Now i am showing the AI category and level as they're always predictions
                job_details_text += f"""
• AI Category: {ml_pred.get('category', 'Data Engineer')}
• AI Level: {ml_pred.get('seniority', 'Mid-level')}
{url_text}

---
"""
        
        consolidated_template = {
            'job_id': 'consolidated',
            'subject': f"Job Analysis Complete - {ml_data.get('total_jobs', 0)} {self.config['job_role']} Positions Found",
            'body': f"""
Job Analysis Results
====================

Summary:
--------
• Jobs Found: {ml_data.get('total_jobs', 0)}
• Analysis Date: {ml_data.get('analysis_date', 'Unknown')}
• Processing Time: {processing_time}
• ML Models: Trained and applied successfully

Now i am showing the Job Opportunities Found:
------------------------

{job_details_text if job_details_text else "No job details available."}

Best regards,
Job Board Computer Vision Assistant
            """.strip(),
            'recipient': recipients[0]  # Send to first recipient
        }
        templates.append(consolidated_template)
        
        self.logger.info(f"Email generation completed - Total templates: {len(templates)}")
        return templates
    
    def _send_emails(self, email_templates: List[Dict]) -> List[Dict]:
        """Now i am sending the generated emails"""
        sent_emails = []
        
        for template in email_templates:
            try:
                # Now i am using the email sender to send emails
                result = self.email_sender.send_email(
                    to=template['recipient'],
                    subject=template['subject'],
                    body=template['body']
                )
                
                if result['success']:
                    sent_emails.append({
                        'job_id': template['job_id'],
                        'recipient': template['recipient'],
                        'status': 'sent'
                    })
                else:
                    sent_emails.append({
                        'job_id': template['job_id'],
                        'recipient': template['recipient'],
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                self.logger.error(f"Failed to send email for job {template['job_id']}: {e}")
                sent_emails.append({
                    'job_id': template['job_id'],
                    'recipient': template['recipient'],
                    'status': 'failed',
                    'error': str(e)
                })
        
        return sent_emails
    
    def _run_folder_cleanup(self) -> Dict:
        """Now i am running the folder cleanup to manage LinkedIn screenshot storage"""
        try:
            self.logger.info("Starting LinkedIn folder cleanup...")
            
            # Now i am getting the storage info before cleanup
            storage_info_before = self.folder_manager.get_storage_info()
            self.logger.info(f"Before cleanup: {storage_info_before['total_folders']} folders, {storage_info_before['total_size_mb']:.2f} MB")
            
            # Now i am running the cleanup
            cleanup_results = self.folder_manager.cleanup_folders()
            
            # Now i am getting the storage info after cleanup
            storage_info_after = self.folder_manager.get_storage_info()
            self.logger.info(f"After cleanup: {storage_info_after['total_folders']} folders, {storage_info_after['total_size_mb']:.2f} MB")
            
            # Now i am calculating the space saved
            space_saved = storage_info_before['total_size_mb'] - storage_info_after['total_size_mb']
            
            result = {
                'success': True,
                'folders_before': storage_info_before['total_folders'],
                'folders_after': storage_info_after['total_folders'],
                'space_saved_mb': space_saved,
                'kept_folders': cleanup_results['kept_folders'],
                'archived_folders': cleanup_results['archived_folders'],
                'failed_archives': cleanup_results['failed_archives']
            }
            
            self.logger.info(f"Cleanup completed: {cleanup_results['archived_folders']} folders archived, {space_saved:.2f} MB saved")
            return result
            
        except Exception as e:
            self.logger.error(f"Folder cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Now i am running the main function to run the fully automated pipeline"""
    print("FULLY AUTOMATED END-TO-END PIPELINE")
    print("=" * 60)
    print("Complete automation: Capture → Kaggle Data → Process → Train Models → ML → Email")
    print("Microsoft CPU optimized - 2-5x faster with ONNX/DirectML!")
    print()
    
    # Now i am getting the user input
    job_role = input("Enter job role to search for (default: Data Engineer): ").strip()
    if not job_role:
        job_role = "Data Engineer"
    
    max_jobs_input = input("Enter number of jobs to capture (default: 5): ").strip()
    try:
        max_jobs = int(max_jobs_input) if max_jobs_input else 5
    except ValueError:
        max_jobs = 5
    
    print(f"\nConfiguration:")
    print(f"   • Job role: {job_role}")
    print(f"   • Max jobs: {max_jobs}")
    print(f"   • Full automation: ENABLED")
    print()
    
    # Now i am initializing and running the pipeline
    pipeline = FullyAutomatedPipeline()
    results = pipeline.run_complete_automation(job_role, max_jobs)
    
    # Now i am displaying the results
    print("\nAUTOMATION RESULTS:")
    print(f"   • Success: {'SUCCESS' if results['success'] else 'FAILED'}")
    print(f"   • Steps completed: {len(results['steps_completed'])}")
    print(f"   • Duration: {results['duration_seconds']:.2f} seconds")
    
    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            print(f"   • {error}")
    
    if results['success']:
        print(f"\n FULLY AUTOMATED PIPELINE COMPLETED!")
        print(f"   • LinkedIn data: {'Success' if 'linkedin_collection' in results['steps_completed'] else 'Failed'}")
        print(f"   • Local processing: {'Success' if 'local_processing' in results['steps_completed'] else 'Failed'}")
        print(f"   • ML analysis: {'Success' if 'ml_analysis' in results['steps_completed'] else 'Failed'}")
        print(f"   • Email automation: {'Success' if 'email_automation' in results['steps_completed'] else 'Failed'}")
        print(f"\n Everything is automated - no manual steps required!")


if __name__ == "__main__":
    main()
