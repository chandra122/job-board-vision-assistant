#!/usr/bin/env python3
"""
Kaggle Data Collector
Downloads and manages job-related datasets from Kaggle

References:
- Kaggle API Documentation: https://www.kaggle.com/docs/api
- Kaggle Datasets: https://www.kaggle.com/datasets
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

class KaggleDataCollector:
    """
    Collecting job-related data from Kaggle
    """
    
    def __init__(self):
        self.datasets_dir = Path("data/raw/kaggle_datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Now i am creating the job-related Kaggle datasets
        self.job_datasets = [
            {
                "name": "comprehensive_job_postings",
                "url": "https://www.kaggle.com/datasets/promptcloud/job-postings",
                "description": "Comprehensive Job Postings (All Industries)",
                "file_type": "csv",
                "job_types": ["All Industries", "Tech", "Finance", "Healthcare", "Education"]
            },
            {
                "name": "tech_jobs_2023",
                "url": "https://www.kaggle.com/datasets/andrewmvd/data-scientist-jobs",
                "description": "Tech Jobs 2023 (Data Science, Engineering, ML Engineering)",
                "file_type": "csv",
                "job_types": ["Data Scientist", "Software Engineer", "Machine Learning Engineer", "ML Engineer"]
            },
            {
                "name": "software_engineering_jobs",
                "url": "https://www.kaggle.com/datasets/ahmedhassan660/software-engineering-jobs",
                "description": "Software Engineering Jobs (Various Levels)",
                "file_type": "csv",
                "job_types": ["Software Engineer", "Senior Developer", "Tech Lead"]
            },
            {
                "name": "remote_jobs_2023",
                "url": "https://www.kaggle.com/datasets/madhab/jobposts",
                "description": "Remote Jobs 2023 (Multiple Industries)",
                "file_type": "csv",
                "job_types": ["Remote", "Work from Home", "Flexible"]
            },
            {
                "name": "startup_jobs",
                "url": "https://www.kaggle.com/datasets/atharvap329/glassdoor-data-science-job-data",
                "description": "Startup Jobs (Glassdoor Data)",
                "file_type": "csv",
                "job_types": ["Startup", "Tech", "Data Science"]
            },
            {
                "name": "job_salaries_2023",
                "url": "https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries",
                "description": "Job Salaries 2023 (Tech & Data Roles)",
                "file_type": "csv",
                "job_types": ["Data Scientist", "ML Engineer", "Data Engineer", "Analyst"]
            },
            {
                "name": "indeed_jobs_diverse",
                "url": "https://www.kaggle.com/datasets/elroyggj/indeed-dataset-data-scientist",
                "description": "Indeed Jobs (Diverse Tech Roles)",
                "file_type": "csv",
                "job_types": ["Data Scientist", "Analyst", "Engineer", "Manager"]
            }
        ]
    
    def download_job_datasets(self):
        """Now i am downloading job datasets from Kaggle"""
        self.logger.info("DOWNLOADING KAGGLE JOB DATASETS")
        self.logger.info("=" * 50)
        
        downloaded_files = []
        
        # Now i am checking if the Kaggle API is available
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.logger.info("Kaggle API found - downloading real datasets...")
            
            # Now i am initializing the Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Now i am downloading each dataset
            for dataset in self.job_datasets:
                try:
                    self.logger.info(f"Downloading {dataset['name']}...")
                    
                    # Now i am extracting the dataset identifier from the URL
                    dataset_id = dataset['url'].split('/')[-1]
                    dataset_owner = dataset['url'].split('/')[-2]
                    full_dataset_name = f"{dataset_owner}/{dataset_id}"
                    
                    # Now i am creating the dataset-specific directory
                    dataset_dir = self.datasets_dir / dataset['name']
                    dataset_dir.mkdir(exist_ok=True)
                    
                    # Download the dataset
                    api.dataset_download_files(
                        full_dataset_name, 
                        path=str(dataset_dir), 
                        unzip=True
                    )
                    
                    # Now i am finding the downloaded files
                    downloaded_files.extend([
                        str(f) for f in dataset_dir.rglob("*") 
                        if f.is_file() and f.suffix in ['.csv', '.json', '.xlsx']
                    ])
                    
                    self.logger.info(f"Successfully downloaded {dataset['name']}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not download {dataset['name']}: {e}")
                    # Now i am continuing with other datasets
            
            # If no real datasets were downloaded, now i am creating sample data as fallback
            if not downloaded_files:
                self.logger.info("No real datasets downloaded - creating enhanced sample data...")
                downloaded_files = self.create_enhanced_sample_data()
            
        except ImportError:
            self.logger.info("Kaggle API not found - creating enhanced sample data...")
            downloaded_files = self.create_enhanced_sample_data()
        except Exception as e:
            self.logger.warning(f"Kaggle API error: {e} - creating enhanced sample data...")
            downloaded_files = self.create_enhanced_sample_data()
        
        return downloaded_files
    
    def create_sample_data(self):
        """Now i am creating sample job data for analysis"""
        self.logger.info("Creating sample job data...")
        
        # Now i am creating comprehensive sample job data
        sample_jobs = []
        
        # Now i am creating Data Engineering jobs
        for i in range(50):
            sample_jobs.append({
                "job_title": f"Data Engineer {i+1}",
                "company": f"Tech Company {i+1}",
                "location": "San Francisco, CA",
                "job_type": "data_engineering",
                "salary": f"${80000 + i*2000}-${120000 + i*3000}",
                "skills": ["Python", "SQL", "Apache Spark", "AWS", "Docker"],
                "requirements": ["Bachelor's degree", "3+ years experience", "Python proficiency"],
                "description": f"Looking for a skilled Data Engineer to build and maintain data pipelines...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "sample_data"
            })
        
        # Now i am creating Software Engineering jobs
        for i in range(40):
            sample_jobs.append({
                "job_title": f"Software Engineer {i+1}",
                "company": f"Software Corp {i+1}",
                "location": "Seattle, WA",
                "job_type": "software_engineering",
                "salary": f"${90000 + i*2500}-${140000 + i*4000}",
                "skills": ["JavaScript", "React", "Node.js", "MongoDB", "Git"],
                "requirements": ["Computer Science degree", "2+ years experience", "Full-stack development"],
                "description": f"Join our team as a Software Engineer to develop cutting-edge applications...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "sample_data"
            })
        
        # Now i am creating Machine Learning jobs
        for i in range(30):
            sample_jobs.append({
                "job_title": f"Machine Learning Engineer {i+1}",
                "company": f"AI Startup {i+1}",
                "location": "Austin, TX",
                "job_type": "machine_learning",
                "salary": f"${100000 + i*3000}-${160000 + i*5000}",
                "skills": ["Python", "TensorFlow", "PyTorch", "MLOps", "Kubernetes"],
                "requirements": ["PhD or Master's", "5+ years ML experience", "Deep learning expertise"],
                "description": f"Seeking a Machine Learning Engineer to build and deploy ML models...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "sample_data"
            })
        
        # Now i am creating Data Science jobs
        for i in range(35):
            sample_jobs.append({
                "job_title": f"Data Scientist {i+1}",
                "company": f"Analytics Inc {i+1}",
                "location": "New York, NY",
                "job_type": "data_science",
                "salary": f"${95000 + i*2500}-${150000 + i*4000}",
                "skills": ["Python", "R", "Pandas", "Scikit-learn", "Tableau"],
                "requirements": ["Statistics background", "3+ years experience", "SQL proficiency"],
                "description": f"Data Scientist needed to analyze complex datasets and build predictive models...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "sample_data"
            })
        
        # Now i am creating DevOps jobs
        for i in range(25):
            sample_jobs.append({
                "job_title": f"DevOps Engineer {i+1}",
                "company": f"Cloud Solutions {i+1}",
                "location": "Denver, CO",
                "job_type": "devops",
                "salary": f"${85000 + i*2000}-${130000 + i*3000}",
                "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"],
                "requirements": ["Linux experience", "2+ years DevOps", "Cloud certifications"],
                "description": f"DevOps Engineer to manage cloud infrastructure and CI/CD pipelines...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "sample_data"
            })
        
            # Now i am saving the sample data
        json_file = self.datasets_dir / "sample_job_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sample_jobs, f, indent=2, ensure_ascii=False)
        
        # Now i am creating the CSV version
        df = pd.DataFrame(sample_jobs)
        csv_file = self.datasets_dir / "sample_job_data.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Now i am creating the summary
        summary = {
            "total_jobs": len(sample_jobs),
            "job_types": {
                "data_engineering": len([j for j in sample_jobs if j["job_type"] == "data_engineering"]),
                "software_engineering": len([j for j in sample_jobs if j["job_type"] == "software_engineering"]),
                "machine_learning": len([j for j in sample_jobs if j["job_type"] == "machine_learning"]),
                "data_science": len([j for j in sample_jobs if j["job_type"] == "data_science"]),
                "devops": len([j for j in sample_jobs if j["job_type"] == "devops"])
            },
            "created_at": datetime.now().isoformat(),
            "files": [str(json_file), str(csv_file)]
        }
        
        summary_file = self.datasets_dir / "sample_data_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Sample data created:")
        self.logger.info(f"   JSON: {json_file}")
        self.logger.info(f"   CSV: {csv_file}")
        self.logger.info(f"   Summary: {summary_file}")
        self.logger.info(f"   Total jobs: {len(sample_jobs)}")
        
        return [str(json_file), str(csv_file), str(summary_file)]
    
    def create_enhanced_sample_data(self):
        """Now i am creating enhanced sample job data with more variety and volume"""
        self.logger.info("Creating enhanced sample job data...")
        
        # Now i am creating comprehensive sample job data
        sample_jobs = []
        
        # Now i am creating Data Engineering jobs (increased)
        for i in range(200):
            sample_jobs.append({
                "job_title": f"Data Engineer {i+1}",
                "company": f"Tech Company {i+1}",
                "location": "San Francisco, CA",
                "job_type": "data_engineering",
                "salary": f"${80000 + i*2000}-${120000 + i*3000}",
                "skills": ["Python", "SQL", "Apache Spark", "AWS", "Docker"],
                "requirements": ["Bachelor's degree", "3+ years experience", "Python proficiency"],
                "description": f"Looking for a skilled Data Engineer to build and maintain data pipelines...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "enhanced_sample_data"
            })
        
        # Now i am creating Software Engineering jobs (increased)
        for i in range(180):
            sample_jobs.append({
                "job_title": f"Software Engineer {i+1}",
                "company": f"Software Corp {i+1}",
                "location": "Seattle, WA",
                "job_type": "software_engineering",
                "salary": f"${90000 + i*2500}-${140000 + i*4000}",
                "skills": ["JavaScript", "React", "Node.js", "MongoDB", "Git"],
                "requirements": ["Computer Science degree", "2+ years experience", "Full-stack development"],
                "description": f"Join our team as a Software Engineer to develop cutting-edge applications...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "enhanced_sample_data"
            })
        
        # Now i am creating Machine Learning jobs (increased)
        for i in range(150):
            sample_jobs.append({
                "job_title": f"Machine Learning Engineer {i+1}",
                "company": f"AI Startup {i+1}",
                "location": "Austin, TX",
                "job_type": "machine_learning",
                "salary": f"${100000 + i*3000}-${160000 + i*5000}",
                "skills": ["Python", "TensorFlow", "PyTorch", "MLOps", "Kubernetes"],
                "requirements": ["PhD or Master's", "5+ years ML experience", "Deep learning expertise"],
                "description": f"Seeking a Machine Learning Engineer to build and deploy ML models...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "enhanced_sample_data"
            })
        
        # Now i am creating Data Science jobs (increased)
        for i in range(170):
            sample_jobs.append({
                "job_title": f"Data Scientist {i+1}",
                "company": f"Analytics Inc {i+1}",
                "location": "New York, NY",
                "job_type": "data_science",
                "salary": f"${95000 + i*2500}-${150000 + i*4000}",
                "skills": ["Python", "R", "Pandas", "Scikit-learn", "Tableau"],
                "requirements": ["Statistics background", "3+ years experience", "SQL proficiency"],
                "description": f"Data Scientist needed to analyze complex datasets and build predictive models...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "enhanced_sample_data"
            })
        
        # Now i am creating DevOps jobs (increased)
        for i in range(120):
            sample_jobs.append({
                "job_title": f"DevOps Engineer {i+1}",
                "company": f"Cloud Solutions {i+1}",
                "location": "Denver, CO",
                "job_type": "devops",
                "salary": f"${85000 + i*2000}-${130000 + i*3000}",
                "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"],
                "requirements": ["Linux experience", "2+ years DevOps", "Cloud certifications"],
                "description": f"DevOps Engineer to manage cloud infrastructure and CI/CD pipelines...",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "enhanced_sample_data"
            })
        
        # Now i am creating Additional job types
        job_types = [
            ("Product Manager", "product_management", 100, "Boston, MA"),
            ("UX Designer", "design", 80, "Portland, OR"),
            ("Data Analyst", "data_analysis", 150, "Chicago, IL"),
            ("Backend Developer", "backend_development", 130, "Miami, FL"),
            ("Frontend Developer", "frontend_development", 120, "Phoenix, AZ"),
            ("Full Stack Developer", "fullstack_development", 140, "Dallas, TX"),
            ("Cloud Architect", "cloud_architecture", 90, "Atlanta, GA"),
            ("Security Engineer", "security", 110, "Washington, DC")
        ]
        
        for job_title, job_type, count, location in job_types:
            for i in range(count):
                sample_jobs.append({
                    "job_title": f"{job_title} {i+1}",
                    "company": f"Company {i+1}",
                    "location": location,
                    "job_type": job_type,
                    "salary": f"${70000 + i*1500}-${120000 + i*2500}",
                    "skills": ["Python", "SQL", "AWS", "Docker", "Git"],
                    "requirements": ["Bachelor's degree", "2+ years experience", "Relevant skills"],
                    "description": f"Looking for a {job_title} to join our growing team...",
                    "posted_date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "enhanced_sample_data"
                })
        
        # Now i am saving the enhanced sample data
        json_file = self.datasets_dir / "enhanced_job_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sample_jobs, f, indent=2, ensure_ascii=False)
        
            # Now i am creating the CSV version
        df = pd.DataFrame(sample_jobs)
        csv_file = self.datasets_dir / "enhanced_job_data.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Now i am creating the summary
        summary = {
            "total_jobs": len(sample_jobs),
            "job_types": {
                job_type: len([j for j in sample_jobs if j["job_type"] == job_type])
                for job_type in set(j["job_type"] for j in sample_jobs)
            },
            "created_at": datetime.now().isoformat(),
            "files": [str(json_file), str(csv_file)],
            "enhanced": True
        }
        
        summary_file = self.datasets_dir / "enhanced_data_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Enhanced sample data created:")
        self.logger.info(f"   JSON: {json_file}")
        self.logger.info(f"   CSV: {csv_file}")
        self.logger.info(f"   Summary: {summary_file}")
        self.logger.info(f"   Total jobs: {len(sample_jobs)}")
        
        return [str(json_file), str(csv_file), str(summary_file)]
    
    def get_available_datasets(self):
        """Now i am getting the list of available datasets"""
        return self.job_datasets

def main():
    """Now i am testing the Kaggle data collector"""
    collector = KaggleDataCollector()
    files = collector.download_job_datasets()
    print(f"Downloaded {len(files)} files")

if __name__ == "__main__":
    main()
