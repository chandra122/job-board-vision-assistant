#!/usr/bin/env python3
"""
Standalone LinkedIn Scraper Runner
==================================

This script runs the LinkedIn scraper independently to avoid Flask/Playwright conflicts.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run LinkedIn scraper with command line arguments"""
    if len(sys.argv) < 3:
        print("Usage: python run_scraper.py <job_role> <max_jobs>")
        sys.exit(1)
    
    job_role = sys.argv[1]
    max_jobs = int(sys.argv[2])
    
    try:
        from data_collection.linkedin_scraper import main as run_linkedin_scraper
        
        print(f"Running LinkedIn scraper for '{job_role}' with {max_jobs} jobs...")
        print("Running in headless mode for web application compatibility...")
        
        # Now i am setting the environment variable to run in headless mode
        import os
        os.environ['PLAYWRIGHT_HEADLESS'] = 'true'
        
        result = run_linkedin_scraper(
            job_term=job_role,
            max_jobs=max_jobs
        )
        
        # Now i am checking if the result is not None (success) vs None/[] (failure)
        if result is not None:
            print("SUCCESS: LinkedIn scraper completed")
            sys.exit(0)
        else:
            print("ERROR: LinkedIn scraper failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
