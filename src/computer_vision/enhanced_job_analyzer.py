#!/usr/bin/env python3
"""
I am creating an enhanced job screenshot analyzer for LinkedIn job data
Updated to work with the new Playwright-based screenshot structure

This module implements OCR-based text extraction and parsing for job posting data.
The parsing logic combines:
1. Original working extraction methods (developed for this project)
2. Enhanced heuristics for common OCR text patterns
3. Standard text preprocessing techniques for structured data

References:
- EasyOCR Documentation: https://github.com/JaidedAI/EasyOCR
- OpenCV Python Tutorials: https://opencv-python-tutroals.readthedocs.io/
- Nicholas Renotte - Computer Vision Tutorials: https://www.youtube.com/c/NicholasRenotte
- Playwright Documentation: https://playwright.dev/python/

All code is original work developed specifically for this computer vision project.
No external code was copied without proper attribution.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional
import easyocr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedJobAnalyzer:
    """
    I am creating an enhanced analyzer for the new Playwright-based screenshot structure
    Handles job_card_X.png, job_attributes_Job_X.png, pay_info_Job_X.png
    Uses proven Google Colab extraction logic
    """
    
    def __init__(self, screenshots_dir: str = "data/raw/linkedin_screenshots"):
        self.screenshots_dir = Path(screenshots_dir)
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])
    
    # -------------------- Job Card Extraction --------------------
    def clean_job_card_lines(self, lines):
        """Now i am cleaning job card lines by filtering out unwanted text"""
        filtered = []
        for l in lines:
            s = l.strip()
            if s and 'actively hiring' not in s.lower():
                filtered.append(s)
        return filtered

    def is_logo_abbreviation(self, text):
        """Now i am checking if text is a logo abbreviation"""
        # Now i am only including clear, unambiguous logo abbreviations
        logos = ['LZD', '0', 'EET', 'ET', 'X', 'Y']  # Removed 'N' as it's too generic
        return text.upper() in logos
    
    def _clean_lines(self, ocr_lines):
        """
        Now i am cleaning OCR lines by removing ignored words and artifacts
        Based on common OCR preprocessing techniques for job posting data by me
        """
        IGNORED_WORDS = {
            'ads', 'promoted', 'actively hiring', 'apply', '', '•', 'new', 'featured'
        }
        
        def ignore_line(line):
            l = line.strip().lower()
            if l in IGNORED_WORDS:
                return True
            # likely logo artifacts: single letter or 1-2 character lines not words
            if len(l) < 2:
                return True
            if re.match(r'^[A-Za-z]\+?$', l):
                return True 
            # skipping line which are just numbers or symbols
            if re.match(r'^[^a-zA-Z]+$', l):
                return True
            return False
        
        # Removing 'Actively Hiring', logo artifacts, etc.
        return [l for l in ocr_lines if not ignore_line(l)]

    def _extract_job_card_enhanced(self, cleaned):
        """
        Now i am enhancing job card extraction using improved heuristics
        Implements common text parsing patterns for structured job posting data by me
        """
        # Heuristic: first = role, last = posted_time, somewhere = location/company
        result = {
            'role': '',
            'company': '',
            'location': '',
            'posted_time': ''
        }
        if not cleaned or len(cleaned) < 3:
            return result

        # Now i am saying that posted time is usually last
        result['posted_time'] = cleaned[-1]

        # Now i am finding a fragment with a city/state pattern, e.g. "Boston, MA", "San Diego, CA"
        location_pattern = r'\b[A-Za-z ]+[,;]\s*[A-Z]{2,}'
        loc_idx = None
        for i, l in enumerate(cleaned):
            if re.search(location_pattern, l):
                result['location'] = l
                loc_idx = i
                break

        # Now i am removing posted_time and location from the list
        body = []
        for i, l in enumerate(cleaned):
            if i == len(cleaned) - 1:
                continue
            if loc_idx is not None and i == loc_idx:
                continue
            body.append(l)
        
        # 'body' should contain role and company
        if body:
            result['role'] = body[0]
            if len(body) > 1:
                result['company'] = body[1]
        return result

    def _fix_parsing_issues(self, company, location, filtered_lines):
        """
        Now i am fixing common parsing issues using enhanced extraction as fallback
        Enhanced parsing logic provided by me and integrated
        to improve accuracy of job card data extraction by me
        """
        # Now i am trying the enhanced extraction method
        cleaned_lines = self._clean_lines(filtered_lines)
        enhanced_result = self._extract_job_card_enhanced(cleaned_lines)
        
        # Now i am using enhanced results if they look better than current ones
        if enhanced_result['company'] and company in ['Ads', 'N', 'ans+', '']:
            company = enhanced_result['company']
        
        if enhanced_result['location'] and location in ['N', 'Ads', 'ans+', '']:
            location = enhanced_result['location']
            
        return company, location

    def extract_job_card(self, image_path):
        """Now i am extracting job card information using proven Google Colab logic"""
        try:
            text_lines = self.reader.readtext(image_path, detail=0)
            print(f"DEBUG OCR raw lines: {text_lines}")
            filtered = self.clean_job_card_lines(text_lines)
            print(f"DEBUG Filtered lines: {filtered}")

            role = filtered[0] if len(filtered) > 0 else ""
            company = ""
            location = ""
            posted_time = ""

            if len(filtered) > 1:
                if self.is_logo_abbreviation(filtered[1]) and len(filtered) > 2:
                    company = filtered[2]
                    location = filtered[3] if len(filtered) > 3 else ""
                else:
                    company = filtered[1]
                    location = filtered[2] if len(filtered) > 2 else ""

            for l in reversed(filtered):
                if 'ago' in l.lower() or 'just now' in l.lower():
                    posted_time = l
                    break
            
            # Now i am doing post-processing fixes for common parsing issues
            company, location = self._fix_parsing_issues(company, location, filtered)
            
            # Let's say if we have "United States" in the filtered lines, use it as location
            for line in filtered:
                if 'united states' in line.lower() or 'usa' in line.lower():
                    location = line
                    break

            card_data = {
                "role": role,
                "company": company,
                "location": location,
                "posted_time": posted_time
            }
            print(f"DEBUG Parsed card data after cleaning: {card_data}")
            return card_data
            
        except Exception as e:
            logger.error(f"Error extracting job card from {image_path}: {e}")
            return {"role": "", "company": "", "location": "", "posted_time": ""}
    
    # -------------------- Job Attributes Extraction --------------------
    def extract_attributes(self, image_path):
        """Now i am extracting job attributes using proven Google Colab logic"""
        try:
            results = self.reader.readtext(image_path, detail=0)
            attributes = {
                "Seniority_level": "",
                "Employment_type": "",
                "Job_Function": "",
                "Industries": ""
            }
            for idx, line in enumerate(results):
                if "Seniority level" in line:
                    attributes["Seniority_level"] = results[idx+1] if idx+1 < len(results) else ""
                if "Employment type" in line:
                    attributes["Employment_type"] = results[idx+1] if idx+1 < len(results) else ""
                if "Job function" in line:
                    attributes["Job_Function"] = results[idx+1] if idx+1 < len(results) else ""
                if "Industries" in line:
                    attributes["Industries"] = results[idx+1] if idx+1 < len(results) else ""
            return attributes
            
        except Exception as e:
            logger.error(f"Error extracting attributes from {image_path}: {e}")
            return {"Seniority_level": "", "Employment_type": "", "Job_Function": "", "Industries": ""}
    
    # -------------------- Pay Info Extraction --------------------
    def extract_pay_range(self, image_path):
        """Now i am extracting pay range using proven Google Colab logic"""
        try:
            text_lines = self.reader.readtext(image_path, detail=0)
            print(f"DEBUG OCR lines from pay info image {image_path}: {text_lines}")

            filtered_lines = [line for line in text_lines if not line.lower().startswith('ppay')]
            normalized_lines = []
            for line in filtered_lines:
                line = re.sub(r'(?<!\$)[Ss]', '$', line)  # S or s -> $ but not if already $
                line = re.sub(r'0[Oo]', '00', line)
                line = re.sub(r'O0', '00', line)
                line = line.replace('oo', '00')

                if len(line) > 0 and not line.startswith('$') and line[0] in ['5', '8']:
                    line = '$' + line[1:]
                normalized_lines.append(line.strip())

            pay_parts = []
            pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?/[a-z]+'
            for line in normalized_lines:
                matches = re.findall(pattern, line.replace(' ', ''))
                pay_parts.extend(matches)

            if pay_parts:
                pay_range_str = ' - '.join(pay_parts)
                print(f"DEBUG Constructed pay range: {pay_range_str}")
                return pay_range_str
            else:
                print(f"DEBUG No valid pay range found in {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting pay range from {image_path}: {e}")
            return None
    
    def extract_pay_from_raw_data(self, job_id: int, screenshots_dir: str) -> str:
        """Now i am extracting pay range from raw LinkedIn data JSON file"""
        try:
            # Now i am looking for the jobs_data JSON file in the screenshots directory
            json_files = [f for f in os.listdir(screenshots_dir) if f.startswith('jobs_data_') and f.endswith('.json')]
            if not json_files:
                return None
            
            json_file = os.path.join(screenshots_dir, json_files[0])
            with open(json_file, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            
            # Now i am finding the job data for this job_id
            for job_data in jobs_data:
                if job_data.get('job_id') == job_id and 'pay_info' in job_data:
                    pay_text = job_data['pay_info'].get('raw_text', '')
                    if pay_text:
                        # Now i am extracting pay range using regex
                        import re
                        pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?/[a-z]+'
                        matches = re.findall(pattern, pay_text)
                        if matches:
                            return ' - '.join(matches)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting pay from raw data for job {job_id}: {e}")
            return None
    
    # -------------------- Main Processing Methods --------------------
    
    def process_job_screenshots(self, job_id: int, screenshots_dir: str) -> Dict[str, any]:
        """Now i am processing all screenshots for a specific job using proven Google Colab logic"""
        try:
            job_data = {
                "job_id": job_id,
                "role": "",
                "company": "",
                "location": "",
                "posted_time": "",
                "Seniority_level": "",
                "Employment_type": "",
                "Job_Function": "",
                "Industries": "",
                "pay_range": None,
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            # Now i am processing job card
            job_card_path = os.path.join(screenshots_dir, f"job_card_{job_id}.png")
            if os.path.exists(job_card_path):
                card_data = self.extract_job_card(job_card_path)
                job_data.update(card_data)
            
            # Now i am processing job attributes
            attributes_path = os.path.join(screenshots_dir, f"job_attributes_Job_{job_id}.png")
            if os.path.exists(attributes_path):
                attributes_data = self.extract_attributes(attributes_path)
                job_data.update(attributes_data)
            
            # Now i am processing pay info
            pay_path = os.path.join(screenshots_dir, f"pay_info_Job_{job_id}.png")
            if os.path.exists(pay_path):
                pay_range = self.extract_pay_range(pay_path)
                job_data["pay_range"] = pay_range
            else:
                # Trying to extract from raw LinkedIn data if available
                pay_range = self.extract_pay_from_raw_data(job_id, screenshots_dir)
                if pay_range:
                    job_data["pay_range"] = pay_range
            
            return job_data
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            return {"error": str(e)}
    
    def process_all_jobs(self, screenshots_dir: str) -> Dict[str, Dict[str, any]]:
        """Now i am processing all job screenshots in a directory using Google Colab logic"""
        try:
            # Now i am initializing data dictionaries
            job_cards_data = {}
            attributes_data = {}
            payinfo_data = {}
            
            # Now i am processing job cards
            job_card_files = list(Path(screenshots_dir).glob("job_card_*.png"))
            for filename in job_card_files:
                match = re.search(r'job_card_(\d+)', filename.name)
                if match:
                    job_key = match.group(1)
                    path = str(filename)
                    job_cards_data[job_key] = self.extract_job_card(path)
            
            # Now i am processing job attributes
            attributes_files = list(Path(screenshots_dir).glob("job_attributes_Job_*.png"))
            for filename in attributes_files:
                match = re.search(r'job_attributes_Job_(\d+)', filename.name)
                if match:
                    job_key = match.group(1)
                    path = str(filename)
                    attributes_data[job_key] = self.extract_attributes(path)
            
            # Now i am processing pay info
            payinfo_files = list(Path(screenshots_dir).glob("pay_info_Job_*.png"))
            for filename in payinfo_files:
                match = re.search(r'pay_info_Job_(\d+)', filename.name)
                if match:
                    job_key = match.group(1)
                    path = str(filename)
                    pay_range = self.extract_pay_range(path)
                    payinfo_data[job_key] = pay_range
            
            # Now i am combining all data
            all_keys = set(job_cards_data.keys()) | set(attributes_data.keys()) | set(payinfo_data.keys())
            combined_jobs = {}
            
            for key in all_keys:
                new_key = f"Job_{key}"
                combined_jobs[new_key] = {
                    "role": job_cards_data.get(key, {}).get("role", ""),
                    "company": job_cards_data.get(key, {}).get("company", ""),
                    "location": job_cards_data.get(key, {}).get("location", ""),
                    "posted_time": job_cards_data.get(key, {}).get("posted_time", ""),
                    "Seniority_level": attributes_data.get(key, {}).get("Seniority_level", ""),
                    "Employment_type": attributes_data.get(key, {}).get("Employment_type", ""),
                    "Job_Function": attributes_data.get(key, {}).get("Job_Function", ""),
                    "Industries": attributes_data.get(key, {}).get("Industries", ""),
                    "pay_range": payinfo_data.get(key, None)
                }
            
            return combined_jobs
            
        except Exception as e:
            logger.error(f"Error processing all jobs: {e}")
            return {}
    
    def save_extracted_data(self, jobs_data: Dict[str, Dict[str, any]], output_path: str):
        """Now i am saving extracted data to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(jobs_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Extracted data saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving extracted data: {e}")

def find_latest_screenshots_directory():
    """Now i am automatically finding the latest screenshots directory"""
    base_dir = Path("data/raw")
    
    # Now i am looking for linkedin_screenshots directories
    linkedin_dirs = list(base_dir.glob("linkedin_screenshots*"))
    
    if not linkedin_dirs:
        # Now i am falling back to default
        return "data/raw/linkedin_screenshots"
    
    # Now i am finding the most recent directory
    latest_dir = max(linkedin_dirs, key=lambda x: x.stat().st_mtime)
    
    # Now i am checking if it has date-based subfolders
    if latest_dir.is_dir():
        date_folders = [d for d in latest_dir.iterdir() if d.is_dir() and d.name.startswith('2025')]
        if date_folders:
            # Now i am using the latest date folder
            latest_date_folder = max(date_folders, key=lambda x: x.stat().st_mtime)
            return str(latest_date_folder)
        else:
            # Now i am using the main directory
            return str(latest_dir)
    
    return str(latest_dir)

def main():
    """Now i am running the main function to run text extraction using Google Colab logic"""
    print("Enhanced Job Text Extraction (Google Colab Logic)")
    print("=" * 60)
    
    # Now i am automatically finding the latest screenshots directory
    screenshots_dir = find_latest_screenshots_directory()
    print(f"Auto-detected screenshots directory: {screenshots_dir}")
    
    # Now i am initializing analyzer with the correct directory
    analyzer = EnhancedJobAnalyzer(screenshots_dir)
    
    # Now i am processing all jobs
    print(f"Processing jobs from: {screenshots_dir}")
    jobs_data = analyzer.process_all_jobs(screenshots_dir)
    
    if jobs_data:
        # Now i am saving extracted data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/processed/all_jobs_final_{timestamp}.json"
        analyzer.save_extracted_data(jobs_data, output_path)
        
        print(f"\n Successfully extracted text from {len(jobs_data)} jobs!")
        print(f" Data saved to: {output_path}")
        
        # Now i am showing the summary
        print(f"\n EXTRACTION SUMMARY:")
        jobs_with_roles = sum(1 for job in jobs_data.values() if job.get('role'))
        jobs_with_companies = sum(1 for job in jobs_data.values() if job.get('company'))
        jobs_with_attributes = sum(1 for job in jobs_data.values() if job.get('Seniority_level'))
        jobs_with_pay = sum(1 for job in jobs_data.values() if job.get('pay_range'))
        
        print(f"   • Total jobs processed: {len(jobs_data)}")
        print(f"   • Jobs with roles: {jobs_with_roles}")
        print(f"   • Jobs with companies: {jobs_with_companies}")
        print(f"   • Jobs with attributes: {jobs_with_attributes}")
        print(f"   • Jobs with pay info: {jobs_with_pay}")
        
        # Now i am showing the sample data
        print(f"\n SAMPLE EXTRACTED DATA:")
        print("=" * 60)
        for job_key, job_data in list(jobs_data.items())[:2]:  # Now i am showing the first 2 jobs
            print(f"\n {job_key}:")
            print(f"    Role: {job_data.get('role', 'N/A')}")
            print(f"    Company: {job_data.get('company', 'N/A')}")
            print(f"    Location: {job_data.get('location', 'N/A')}")
            print(f"    Seniority: {job_data.get('Seniority_level', 'N/A')}")
            print(f"    Pay Range: {job_data.get('pay_range', 'N/A')}")
        
    else:
        print(" No job data extracted")

if __name__ == "__main__":
    main()
