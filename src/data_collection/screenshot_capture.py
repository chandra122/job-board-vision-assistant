#!/usr/bin/env python3
"""
Now i am creating Screenshot Capture for LinkedIn Job Boards
I developed this module to handle capturing screenshots from LinkedIn job search pages
with visual debugging and multiple job search terms for my computer vision project.
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Import our configuration
from src.utils.config import (
    RAW_SCREENSHOTS_DIR, LOGS_DIR, LINKEDIN_URLS, BROWSER_SETTINGS, 
    DEBUG, LOGGING
)

class LinkedInScreenshotCapture:
    """
    Now i am creating this class for LinkedIn screenshot capture with visual debugging
    to automate job board data collection for my AI/ML project.
    """
    
    def __init__(self):
        """I initialized the screenshot capture system for my job board automation"""
        self.setup_logging()
        self.driver = None
        self.captured_screenshots = []
        self.driver = self.get_driver()
        
    def setup_logging(self):
        """I set up the logging for screenshot capture to track my automation process"""
        # Now i am ensuring logs directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        log_file = os.path.join(LOGS_DIR, f"screenshot_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=getattr(logging, LOGGING['level']),
            format=LOGGING['format'],
            handlers=[
                logging.FileHandler(log_file, encoding=LOGGING['file_encoding']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("LinkedIn Screenshot Capture initialized")
        
    def get_driver(self):
        """Now i am initializing and returning a Chrome WebDriver with visual debugging"""
        try:
            options = ChromeOptions()
            
            # Browser settings
            if BROWSER_SETTINGS['headless']:
                options.add_argument('--headless')
            else:
                # Visual debugging mode
                options.add_argument('--start-maximized')
                options.add_argument('--disable-web-security')
                options.add_argument('--disable-features=VizDisplayCompositor')
            
            # Performance and stability
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-images')  # Faster loading
            
            # Window size
            width, height = BROWSER_SETTINGS['window_size']
            options.add_argument(f'--window-size={width},{height}')
            
            # User agent
            options.add_argument(f'--user-agent={BROWSER_SETTINGS["user_agent"]}')
            
            # Now i am creating driver
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            self.logger.info("Chrome WebDriver initialized successfully")
            return driver
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def capture_linkedin_page(self, job_type, url):
        """
        Now i am capturing a LinkedIn job search page with scrolling and visual debugging
        
        Args:
            job_type (str): Type of job search (e.g., 'data_scientist')
            url (str): LinkedIn job search URL
            
        Returns:
            str: Path to saved screenshot, or None if failed
        """
        try:
            self.logger.info(f"Starting capture for {job_type}: {url}")
            
            # Navigate to the page
            self.driver.get(url)
            self.logger.info(f"Navigated to {url}")
            
            # Wait for page to load
            wait = WebDriverWait(self.driver, BROWSER_SETTINGS['wait_time'])
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Handle LinkedIn sign-in popup
            self.handle_linkedin_popups()
            
            # Scroll to load more jobs
            self.scroll_and_load_jobs()
            
            # Now i am taking screenshot
            screenshot_path = self.take_screenshot(job_type)
            
            if screenshot_path:
                self.captured_screenshots.append(screenshot_path)
                self.logger.info(f"Screenshot saved: {screenshot_path}")
                
                # Visual debugging
                if DEBUG['show_images']:
                    self.show_screenshot_preview(screenshot_path)
                
                return screenshot_path
            else:
                self.logger.error(f"Failed to capture screenshot for {job_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing {job_type}: {e}")
            return None
    
    def handle_linkedin_popups(self):
        """Handle LinkedIn sign-in popups and other modals"""
        try:
            self.logger.info("Checking for LinkedIn popups...")
            
            # Wait a moment for popups to appear
            time.sleep(2)
            
            # Now i am trying to close various types of popups
            popup_selectors = [
                # Sign-in modal popups
                "button[aria-label='Dismiss']",
                "button[data-control-name='dismiss']",
                ".sign-in-modal__dismiss",
                "button[class*='dismiss']",
                "button[class*='close']",
                ".modal__dismiss",
                "[data-test-id='dismiss']",
                
                # App promotion popups
                "button[aria-label='Close']",
                "button[class*='close']",
                ".app-promotion-modal__dismiss",
                ".app-promotion-modal__close",
                "button[data-control-name='dismiss_app_promotion']",
                
                # Generic modal close buttons
                "button[aria-label='Close modal']",
                "button[class*='modal-close']",
                ".modal-close-button",
                "button[class*='close-button']",
                
                # X buttons in various formats
                "button[class*='x']",
                "button[class*='X']",
                ".close-x",
                ".dismiss-x"
            ]
            
            popups_closed = 0
            
            for selector in popup_selectors:
                try:
                    popup = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if popup.is_displayed():
                        popup.click()
                        self.logger.info(f"Closed popup using selector: {selector}")
                        popups_closed += 1
                        time.sleep(1)
                except:
                    continue
            
            # Now i am trying to press Escape key multiple times as fallback
            try:
                from selenium.webdriver.common.keys import Keys
                for _ in range(3):  # Try multiple times
                    self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                    time.sleep(0.5)
                self.logger.info("Pressed Escape key multiple times to close popups")
            except:
                pass
            
            # Now i am doing an additional check for any remaining modals
            try:
                # Now i am looking for any modal or popup containers
                modal_containers = [
                    ".modal",
                    ".popup",
                    ".overlay",
                    "[role='dialog']",
                    "[aria-modal='true']"
                ]
                
                for container_selector in modal_containers:
                    try:
                        modals = self.driver.find_elements(By.CSS_SELECTOR, container_selector)
                        for modal in modals:
                            if modal.is_displayed():
                                # Now i am trying to find close button within the modal
                                close_buttons = modal.find_elements(By.CSS_SELECTOR, "button, [role='button']")
                                for btn in close_buttons:
                                    if btn.is_displayed() and any(keyword in btn.get_attribute("class") or "" for keyword in ["close", "dismiss", "x"]):
                                        btn.click()
                                        self.logger.info("Closed modal using generic close button")
                                        popups_closed += 1
                                        time.sleep(1)
                                        break
                    except:
                        continue
                        
            except Exception as e:
                self.logger.warning(f"Error in additional modal check: {e}")
                
            self.logger.info(f"Popup handling completed - {popups_closed} popups closed")
            
        except Exception as e:
            self.logger.warning(f"Error handling popups: {e}")
    
    def scroll_and_load_jobs(self):
        """Now i am scrolling the page to load more job postings"""
        try:
            self.logger.info("Scrolling to load more jobs...")
            
            # Now i am getting initial page height
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scrolls = 0
            max_scrolls = BROWSER_SETTINGS['max_scrolls']
            
            while scrolls < max_scrolls:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Now i am waiting for new content to load
                time.sleep(BROWSER_SETTINGS['scroll_pause'])
                
                # Now i am checking if new content loaded
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                
                if new_height == last_height:
                    self.logger.info("No more content to load")
                    break
                
                last_height = new_height
                scrolls += 1
                self.logger.info(f"Scroll {scrolls}/{max_scrolls} completed")
            
            # Now i am scrolling back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Error during scrolling: {e}")
    
    def close_driver(self):
        """Now i am closing the WebDriver and cleaning up resources"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("WebDriver closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing WebDriver: {e}")
    
    def take_screenshot(self, job_type):
        """Now i am taking a full-page screenshot with date-based organization"""
        try:
            # Now i am creating date-based directory structure
            today = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(RAW_SCREENSHOTS_DIR, today)
            os.makedirs(date_dir, exist_ok=True)
            
            # Now i am creating filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"full_{job_type}_{timestamp}.png"
            filepath = os.path.join(date_dir, filename)
            
            # Now i am taking screenshot
            self.driver.save_screenshot(str(filepath))
            
            # Now i am verifying screenshot was saved
            if Path(filepath).exists() and Path(filepath).stat().st_size > 0:
                self.logger.info(f"Screenshot saved: {filepath}")
                return str(filepath)
            else:
                self.logger.error("Screenshot file is empty or not created")
                return None
                
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            return None
    
    def show_screenshot_preview(self, screenshot_path):
        """Now i am showing a preview of the captured screenshot for visual debugging"""
        try:
            if DEBUG['show_images']:
                # Now i am loading and displaying the image
                img = cv2.imread(screenshot_path)
                if img is not None:
                    # Now i am resizing for display if too large
                    height, width = img.shape[:2]
                    if width > 1200:
                        scale = 1200 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = cv2.resize(img, (new_width, new_height))
                    
                    # Now i am displaying the image
                    cv2.imshow(f"LinkedIn Screenshot Preview", img)
                    cv2.waitKey(2000)  # Show for 2 seconds
                    cv2.destroyAllWindows()
                    
                    self.logger.info("Screenshot preview displayed")
                else:
                    self.logger.warning("Could not load screenshot for preview")
                    
        except Exception as e:
            self.logger.error(f"Error showing screenshot preview: {e}")
    
    def capture_all_job_types(self):
        """Now i am capturing screenshots for all configured job types"""
        try:
            self.logger.info("Starting capture for all job types...")
            
            # Now i am initializing driver
            self.driver = self.get_driver()
            
            # Now i am capturing each job type
            for job_type, url in LINKEDIN_URLS.items():
                self.logger.info(f"Capturing {job_type}...")
                screenshot_path = self.capture_linkedin_page(job_type, url)
                
                if screenshot_path:
                    self.logger.info(f" {job_type}: {screenshot_path}")
                else:
                    self.logger.error(f" {job_type}: Failed to capture")
                
                # Now i am doing a small delay between captures
                time.sleep(2)
            
            self.logger.info(f"Capture completed. Total screenshots: {len(self.captured_screenshots)}")
            return self.captured_screenshots
            
        except Exception as e:
            self.logger.error(f"Error in capture_all_job_types: {e}")
            return []
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("WebDriver closed")
    
    def capture_single_job_type(self, job_type):
        """Now i am capturing screenshot for a single job type"""
        try:
            if job_type not in LINKEDIN_URLS:
                self.logger.error(f"Unknown job type: {job_type}")
                return None
            
            url = LINKEDIN_URLS[job_type]
            self.logger.info(f"Capturing single job type: {job_type}")
            
            # Now i am initializing driver
            self.driver = self.get_driver()
            
            # Now i am capturing the page
            screenshot_path = self.capture_linkedin_page(job_type, url)
            
            return screenshot_path
            
        except Exception as e:
            self.logger.error(f"Error in capture_single_job_type: {e}")
            return None
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("WebDriver closed")


def main():
    """Main function for testing the screenshot capture"""
    print(" LinkedIn Screenshot Capture - Phase 1 Module 1")
    print("=" * 50)
    
    # Now i am creating screenshot capture instance
    capture = LinkedInScreenshotCapture()
    
    # Now i am testing with a single job type first
    print("Testing with 'data_scientist' job type...")
    screenshot_path = capture.capture_single_job_type('data_scientist')
    
    if screenshot_path:
        print(f" Screenshot captured successfully: {screenshot_path}")
        print(" (Screenshot Capture) is working!")
    else:
        print(" Screenshot capture failed")
        print(" Check the logs for error details")
    
    print("\n Module 1 testing completed!")


if __name__ == "__main__":
    main()
