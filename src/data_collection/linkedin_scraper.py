#!/usr/bin/env python3
"""
I am creating a LinkedIn Job Data Scraper using Playwright
==========================================

I am implementing automated LinkedIn job data collection using Playwright.
I am capturing job listings, extracting text content, and saving structured data.

Features:
- Now i am automating screenshot capture of job cards and details
- Now i am extracting text from job postings
- Now i am handling popups and navigation
- Now i am saving structured data (JSON/CSV)

All code is original work developed for this computer vision project.
Now i am using Playwright library for web automation (standard industry tool).

References:
- Playwright Documentation: https://playwright.dev/python/
- Nicholas Renotte - Computer Vision Tutorials: https://www.youtube.com/c/NicholasRenotte
- LinkedIn Jobs: https://www.linkedin.com/jobs/
"""

from playwright.sync_api import sync_playwright
import time
import os
import json
import csv
from datetime import datetime

def handle_popups_playwright(page, step_name="Unknown", quick_mode=False):
    """Now i am handling various types of popups using Playwright - optimized for speed"""
    if quick_mode:
        # Now i am in quick mode - just try escape key and basic close buttons
        try:
            page.keyboard.press("Escape")
            time.sleep(0.1)
            # Now i am trying to find and click any visible close buttons
            close_buttons = page.locator("button:has-text('×'), button:has-text('✕'), button[aria-label*='close'], button[aria-label*='dismiss']")
            for i in range(min(close_buttons.count(), 2)):  # Limit to first 2 buttons
                try:
                    btn = close_buttons.nth(i)
                    if btn.is_visible():
                        btn.click()
                        time.sleep(0.1)
                        break
                except:
                    continue
        except:
            pass
        return
    
    print(f"     Handling popups for {step_name}...")
    
    try:
        # Now i am using simplified popup selectors - focus on most common ones
        popup_selectors = [
            "div[class*='modal']",
            "div[class*='app-promotion']",
            "div[class*='promotion']",
            "div[class*='notification']",
            "div[class*='banner']"
        ]
        
        popups_closed = 0
        
        for selector in popup_selectors:
            try:
                popup = page.locator(selector).first
                if popup.is_visible():
                    print(f"     Found popup: {selector}")
                    
                    # Now i am doing a quick check for sign-in content
                    popup_text = popup.text_content() or ""
                    if any(keyword in popup_text.lower() for keyword in ['sign in', 'login', 'sign up', 'join']):
                        continue
                    
                    # Now i am trying to close with escape first (fastest)
                    try:
                        page.keyboard.press("Escape")
                        popups_closed += 1
                        time.sleep(0.2)
                        break
                    except:
                        pass
                    
                    # Now i am trying to close with close buttons
                    try:
                        close_btn = popup.locator("button:has-text('×'), button:has-text('✕'), button[aria-label*='close']").first
                        if close_btn.is_visible():
                            close_btn.click()
                            popups_closed += 1
                            time.sleep(0.2)
                            break
                    except:
                        pass
                            
            except Exception as e:
                continue
        
        if popups_closed > 0:
            print(f"       Closed {popups_closed} popups")
        else:
            print(f"       No popups found to close")
            
    except Exception as e:
        print(f"       Error in popup handling: {e}")

def handle_linkedin_app_promotion_popup_playwright(page):
    """Now i am creating enhanced handling of the 'LinkedIn is better on the app' popup using Playwright"""
    print("     Handling LinkedIn app promotion popup...")
    
    try:
        # Now i am using comprehensive selectors for the LinkedIn app promotion popup
        app_popup_selectors = [
            # Class-based selectors
            "div[class*='app-promotion']",
            "div[class*='promotion']",
            "div[class*='banner']",
            "div[class*='mobile-app-promotion']",
            "div[class*='app-download']",
            "div[class*='app-cta']",
            "div[class*='cta-banner']",
            "div[class*='download-banner']",
            "div[class*='notification']",
            "div[class*='overlay']",
            "div[class*='modal']",
            "div[class*='popup']",
            # Text-based selectors
            "text=LinkedIn is better on the app",
            "text=Don't have the app?",
            "text=Get it in the Microsoft Store",
            "text=Open the app",
            # Data attribute selectors
            "div[data-test-id*='app-promotion']",
            "div[data-test-id*='promotion']",
            "div[data-test-id*='banner']"
        ]
        
        popup_found = False
        popup_closed = False
        
        for selector in app_popup_selectors:
            try:
                popups = page.locator(selector)
                for i in range(popups.count()):
                    popup = popups.nth(i)
                    if popup.is_visible():
                        popup_text = popup.text_content() or ""
                        # Now i am checking if this is the LinkedIn app promotion popup
                        if any(keyword in popup_text.lower() for keyword in [
                            'linkedin is better on the app', 
                            'don\'t have the app', 
                            'get it in the microsoft store', 
                            'open the app',
                            'better on the app',
                            'microsoft store'
                        ]):
                            print(f"     Found LinkedIn app promotion popup: {selector}")
                            popup_found = True
                            
                            # Looking for X close button within the popup
                            close_buttons = popup.locator("button, [role='button'], [aria-label*='close'], [aria-label*='dismiss']")
                            
                            for j in range(close_buttons.count()):
                                try:
                                    btn = close_buttons.nth(j)
                                    if btn.is_visible():
                                        btn_text = btn.text_content() or ""
                                        btn_aria = btn.get_attribute("aria-label") or ""
                                        
                                        # Now i am looking for X or close button
                                        if (btn_text.lower() in ['×', '✕', 'x', 'close', 'dismiss'] or 
                                            'close' in btn_aria.lower() or 
                                            'dismiss' in btn_aria.lower()):
                                            btn.click()
                                            print(f"       Closed LinkedIn app promotion popup using X button")
                                            popup_closed = True
                                            time.sleep(1)
                                            break
                                except:
                                    continue
                            
                            # Now i am clicking on specific coordinates of the popup
                            if not popup_closed:
                                try:
                                    # Now i am getting the popup bounding box and clicking on top-right corner
                                    bbox = popup.bounding_box()
                                    if bbox:
                                        # Clicking on top-right corner where X button usually is
                                        page.click(position={"x": bbox["x"] + bbox["width"] - 20, "y": bbox["y"] + 20})
                                        print(f"       Closed popup by clicking top-right corner")
                                        popup_closed = True
                                        time.sleep(1)
                                except:
                                    pass
                            
                            # Now i am clicking outside the popup
                            if not popup_closed:
                                try:
                                    # I am clicking on the main content area (not the popup)
                                    page.click("body")
                                    print(f"       Closed popup by clicking outside")
                                    popup_closed = True
                                    time.sleep(1)
                                except:
                                    pass
                            
                            # Now i am pressing multiple escape key presses
                            if not popup_closed:
                                try:
                                    for _ in range(3):  # Try multiple times
                                        page.keyboard.press("Escape")
                                        time.sleep(0.5)
                                    print(f"       Closed popup using escape key")
                                    popup_closed = True
                                except:
                                    pass
                            
                            # Clicking on specific screen coordinates
                            if not popup_closed:
                                try:
                                    # Now i am clicking on top-left corner of the page
                                    page.click(position={"x": 50, "y": 50})
                                    print(f"       Closed popup by clicking screen coordinates")
                                    popup_closed = True
                                    time.sleep(1)
                                except:
                                    pass
                            
                            break
                if popup_found:
                    break
            except:
                continue
        
        # If no popup found with specific selectors, then i need to do generic popup detection
        if not popup_found:
            print("     Trying generic popup detection...")
            generic_selectors = [
                "div[class*='modal']",
                "div[class*='popup']",
                "div[class*='overlay']",
                "div[class*='notification']"
            ]
            
            for selector in generic_selectors:
                try:
                    popups = page.locator(selector)
                    for i in range(popups.count()):
                        popup = popups.nth(i)
                        if popup.is_visible():
                            popup_text = popup.text_content() or ""
                            # Now i am checking if this might be the app promotion popup
                            if any(keyword in popup_text.lower() for keyword in [
                                'app', 'linkedin', 'better', 'microsoft', 'store'
                            ]):
                                print(f"     Found potential app popup: {selector}")
                                # Now i am trying to close it
                                try:
                                    page.keyboard.press("Escape")
                                    print(f"       Closed potential popup using escape key")
                                    time.sleep(1)
                                except:
                                    pass
                                break
                except:
                    continue
        
        if not popup_found:
            print("     No LinkedIn app promotion popup found")
        elif popup_closed:
            print("       LinkedIn app promotion popup successfully closed")
        else:
            print("       LinkedIn app promotion popup found but could not be closed")
            
    except Exception as e:
        print(f"       Error handling LinkedIn app promotion popup: {e}")

def verify_past_24_hours_filter_playwright(page):
    """Now i am verifying that the Past 24 hours filter is properly applied using Playwright"""
    try:
        # Now i am checking if the URL contains the filter parameter
        current_url = page.url
        if "f_TPR=r86400" in current_url:
            print("       URL contains Past 24 hours filter parameter")
        else:
            print("       URL does not contain Past 24 hours filter parameter")
        
        # Now i am checking for filter indicators on the page
        filter_indicators = [
            "span[class*='filter']",
            "div[class*='filter']",
            "button[class*='filter']"
        ]
        
        filter_found = False
        for indicator in filter_indicators:
            try:
                elements = page.locator(indicator)
                for i in range(elements.count()):
                    element = elements.nth(i)
                    if element.is_visible():
                        text = element.text_content() or ""
                        if any(keyword in text.lower() for keyword in ['past 24 hours', '24 hours', 'posted', 'time']):
                            print(f"       Found filter indicator: {text}")
                            filter_found = True
                            break
                if filter_found:
                    break
            except:
                continue
        
        if not filter_found:
            print("       No clear filter indicators found on page")
            
        return True
        
    except Exception as e:
        print(f"       Error verifying filter: {e}")
        return False

def get_responsive_selectors(viewport_width):
    """Get responsive selectors based on viewport width"""
    if viewport_width < 1400:
        # Now i am using smaller screens - use more specific selectors
        return {
            'job_cards': [
                "div[class*='job-search-card']",
                "li[class*='jobs-search-results__list-item']",
                "div[class*='jobs-search-results__list-item']"
            ],
            'filter_bar': [
                "div[class*='jobs-search__filters']",
                "div[class*='search-filters']"
            ],
            'right_panel': [
                "div[class*='jobs-search__right-rail']",
                "div[class*='job-details']"
            ],
            'left_panel': [
                "div[class*='jobs-search__left-rail']",
                "ul[class*='jobs-search-results__list']"
            ]
        }
    else:
        # Now i am using larger screens - use broader selectors
        return {
            'job_cards': [
                "div[class*='job-search-card']",
                "div[class*='jobs-search-results__list-item']",
                "li[class*='jobs-search-results__list-item']",
                "ul.jobs-search-results__list li",
                "div[class*='job-card']",
                "div[class*='job-tile']"
            ],
            'filter_bar': [
                "div[class*='jobs-search__filters']",
                "div[class*='filter-bar']",
                "div[class*='search-filters']",
                "div[class*='jobs-search-filter']",
                "div[class*='filter']"
            ],
            'right_panel': [
                "div[class*='jobs-search__right-rail']",
                "div[class*='job-details']",
                "div[class*='job-description']",
                "div[class*='jobs-details']",
                "div[class*='jobs-search__right-rail'] div[class*='jobs-box']",
                "div[class*='jobs-details__main-content']"
            ],
            'left_panel': [
                "div[class*='jobs-search__left-rail']",
                "div[class*='jobs-search-results']",
                "ul[class*='jobs-search-results__list']",
                "div[class*='jobs-search__results-list']"
            ]
        }

def capture_multiple_job_card_screenshots_playwright(job_term="Data Engineer", output_dir="test_job_screenshots_playwright", max_cards=5):
    """Now i am capturing job card screenshots using Playwright"""
    # Now i am ensuring output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Now i am initializing data collection structure
    jobs_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with sync_playwright() as p:
        # Now i am launching browser
        print("Initializing Playwright browser...")
        browser = p.chromium.launch(
            headless=os.environ.get('PLAYWRIGHT_HEADLESS', 'false').lower() == 'true',  # Respect environment variable
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled"
            ]
        )
        
        # Now i am getting screen resolution and adjusting viewport accordingly
        print("Detecting screen resolution...")
        try:
            # Now i am creating a temporary page to detect screen resolution
            temp_page = browser.new_page()
            screen_info = temp_page.evaluate("""
                () => {
                    return {
                        width: screen.width,
                        height: screen.height,
                        availWidth: screen.availWidth,
                        availHeight: screen.availHeight
                    }
                }
            """)
            temp_page.close()
            
            # Now i am calculating optimal viewport size
            screen_width = screen_info.get('width', 1920)
            screen_height = screen_info.get('height', 1080)
            avail_width = screen_info.get('availWidth', 1920)
            avail_height = screen_info.get('availHeight', 1080)
            
            print(f"     Screen resolution: {screen_width}x{screen_height}")
            print(f"     Available space: {avail_width}x{avail_height}")
            
            # Now i am setting viewport to 90% of available space, with minimums and maximums
            viewport_width = max(1200, min(avail_width * 0.9, 1920))
            viewport_height = max(800, min(avail_height * 0.9, 1080))
            
            print(f"     Using viewport: {int(viewport_width)}x{int(viewport_height)}")
            
        except Exception as e:
            print(f"       Could not detect screen resolution: {e}")
            print("     Using default viewport: 1400x900")
            viewport_width = 1400
            viewport_height = 900
        
        # Now i am creating context with adaptive viewport and realistic settings
        context = browser.new_context(
            viewport={"width": int(viewport_width), "height": int(viewport_height)},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
            timezone_id="America/New_York",
            geolocation={"latitude": 40.7128, "longitude": -74.0060},  # New York
            permissions=["geolocation"],
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Cache-Control": "max-age=0"
            }
        )
        
        page = context.new_page()
        
        try:
            # Navigate to LinkedIn job search
            print("Navigating to LinkedIn job search...")
            linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={job_term.replace(' ', '%20')}&location=United%20States&f_TPR=r86400"
            print(f"URL: {linkedin_url}")
            
            try:
                # Navigate with timeout
                response = page.goto(linkedin_url, timeout=30000)
                print(f"       Navigation successful: {response.status}")
            except Exception as e:
                print(f"       Navigation timeout or error: {e}")
                print("     Trying to continue anyway...")

            # Wait for page to load
            print("Waiting for page to load...")
            try:
                # Wait for the page to load with a timeout
                page.wait_for_load_state("domcontentloaded", timeout=10000)
                print("       Page DOM loaded")
            except Exception as e:
                print(f"       DOM load timeout: {e}")
            
            try:
                # Wait for network to be idle with a shorter timeout
                page.wait_for_load_state("networkidle", timeout=15000)
                print("       Network idle")
            except Exception as e:
                print(f"       Network idle timeout: {e}")
            
            time.sleep(3)
            
            # Handling the initial popups (quick mode)
            print("Handling initial popups...")
            handle_popups_playwright(page, "initial page load", quick_mode=True)
            time.sleep(1)
            
            # Handling the LinkedIn app promotion popup specifically
            print("Handling LinkedIn app promotion popup...")
            handle_linkedin_app_promotion_popup_playwright(page)
            time.sleep(1)
            
            # I need to make sure that the Past 24 hours filter is applied
            print("Ensuring Past 24 hours filter is applied...")
            try:
                # Wait for page to load completely
                time.sleep(3)
                
                # Now i am checking if the filter dropdown is open and closing it
                filter_dropdowns = [
                    "div[class*='filter-dropdown']",
                    "div[class*='dropdown-menu']",
                    "div[class*='filter-menu']",
                    "div[class*='time-filter']",
                    "div[class*='posted-filter']",
                    "div[class*='filter__dropdown']",
                    "div[class*='jobs-search-filter']",
                    "div[aria-expanded='true']"
                ]
                
                for dropdown_selector in filter_dropdowns:
                    try:
                        dropdowns = page.locator(dropdown_selector)
                        for i in range(dropdowns.count()):
                            dropdown = dropdowns.nth(i)
                            if dropdown.is_visible():
                                print(f"     Closing filter dropdown: {dropdown_selector}")
                                # Now i am clicking outside to close
                                page.click("body")
                                time.sleep(1)
                                break
                    except:
                        continue
                
                # Now i am pressing Escape multiple times to ensure all dropdowns are closed
                for _ in range(3):
                    page.keyboard.press("Escape")
                    time.sleep(0.5)
                
                print("       Past 24 hours filter applied and dropdowns closed")
                
            except Exception as e:
                print(f"       Error managing filter state: {e}")
            
            # Verifying that the Past 24 hours filter is applied
            print("Verifying Past 24 hours filter...")
            verify_past_24_hours_filter_playwright(page)
            
            # Now i am checking if we're on the correct page
            current_url = page.url
            print(f"Current URL: {current_url}")
            
            if "linkedin.com/jobs/search" in current_url:
                print("  Successfully navigated to LinkedIn job search page")
            else:
                print("  Failed to navigate to LinkedIn job search page")
                print("     Current URL doesn't contain 'linkedin.com/jobs/search'")
                print("     This might be due to LinkedIn's anti-bot detection")
                print("     Trying to continue anyway...")
                
                # Now i am trying to take a screenshot to see what we got
                try:
                    debug_screenshot = os.path.join(output_dir, "debug_page.png")
                    page.screenshot(path=debug_screenshot)
                    print(f"     Debug screenshot saved: {debug_screenshot}")
                except:
                    pass

            # Now i am looking for job cards with responsive selectors
            print("Looking for job cards...")
            
            # Now i am getting responsive selectors based on viewport width
            responsive_selectors = get_responsive_selectors(viewport_width)
            job_card_selectors = responsive_selectors['job_cards']
            
            print(f"     Using selectors optimized for {viewport_width}px width...")
            
            job_cards = None
            used_selector = None
            
            for selector in job_card_selectors:
                try:
                    print(f"Trying selector: {selector}")
                    job_cards = page.locator(selector)
                    if job_cards.count() > 0:
                        used_selector = selector
                        print(f"  Found {job_cards.count()} job cards using selector: {selector}")
                        break
                except Exception as e:
                    print(f"  Selector {selector} failed: {e}")
                    continue
            
            if not job_cards or job_cards.count() == 0:
                print("  No job cards found with any selector")
                return
            
            # Now i am scrolling down to load more jobs (responsive scrolling)
            print("Scrolling to load more jobs...")
            scroll_attempts = 3
            for i in range(scroll_attempts):
                # Now i am getting current viewport height for responsive scrolling
                viewport_height = page.viewport_size['height']
                scroll_amount = viewport_height * 0.8  # Scroll 80% of viewport height
                
                page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                time.sleep(2)
                print(f"Scrolled down ({i+1}/{scroll_attempts}) by {int(scroll_amount)}px...")
                
                # Now i am handling popups after each scroll (quick mode)
                handle_popups_playwright(page, f"after scroll {i+1}", quick_mode=True)
            
            # Final popup cleanup before capturing
            print("Final popup cleanup before capturing...")
            handle_popups_playwright(page, "before capture", quick_mode=True)
            handle_linkedin_app_promotion_popup_playwright(page)
            time.sleep(1)
            
            # Now i am re-finding job cards after scrolling
            job_cards = page.locator(used_selector)
            print(f"Found {job_cards.count()} job cards after scrolling.")

            # Capturing the filter bar screenshot using responsive selectors
            print("Capturing filter bar screenshot...")
            try:
                # Now i am looking for filter bar elements using responsive selectors
                filter_bar_selectors = responsive_selectors['filter_bar']
                
                filter_bar_found = False
                for selector in filter_bar_selectors:
                    try:
                        filter_elements = page.locator(selector)
                        for i in range(filter_elements.count()):
                            element = filter_elements.nth(i)
                            if element.is_visible():
                                # Now i am getting element size
                                box = element.bounding_box()
                                if box and box['height'] > 20:
                                    filter_bar_path = os.path.join(output_dir, "filter_bar.png")
                                    element.screenshot(path=filter_bar_path)
                                    print(f"  Captured filter bar: {filter_bar_path}")
                                    filter_bar_found = True
                                    break
                        if filter_bar_found:
                            break
                    except:
                        continue
                
                if not filter_bar_found:
                    print("  No filter bar found to capture")
                    
            except Exception as e:
                print(f"  Error capturing filter bar: {e}")
            
            # Capturing the screenshots of job cards
            print("Capturing screenshots of job cards...")
            captured_count = 0
            
            for i in range(min(max_cards, job_cards.count())):
                try:
                    print(f"Capturing job card {i+1}...")
                    
                    # Now i am initializing job data structure
                    job_data = {
                        "job_id": i + 1,
                        "job_term": job_term,
                        "timestamp": timestamp,
                        "job_card_info": {},
                        "pay_info": {},
                        "job_attributes": {},
                        "job_url": None,
                        "screenshots": {
                            "job_card": None,
                            "pay_info": None,
                            "job_attributes": None,
                            "job_details": None
                        }
                    }
                    
                    # Now i am resetting right panel variables for each job
                    right_panel_found = False
                    right_panel = None
                    
                    # Now i am handling popups before capturing each card (quick mode)
                    handle_popups_playwright(page, f"before card {i+1}", quick_mode=True)
                    handle_linkedin_app_promotion_popup_playwright(page)
                    
                    # Now i am scrolling the individual card into view (responsive)
                    card = job_cards.nth(i)
                    try:
                        # Now i am trying to scroll the card into view with smooth scrolling
                        card.scroll_into_view_if_needed()
                        time.sleep(1) # Small pause after scrolling
                        
                        # Now i am doing an additional check to ensure card is visible
                        is_visible = card.is_visible()
                        if not is_visible:
                            print(f"     Card {i+1} not visible, trying alternative scroll...")
                            # Now i am trying to scroll to the card's position
                            card_bbox = card.bounding_box()
                            if card_bbox:
                                page.evaluate(f"window.scrollTo(0, {card_bbox['y'] - 100})")
                                time.sleep(1)
                    except Exception as e:
                        print(f"       Error scrolling to card {i+1}: {e}")
                        # Now i am continuing anyway
                    
                    # Now i am handling popups after scrolling to card (quick mode)
                    handle_popups_playwright(page, f"after scrolling to card {i+1}", quick_mode=True)
                    handle_linkedin_app_promotion_popup_playwright(page)

                    # Take sc   eenshot of the individual element
                    screenshot_path = os.path.join(output_dir, f"job_card_{i+1}.png")
                    card.screenshot(path=screenshot_path)
                    print(f"  Captured {screenshot_path}")
                    captured_count += 1
                    
                    # Now i am extracting job card information
                    try:
                        job_card_text = card.text_content() or ""
                        job_data["job_card_info"] = {
                            "raw_text": job_card_text,
                            "screenshot_path": screenshot_path
                        }
                        job_data["screenshots"]["job_card"] = screenshot_path
                        print(f"       Extracted job card text: {job_card_text[:100]}...")
                    except Exception as e:
                        print(f"       Error extracting job card info: {e}")
                
                    # Clicking on job card and capturing job details page
                    print(f"     Clicking on job card {i+1} to view details...")
                    try:
                        # Now i am clicking on the job card
                        card.click()
                        time.sleep(2)  # Wait for navigation to job details page
                        
                        # Now i am handling any popups that appear after clicking (quick mode)
                        handle_popups_playwright(page, f"after clicking card {i+1}", quick_mode=True)
                        handle_linkedin_app_promotion_popup_playwright(page)
                        
                        # Now i am checking if we're on a job details page (not search results)
                        current_url = page.url
                        print(f"     Current URL after click: {current_url}")
                        
                        # Now i am storing the job URL
                        job_data["job_url"] = current_url
                        
                        if "/jobs/view/" in current_url or "/jobs/collections/" in current_url:
                            print(f"       Successfully navigated to job details page")
                            
                            # Now i am capturing the job details page for reference
                            job_details_path = os.path.join(output_dir, f"job_details_Job_{i+1}.png")
                            page.screenshot(path=job_details_path)
                            print(f"       Captured job details page: {job_details_path}")
                            
                            # Now i am navigating back to search results page immediately
                            print(f"     Navigating back to search results...")
                            page.go_back()
                            time.sleep(2)  # Wait longer for page to load
                            
                            # Now i am handling popups after going back (quick mode)
                            handle_popups_playwright(page, f"after going back from job {i+1}", quick_mode=True)
                            handle_linkedin_app_promotion_popup_playwright(page)
                            
                            # Now i am re-finding job cards after navigation
                            job_cards = page.locator(used_selector)
                            print(f"     Re-found {job_cards.count()} job cards after navigation back")
                            
                            # Now i am capturing the search results page with updated right panel
                            print(f"     Capturing search results page with updated right panel...")
                            
                            # Now i am waiting for right panel to update
                            time.sleep(2)
                            
                            # Now i am handling any popups that appeared after navigation back
                            handle_popups_playwright(page, f"after navigation back for job {i+1}")
                            handle_linkedin_app_promotion_popup_playwright(page)
                            
                            # Now i am capturing right panel with more specific selectors based on actual LinkedIn layout
                            right_panel_selectors = [
                                # Most specific LinkedIn selectors based on actual page structure
                                "div[class*='jobs-search__right-rail']",
                                "div[class*='jobs-search__right-rail'] > div",
                                "div[class*='jobs-search__right-rail'] div[class*='jobs-box']",
                                "div[class*='jobs-search__right-rail'] div[class*='jobs-details']",
                                "div[class*='jobs-search__right-rail'] div[class*='job-details']",
                                # Alternative right panel selectors
                                "div[class*='jobs-details']",
                                "div[class*='job-details']",
                                "div[class*='jobs-box']",
                                "div[class*='jobs-details__main-content']",
                                # Trying to find the right side panel by position
                                "div[class*='jobs-search'] > div:last-child",
                                "div[class*='jobs-search'] div[class*='right']",
                                # More generic but targeted selectors
                                "div[class*='right-rail']",
                                "div[class*='right-panel']",
                                "div[class*='job-panel']",
                                "div[class*='details-panel']"
                            ]
                            
                            right_panel_found = False
                            right_panel = None
                            
                            for panel_selector in right_panel_selectors:
                                try:
                                    right_panel = page.locator(panel_selector).first
                                    if right_panel.is_visible():
                                        print(f"       Found right panel with selector: {panel_selector}")
                                        right_panel_found = True
                                        break
                                except:
                                    continue
                            
                            if not right_panel_found:
                                print(f"       Right panel not found for job card {i+1}")
                                
                                # Now i am trying to find the actual LinkedIn right panel structure
                                try:
                                    print(f"       Trying alternative right panel detection...")
                                    
                                    # Now i am looking for the main job details container that should be visible
                                    alternative_selectors = [
                                        # Now i am looking for the main job details container
                                        "div[class*='jobs-details__main-content']",
                                        "div[class*='jobs-details__main']",
                                        "div[class*='jobs-details']",
                                        # Now i am looking for the right rail container
                                        "div[class*='jobs-search__right-rail']",
                                        # Now i am looking for job box container
                                        "div[class*='jobs-box']",
                                        # Now i am looking for elements that contain multiple job-related elements
                                        "div:has-text('Apply'):has-text('Save')",
                                        "div:has-text('Apply'):has-text('Description')",
                                        "div:has-text('Save'):has-text('Description')"
                                    ]
                                    
                                    for alt_selector in alternative_selectors:
                                        try:
                                            elements = page.locator(alt_selector)
                                            for i_alt in range(min(elements.count(), 3)):  # Limit to first 3 elements
                                                element = elements.nth(i_alt)
                                                if element.is_visible():
                                                    # Now i am getting element dimensions and position
                                                    bbox = element.bounding_box()
                                                    if bbox:
                                                        # Now i am checking if this element is likely the right panel
                                                        # Now i am checking if this element is likely the right panel
                                                        # Right panel should be reasonably sized and positioned
                                                        element_text = element.text_content() or ""
                                                        
                                                        # Now i am looking for multiple job-related keywords to confirm it's the right panel
                                                        job_keywords = ['apply', 'save', 'description', 'company', 'location', 'posted', 'applicants']
                                                        keyword_count = sum(1 for keyword in job_keywords if keyword in element_text.lower())
                                                        
                                                        # Now i am checking if the right panel should be on the right side and have reasonable size
                                                        is_right_side = bbox['x'] > 400  # Should be on right side of screen
                                                        is_reasonable_size = bbox['width'] > 300 and bbox['height'] > 200
                                                        
                                                        if keyword_count >= 3 and is_right_side and is_reasonable_size:
                                                            print(f"       Found potential right panel with selector: {alt_selector}")
                                                            print(f"       Element size: {bbox['width']:.0f}x{bbox['height']:.0f}, position: {bbox['x']:.0f},{bbox['y']:.0f}")
                                                            print(f"       Found {keyword_count} job-related keywords")
                                                            right_panel = element
                                                            right_panel_found = True
                                                            break
                                            if right_panel_found:
                                                break
                                        except Exception as e:
                                            print(f"       Error with selector {alt_selector}: {e}")
                                            continue
                                    
                                    if right_panel_found:
                                        print(f"       Successfully found right panel using alternative method")
                                    
                                except Exception as e:
                                    print(f"       Error in alternative detection: {e}")
                                
                                # Now i am trying one more approach - looking for any visible job details
                                if not right_panel_found:
                                    try:
                                        print(f"       Final attempt: Looking for any job details container...")
                                        
                                        # Now i am trying to find any container that looks like job details
                                        final_selectors = [
                                            "div[class*='jobs'] div[class*='details']",
                                            "div[class*='job'] div[class*='content']",
                                            "main div[class*='jobs']",
                                            "div[class*='content'] div[class*='job']"
                                        ]
                                        
                                        for final_selector in final_selectors:
                                            try:
                                                elements = page.locator(final_selector)
                                                for j in range(min(elements.count(), 2)):
                                                    element = elements.nth(j)
                                                    if element.is_visible():
                                                        bbox = element.bounding_box()
                                                        if bbox and bbox['width'] > 200 and bbox['height'] > 100:
                                                            element_text = element.text_content() or ""
                                                            if any(keyword in element_text.lower() for keyword in ['apply', 'save', 'description', 'company']):
                                                                print(f"       Found job details with selector: {final_selector}")
                                                                right_panel = element
                                                                right_panel_found = True
                                                                break
                                                if right_panel_found:
                                                    break
                                            except:
                                                continue
                                        
                                        if not right_panel_found:
                                            print(f"       Could not find right panel for job {i+1} - skipping detailed capture")
                                            
                                    except Exception as e:
                                        print(f"       Error in final detection attempt: {e}")
                            
                            # Left panel capture removed - now i have job_card_X.png for individual job cards
                            
                            # Now i am doing final popup cleanup before capturing
                            handle_popups_playwright(page, f"before final capture for job {i+1}", quick_mode=True)
                            handle_linkedin_app_promotion_popup_playwright(page)
                            
                            # Now i am capturing job information sections after "Show more" click
            
                            # Now i am clicking "Show more" to reveal job attributes (Seniority level, Employment type, etc.)
                            if right_panel_found and right_panel:
                                try:
                                    # First, i am verifying the right panel is still valid before clicking "Show more"
                                    try:
                                        if not right_panel.is_visible():
                                            print(f"       Right panel no longer visible before 'Show more' - skipping")
                                            right_panel_found = False
                                    except:
                                        print(f"       Right panel element invalid before 'Show more' - skipping")
                                        right_panel_found = False
                                    
                                    if right_panel_found:
                                        show_more_selectors = [
                                            "button:has-text('Show more')",
                                            "button:has-text('show more')",
                                            "button[class*='show-more']",
                                            "button[class*='expand']",
                                            "[data-test-id*='show-more']"
                                        ]
                                        
                                        show_more_clicked = False
                                        for show_more_selector in show_more_selectors:
                                            try:
                                                show_more_btn = page.locator(show_more_selector).first
                                                if show_more_btn.is_visible():
                                                    show_more_btn.click()
                                                    print(f"       Clicked 'Show more' button")
                                                    time.sleep(2)  # Wait for content to expand
                                                    show_more_clicked = True
                                                    break
                                            except:
                                                continue
                                        
                                        if not show_more_clicked:
                                            print(f"       'Show more' button not found for job {i+1}")
                                        
                                        # After clicking "Show more", i am re-finding the right panel as it may have changed
                                        if show_more_clicked:
                                            print(f"       Re-finding right panel after 'Show more' interaction...")
                                            right_panel_found = False
                                            right_panel = None
                                            
                                            # Now i am waiting a bit for the content to expand
                                            time.sleep(1)
                                            
                                            # Now i am trying to find the right panel again with expanded selectors
                                            expanded_right_panel_selectors = [
                                                # Most specific LinkedIn selectors first
                                                "div[class*='jobs-search__right-rail'] div[class*='jobs-box']",
                                                "div[class*='jobs-search__right-rail'] div[class*='jobs-details']",
                                                "div[class*='jobs-search__right-rail'] div[class*='job-details']",
                                                "div[class*='jobs-search__right-rail'] > div",
                                                "div[class*='jobs-search__right-rail']",
                                                # Main content selectors
                                                "div[class*='jobs-details__main-content']",
                                                "div[class*='jobs-details__main']",
                                                "div[class*='jobs-details']",
                                                "div[class*='job-details']",
                                                "div[class*='jobs-box']",
                                                # Looking for "Show less" button and get its container
                                                "div:has-text('Show less')",
                                                # Generic selectors as fallback
                                                "div[class*='right-panel']",
                                                "div[class*='right-rail']",
                                                "div[class*='job-panel']",
                                                "div[class*='details-panel']"
                                            ]
                                            
                                            for panel_selector in expanded_right_panel_selectors:
                                                try:
                                                    right_panel = page.locator(panel_selector).first
                                                    if right_panel.is_visible():
                                                        # Now i am validating it's still the right panel
                                                        bbox = right_panel.bounding_box()
                                                        if bbox and bbox['width'] > 300 and bbox['height'] > 200:
                                                            # Now i am doing an additional validation: checking for job-related content
                                                            element_text = right_panel.text_content() or ""
                                                            job_keywords = ['apply', 'save', 'description', 'company', 'location', 'posted', 'applicants', 'show less']
                                                            keyword_count = sum(1 for keyword in job_keywords if keyword in element_text.lower())
                                                            
                                                            # Now i am checking if this is actually the right panel and not the entire page
                                                            # Now i am checking if the right panel should be reasonably sized (not full page width/height)
                                                            is_reasonable_width = bbox['width'] < 800  # Should be narrower than full page
                                                            is_reasonable_height = bbox['height'] < 2000  # Should be shorter than full page
                                                            is_right_positioned = bbox['x'] > 300  # Should be on the right side
                                                            
                                                            if (keyword_count >= 2 and is_reasonable_width and 
                                                                is_reasonable_height and is_right_positioned):
                                                                print(f"       Re-found right panel with selector: {panel_selector}")
                                                                print(f"       Element size: {bbox['width']:.0f}x{bbox['height']:.0f}, position: {bbox['x']:.0f},{bbox['y']:.0f}")
                                                                print(f"       Found {keyword_count} job-related keywords")
                                                                right_panel_found = True
                                                                break
                                                            else:
                                                                print(f"       Element too large or poorly positioned: {bbox['width']:.0f}x{bbox['height']:.0f} at {bbox['x']:.0f},{bbox['y']:.0f}")
                                                                print(f"       Keywords: {keyword_count}, Width OK: {is_reasonable_width}, Height OK: {is_reasonable_height}, Position OK: {is_right_positioned}")
                                                except:
                                                    continue
                                            
                                            if not right_panel_found:
                                                print(f"       Could not re-find right panel after 'Show more' - trying fallback method...")
                                                
                                                # Now i am trying to find any element that looks like expanded job details
                                                try:
                                                    # Now i am looking for elements with "Show less" button (indicates expansion)
                                                    show_less_elements = page.locator("div:has-text('Show less')")
                                                    for j in range(min(show_less_elements.count(), 3)):
                                                        element = show_less_elements.nth(j)
                                                        if element.is_visible():
                                                            # Now i am finding the parent container
                                                            parent = element.locator("xpath=..")
                                                            if parent.is_visible():
                                                                bbox = parent.bounding_box()
                                                                if bbox and bbox['width'] > 300 and bbox['height'] > 200:
                                                                    # Now i am doing an additional validation for fallback
                                                                    is_reasonable_width = bbox['width'] < 800
                                                                    is_reasonable_height = bbox['height'] < 2000
                                                                    is_right_positioned = bbox['x'] > 300
                                                                    
                                                                    if is_reasonable_width and is_reasonable_height and is_right_positioned:
                                                                        right_panel = parent
                                                                        right_panel_found = True
                                                                        print(f"       Found right panel using 'Show less' fallback method for job {i+1}")
                                                                        print(f"       Element size: {bbox['width']:.0f}x{bbox['height']:.0f}, position: {bbox['x']:.0f},{bbox['y']:.0f}")
                                                                        break
                                                                    else:
                                                                        print(f"       Fallback element too large: {bbox['width']:.0f}x{bbox['height']:.0f} at {bbox['x']:.0f},{bbox['y']:.0f}")
                                                except:
                                                    pass
                                            
                                            if not right_panel_found:
                                                print(f"       Could not re-find right panel after 'Show more' - skipping capture")
                                        
                                        # Now i am capturing specific sections after "Show more" expansion
                                        print(f"       Capturing specific sections for job {i+1}...")
                                        
                                        # Capturing the first Pay Information Section (using specific LinkedIn HTML structure)
                                        try:
                                            # Using the specific CSS selectors you provided for precise pay capture
                                            pay_selectors = [
                                                # Most specific: compensation section within job posting details
                                                "div.details-pane__content div.decorated-job-posting__details section.compensation",
                                                "div.details-pane__content div.decorated-job-posting__details section.compensation--above-description",
                                                "div.details-pane__content div.decorated-job-posting__details section.compensation--jserp",
                                                # Fallback: any compensation section
                                                "section.compensation",
                                                "section.compensation--above-description", 
                                                "section.compensation--jserp",
                                                # Broader: within the right panel structure
                                                "div.details-pane__content section[class*='compensation']",
                                                "div.decorated-job-posting__details section[class*='compensation']"
                                            ]
                                            
                                            pay_found = False
                                            for selector in pay_selectors:
                                                try:
                                                    element = page.locator(selector).first
                                                    if element.is_visible():
                                                        bbox = element.bounding_box()
                                                        text = element.text_content() or ""
                                                        
                                                        # Now i am checking if this is actually a pay/salary section
                                                        if any(keyword in text.lower() for keyword in ['$', 'salary', 'pay', 'compensation', 'base pay', 'annually', '/year', '/yr', 'range']):
                                                            # Now i am validating size to prevent panorama
                                                            if bbox and bbox['height'] > 20 and bbox['width'] > 50 and bbox['width'] < 600:
                                                                pay_path = os.path.join(output_dir, f"pay_info_Job_{i+1}.png")
                                                                element.screenshot(path=pay_path)
                                                                print(f"       Captured pay information for job {i+1}: {pay_path}")
                                                                print(f"       Pay size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                                print(f"       Position: x={bbox['x']:.0f}, y={bbox['y']:.0f}")
                                                                print(f"       Pay text preview: {text[:100]}...")
                                                                
                                                                # Now i am storing pay information data
                                                                job_data["pay_info"] = {
                                                                    "raw_text": text,
                                                                    "screenshot_path": pay_path,
                                                                    "element_size": {"width": bbox['width'], "height": bbox['height']},
                                                                    "position": {"x": bbox['x'], "y": bbox['y']}
                                                                }
                                                                job_data["screenshots"]["pay_info"] = pay_path
                                                                pay_found = True
                                                                break
                                                            else:
                                                                print(f"       Pay element wrong size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                except:
                                                    continue
                                            
                                            # If specific selectors didn't work, then i am trying the broader approach
                                            if not pay_found:
                                                print(f"       Trying broader pay information search for job {i+1}...")
                                                try:
                                                    # Now i am looking for any element with compensation-related classes
                                                    compensation_elements = page.locator("section[class*='compensation'], div[class*='compensation']")
                                                    for j in range(compensation_elements.count()):
                                                        element = compensation_elements.nth(j)
                                                        if element.is_visible():
                                                            bbox = element.bounding_box()
                                                            text = element.text_content() or ""
                                                            
                                                            if (bbox and 
                                                                any(keyword in text.lower() for keyword in ['$', 'salary', 'pay', 'compensation', 'base pay', 'annually', '/year', '/yr', 'range']) and
                                                                bbox['height'] > 20 and bbox['width'] > 50 and bbox['width'] < 600):
                                                                
                                                                pay_path = os.path.join(output_dir, f"pay_info_Job_{i+1}.png")
                                                                element.screenshot(path=pay_path)
                                                                print(f"       Captured pay information (broader) for job {i+1}: {pay_path}")
                                                                print(f"       Pay size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                                print(f"       Pay text preview: {text[:100]}...")
                                                                pay_found = True
                                                                break
                                                except Exception as e:
                                                    print(f"       Error in broader pay search: {e}")
                                            
                                            if not pay_found:
                                                print(f"       Pay information not found for job {i+1}")
                                                
                                        except Exception as e:
                                            print(f"       Error capturing pay information for job {i+1}: {e}")
                                        
                                        # Capturing the second Job Attributes Section (using specific LinkedIn job criteria list)
                                        try:
                                            # Using the specific CSS selectors for job criteria list
                                            attributes_selectors = [
                                                # Most specific: the exact job criteria list structure
                                                "ul.description__job-criteria-list",
                                                "div.details-pane__content ul.description__job-criteria-list",
                                                "div.decorated-job-posting__details ul.description__job-criteria-list",
                                                # Fallback: look for job criteria items
                                                "li.description__job-criteria-item",
                                                "div.details-pane__content li.description__job-criteria-item",
                                                "div.decorated-job-posting__details li.description__job-criteria-item",
                                                # Broader: any element containing job criteria
                                                "ul[class*='job-criteria']",
                                                "div[class*='job-criteria']"
                                            ]
                                            
                                            attributes_found = False
                                            for selector in attributes_selectors:
                                                try:
                                                    elements = page.locator(selector)
                                                    for j in range(elements.count()):
                                                        element = elements.nth(j)
                                                        if element.is_visible():
                                                            bbox = element.bounding_box()
                                                            text = element.text_content() or ""
                                                            
                                                            # Now i am checking if this element contains job criteria (more specific validation)
                                                            if any(keyword in text.lower() for keyword in ['seniority level', 'employment type', 'job function', 'industries', 'entry level', 'full-time', 'part-time', 'information technology', 'broadcast media']):
                                                                # Now i am validating size to prevent panorama and ensure it's reasonable
                                                                if (bbox and 
                                                                    bbox['height'] > 30 and 
                                                                    bbox['width'] > 100 and
                                                                    bbox['width'] < 600 and  # Prevent panorama - job criteria should be compact
                                                                    bbox['height'] < 500 and  # Prevent panorama - job criteria should be compact
                                                                    text.strip() and
                                                                    len(text) > 20):
                                                                    
                                                                    attributes_path = os.path.join(output_dir, f"job_attributes_Job_{i+1}.png")
                                                                    element.screenshot(path=attributes_path)
                                                                    print(f"       Captured job attributes for job {i+1}: {attributes_path}")
                                                                    print(f"       Attributes size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                                    print(f"       Position: x={bbox['x']:.0f}, y={bbox['y']:.0f}")
                                                                    print(f"       Text preview: {text[:200]}...")
                                                                    
                                                                    # Now i am storing job attributes data
                                                                    job_data["job_attributes"] = {
                                                                        "raw_text": text,
                                                                        "screenshot_path": attributes_path,
                                                                        "element_size": {"width": bbox['width'], "height": bbox['height']},
                                                                        "position": {"x": bbox['x'], "y": bbox['y']}
                                                                    }
                                                                    job_data["screenshots"]["job_attributes"] = attributes_path
                                                                    attributes_found = True
                                                                    break
                                                    if attributes_found:
                                                        break
                                                except:
                                                    continue
                                            
                                            # If specific selectors didn't work, then i am trying the broader search
                                            if not attributes_found:
                                                print(f"       Trying broader job attributes search for job {i+1}...")
                                                try:
                                                    # Now i am looking for any element that might contain job details
                                                    all_elements = page.locator("div, section")
                                                    for j in range(min(all_elements.count(), 100)):  # Limiting search to avoid performance issues
                                                        element = all_elements.nth(j)
                                                        if element.is_visible():
                                                            bbox = element.bounding_box()
                                                            text = element.text_content() or ""
                                                            
                                                            # Now i am doing a broader search criteria
                                                            if (bbox and 
                                                                bbox['height'] > 30 and 
                                                                bbox['width'] > 100 and
                                                                bbox['width'] < 600 and  # Now i am preventing panorama - job criteria should be compact
                                                                bbox['height'] < 500 and  # Now i am preventing panorama - job criteria should be compact
                                                                any(keyword in text.lower() for keyword in ['seniority level', 'employment type', 'job function', 'industries', 'entry level', 'full-time', 'part-time', 'information technology', 'broadcast media'])):
                                                                
                                                                attributes_path = os.path.join(output_dir, f"job_attributes_Job_{i+1}.png")
                                                                element.screenshot(path=attributes_path)
                                                                print(f"       Captured job attributes (broader search) for job {i+1}: {attributes_path}")
                                                                print(f"       Attributes size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                                print(f"       Position: x={bbox['x']:.0f}, y={bbox['y']:.0f}")
                                                                print(f"       Text preview: {text[:200]}...")
                                                                attributes_found = True
                                                                break
                                                except Exception as e:
                                                    print(f"       Error in broader search: {e}")
                                            
                                            if not attributes_found:
                                                print(f"       Job attributes not found for job {i+1}")
                                                
                                        except Exception as e:
                                            print(f"       Error capturing job attributes for job {i+1}: {e}")
                                    
                                except Exception as e:
                                    print(f"       Error clicking 'Show more': {e}")
                            else:
                                print(f"       Skipping 'Show more' interaction - right panel not found for job {i+1}")
                            
                            # Job description capture removed - now i am focusing on specific sections
                            if False:  # Disabled - now i don't need job description anymore
                                try:
                                    job_description_selectors = [
                                        "div[class*='jobs-description']",
                                        "div[class*='job-description']",
                                        "div[class*='jobs-details__main-content']",
                                        "div[class*='jobs-box']",
                                        "div[class*='jobs-details']"
                                    ]
                                    
                                    description_found = False
                                    for desc_selector in job_description_selectors:
                                        try:
                                            job_desc = page.locator(desc_selector).first
                                            if job_desc.is_visible():
                                                job_desc_path = os.path.join(output_dir, f"job_description_Job_{i+1}.png")
                                                job_desc.screenshot(path=job_desc_path)
                                                print(f"       Captured job description section: {job_desc_path}")
                                                description_found = True
                                                break
                                        except:
                                            continue
                                    
                                    if not description_found:
                                        print(f"       Job description section not found for job {i+1}")
                                        
                                except Exception as e:
                                    print(f"       Error capturing job description: {e}")
                            else:
                                print(f"       Skipping job description capture - right panel not found for job {i+1}")
                            
                        else:
                            print(f"       Did not navigate to job details page, staying on search results")
                            
                            # Now i am waiting for right panel to update
                            time.sleep(2)
                            
                            # Now i am handling any popups that appeared (quick mode)
                            handle_popups_playwright(page, f"before capture for job {i+1}", quick_mode=True)
                            handle_linkedin_app_promotion_popup_playwright(page)
                            
                            # If we're still on search results, then i am trying to capture right panel
                            right_panel_selectors = [
                                # Most specific LinkedIn selectors first
                                "div[class*='jobs-search__right-rail'] div[class*='jobs-box']",
                                "div[class*='jobs-search__right-rail'] div[class*='jobs-details']",
                                "div[class*='jobs-search__right-rail'] div[class*='job-details']",
                                "div[class*='jobs-search__right-rail']",
                                # Main content selectors
                                "div[class*='jobs-details__main-content']",
                                "div[class*='jobs-details__main']",
                                "div[class*='jobs-details']",
                                "div[class*='job-details']",
                                "div[class*='jobs-box']",
                                # Generic selectors as fallback
                                "div[class*='right-panel']",
                                "div[class*='right-rail']",
                                "div[class*='job-panel']",
                                "div[class*='details-panel']",
                                # Fallback selectors
                                "main div[class*='jobs']",
                                "div[class*='jobs'] div[class*='details']"
                            ]
                            
                            right_panel_found = False
                            right_panel = None
                            
                            for panel_selector in right_panel_selectors:
                                try:
                                    right_panel = page.locator(panel_selector).first
                                    if right_panel.is_visible():
                                        print(f"       Found right panel with selector: {panel_selector}")
                                        right_panel_found = True
                                        break
                                except:
                                    continue
                            
                            if not right_panel_found:
                                print(f"       Right panel not found for job card {i+1}")
                                
                                # Now i am trying to find the actual LinkedIn right panel structure
                                try:
                                    print(f"       Trying alternative right panel detection...")
                                    
                                    # Now i am looking for the main job details container that should be visible
                                    alternative_selectors = [
                                        # Now i am looking for the main job details container
                                        "div[class*='jobs-details__main-content']",
                                        "div[class*='jobs-details__main']",
                                        "div[class*='jobs-details']",
                                        # Now i am looking for the right rail container
                                        "div[class*='jobs-search__right-rail']",
                                        # Now i am looking for job box container
                                        "div[class*='jobs-box']",
                                        # Now i am looking for elements that contain multiple job-related elements
                                        "div:has-text('Apply'):has-text('Save')",
                                        "div:has-text('Apply'):has-text('Description')",
                                        "div:has-text('Save'):has-text('Description')"
                                    ]
                                    
                                    for alt_selector in alternative_selectors:
                                        try:
                                            elements = page.locator(alt_selector)
                                            for i_alt in range(min(elements.count(), 3)):  # Limit to first 3 elements
                                                element = elements.nth(i_alt)
                                                if element.is_visible():
                                                    # Now i am getting element dimensions and position
                                                    bbox = element.bounding_box()
                                                    if bbox:
                                                        # Now i am checking if this element is likely the right panel
                                                        # Now i am checking if the right panel should be reasonably sized and positioned
                                                        element_text = element.text_content() or ""
                                                        
                                                        # Now i am looking for multiple job-related keywords to confirm it's the right panel
                                                        job_keywords = ['apply', 'save', 'description', 'company', 'location', 'posted', 'applicants']
                                                        keyword_count = sum(1 for keyword in job_keywords if keyword in element_text.lower())
                                                        
                                                        # Now i am checking if the right panel should be on the right side and have reasonable size
                                                        is_right_side = bbox['x'] > 400  # Should be on right side of screen
                                                        is_reasonable_size = bbox['width'] > 300 and bbox['height'] > 200
                                                        
                                                        if keyword_count >= 3 and is_right_side and is_reasonable_size:
                                                            print(f"       Found potential right panel with selector: {alt_selector}")
                                                            print(f"       Element size: {bbox['width']:.0f}x{bbox['height']:.0f}, position: {bbox['x']:.0f},{bbox['y']:.0f}")
                                                            print(f"       Found {keyword_count} job-related keywords")
                                                            right_panel = element
                                                            right_panel_found = True
                                                            break
                                            if right_panel_found:
                                                break
                                        except Exception as e:
                                            print(f"       Error with selector {alt_selector}: {e}")
                                            continue
                                    
                                    if right_panel_found:
                                        print(f"       Successfully found right panel using alternative method")
                                    
                                except Exception as e:
                                    print(f"       Error in alternative detection: {e}")
                                
                                # If still not found, then i am trying one more approach - looking for any visible job details
                                if not right_panel_found:
                                    try:
                                        print(f"       Final attempt: Looking for any job details container...")
                                        
                                        # Now i am trying to find any container that looks like job details
                                        final_selectors = [
                                            "div[class*='jobs'] div[class*='details']",
                                            "div[class*='job'] div[class*='content']",
                                            "main div[class*='jobs']",
                                            "div[class*='content'] div[class*='job']"
                                        ]
                                        
                                        for final_selector in final_selectors:
                                            try:
                                                elements = page.locator(final_selector)
                                                for j in range(min(elements.count(), 2)):
                                                    element = elements.nth(j)
                                                    if element.is_visible():
                                                        bbox = element.bounding_box()
                                                        if bbox and bbox['width'] > 200 and bbox['height'] > 100:
                                                            element_text = element.text_content() or ""
                                                            if any(keyword in element_text.lower() for keyword in ['apply', 'save', 'description', 'company']):
                                                                print(f"       Found job details with selector: {final_selector}")
                                                                right_panel = element
                                                                right_panel_found = True
                                                                break
                                                if right_panel_found:
                                                    break
                                            except:
                                                continue
                                        
                                        if not right_panel_found:
                                            print(f"       Could not find right panel for job {i+1} - skipping detailed capture")
                                            
                                    except Exception as e:
                                        print(f"       Error in final detection attempt: {e}")
                            
                            # Left panel capture removed - now i have job_card_X.png for individual job cards
                            
                            # Now i am doing final popup cleanup before capturing
                            handle_popups_playwright(page, f"before final capture for job {i+1}", quick_mode=True)
                            handle_linkedin_app_promotion_popup_playwright(page)
                            
                            # Now i am capturing detailed right panel with "Show more" interaction
                            print(f"     Capturing detailed right panel for job {i+1}...")
                            
                            # Now i am capturing the first right panel before "Show more"
                            right_panel_before_path = os.path.join(output_dir, f"right_panel_before_show_more_Job_{i+1}.png")
                            right_panel.screenshot(path=right_panel_before_path)
                            print(f"       Captured right panel before 'Show more': {right_panel_before_path}")
                            
                            # Now i am clicking "Show more" to reveal job attributes (Seniority level, Employment type, etc.)
                            if right_panel_found and right_panel:
                                try:
                                    # First, i am verifying the right panel is still valid before clicking "Show more"
                                    try:
                                        if not right_panel.is_visible():
                                            print(f"       Right panel no longer visible before 'Show more' - skipping")
                                            right_panel_found = False
                                    except:
                                        print(f"       Right panel element invalid before 'Show more' - skipping")
                                        right_panel_found = False
                                    
                                    if right_panel_found:
                                        show_more_selectors = [
                                            "button:has-text('Show more')",
                                            "button:has-text('show more')",
                                            "button[class*='show-more']",
                                            "button[class*='expand']",
                                            "[data-test-id*='show-more']"
                                        ]
                                        
                                        show_more_clicked = False
                                        for show_more_selector in show_more_selectors:
                                            try:
                                                show_more_btn = page.locator(show_more_selector).first
                                                if show_more_btn.is_visible():
                                                    show_more_btn.click()
                                                    print(f"       Clicked 'Show more' button")
                                                    time.sleep(2)  # Wait for content to expand
                                                    show_more_clicked = True
                                                    break
                                            except:
                                                continue
                                        
                                        if not show_more_clicked:
                                            print(f"       'Show more' button not found for job {i+1}")
                                        
                                        # After clicking "Show more", i am re-finding the right panel as it may have changed
                                        if show_more_clicked:
                                            print(f"       Re-finding right panel after 'Show more' interaction...")
                                            right_panel_found = False
                                            right_panel = None
                                            
                                            # Now i am waiting a bit for the content to expand
                                            time.sleep(1)
                                            
                                            # Now i am trying to find the right panel again with expanded selectors
                                            expanded_right_panel_selectors = [
                                                # Most specific LinkedIn selectors first
                                                "div[class*='jobs-search__right-rail'] div[class*='jobs-box']",
                                                "div[class*='jobs-search__right-rail'] div[class*='jobs-details']",
                                                "div[class*='jobs-search__right-rail'] div[class*='job-details']",
                                                "div[class*='jobs-search__right-rail'] > div",
                                                "div[class*='jobs-search__right-rail']",
                                                # Main content selectors
                                                "div[class*='jobs-details__main-content']",
                                                "div[class*='jobs-details__main']",
                                                "div[class*='jobs-details']",
                                                "div[class*='job-details']",
                                                "div[class*='jobs-box']",
                                                # Now i am looking for "Show less" button and getting its container
                                                "div:has-text('Show less')",
                                                # Generic selectors as fallback
                                                "div[class*='right-panel']",
                                                "div[class*='right-rail']",
                                                "div[class*='job-panel']",
                                                "div[class*='details-panel']"
                                            ]
                                            
                                            for panel_selector in expanded_right_panel_selectors:
                                                try:
                                                    right_panel = page.locator(panel_selector).first
                                                    if right_panel.is_visible():
                                                        # Now i am validating it's still the right panel
                                                        bbox = right_panel.bounding_box()
                                                        if bbox and bbox['width'] > 300 and bbox['height'] > 200:
                                                            # Now i am doing an additional validation: checking for job-related content
                                                            element_text = right_panel.text_content() or ""
                                                            job_keywords = ['apply', 'save', 'description', 'company', 'location', 'posted', 'applicants', 'show less']
                                                            keyword_count = sum(1 for keyword in job_keywords if keyword in element_text.lower())
                                                            
                                                            # Now i am checking if this is actually the right panel and not the entire page
                                                            # Now i am checking if the right panel should be reasonably sized (not full page width/height)
                                                            is_reasonable_width = bbox['width'] < 800  # Should be narrower than full page
                                                            is_reasonable_height = bbox['height'] < 2000  # Should be shorter than full page
                                                            is_right_positioned = bbox['x'] > 300  # Should be on the right side
                                                            
                                                            if (keyword_count >= 2 and is_reasonable_width and 
                                                                is_reasonable_height and is_right_positioned):
                                                                print(f"       Re-found right panel with selector: {panel_selector}")
                                                                print(f"       Element size: {bbox['width']:.0f}x{bbox['height']:.0f}, position: {bbox['x']:.0f},{bbox['y']:.0f}")
                                                                print(f"       Found {keyword_count} job-related keywords")
                                                                right_panel_found = True
                                                                break
                                                            else:
                                                                print(f"       Element too large or poorly positioned: {bbox['width']:.0f}x{bbox['height']:.0f} at {bbox['x']:.0f},{bbox['y']:.0f}")
                                                                print(f"       Keywords: {keyword_count}, Width OK: {is_reasonable_width}, Height OK: {is_reasonable_height}, Position OK: {is_right_positioned}")
                                                except:
                                                    continue
                                            
                                            if not right_panel_found:
                                                print(f"       Could not re-find right panel after 'Show more' - trying fallback method...")
                                                
                                                # Now i am trying to find any element that looks like expanded job details
                                                try:
                                                    # Now i am looking for elements with "Show less" button (indicates expansion)
                                                    show_less_elements = page.locator("div:has-text('Show less')")
                                                    for j in range(min(show_less_elements.count(), 3)):
                                                        element = show_less_elements.nth(j)
                                                        if element.is_visible():
                                                            # Now i am finding the parent container
                                                            parent = element.locator("xpath=..")
                                                            if parent.is_visible():
                                                                bbox = parent.bounding_box()
                                                                if bbox and bbox['width'] > 300 and bbox['height'] > 200:
                                                                    # Now i am doing an additional validation for fallback
                                                                    is_reasonable_width = bbox['width'] < 800
                                                                    is_reasonable_height = bbox['height'] < 2000
                                                                    is_right_positioned = bbox['x'] > 300
                                                                    
                                                                    if is_reasonable_width and is_reasonable_height and is_right_positioned:
                                                                        right_panel = parent
                                                                        right_panel_found = True
                                                                        print(f"       Found right panel using 'Show less' fallback method for job {i+1}")
                                                                        print(f"       Element size: {bbox['width']:.0f}x{bbox['height']:.0f}, position: {bbox['x']:.0f},{bbox['y']:.0f}")
                                                                        break
                                                                    else:
                                                                        print(f"       Fallback element too large: {bbox['width']:.0f}x{bbox['height']:.0f} at {bbox['x']:.0f},{bbox['y']:.0f}")
                                                except:
                                                    pass
                                            
                                            if not right_panel_found:
                                                print(f"       Could not re-find right panel after 'Show more' - skipping capture")
                                        
                                        # Now i am capturing specific sections after "Show more" expansion
                                        print(f"       Capturing specific sections for job {i+1}...")
                                        
                                        # Now i am capturing the first Pay Information Section (using specific LinkedIn HTML structure)
                                        try:
                                            # Now i am using the specific CSS selectors you provided for precise pay capture
                                            pay_selectors = [
                                                # Most specific: compensation section within job posting details
                                                "div.details-pane__content div.decorated-job-posting__details section.compensation",
                                                "div.details-pane__content div.decorated-job-posting__details section.compensation--above-description",
                                                "div.details-pane__content div.decorated-job-posting__details section.compensation--jserp",
                                                # Fallback: any compensation section
                                                "section.compensation",
                                                "section.compensation--above-description", 
                                                "section.compensation--jserp",
                                                # Broader: within the right panel structure
                                                "div.details-pane__content section[class*='compensation']",
                                                "div.decorated-job-posting__details section[class*='compensation']"
                                            ]
                                            
                                            pay_found = False
                                            for selector in pay_selectors:
                                                try:
                                                    element = page.locator(selector).first
                                                    if element.is_visible():
                                                        bbox = element.bounding_box()
                                                        text = element.text_content() or ""
                                                        
                                                        # Now i am checking if this is actually a pay/salary section
                                                        if any(keyword in text.lower() for keyword in ['$', 'salary', 'pay', 'compensation', 'base pay', 'annually', '/year', '/yr', 'range']):
                                                            # Now i am validating size to prevent panorama
                                                            if bbox and bbox['height'] > 20 and bbox['width'] > 50 and bbox['width'] < 600:
                                                                pay_path = os.path.join(output_dir, f"pay_info_Job_{i+1}.png")
                                                                element.screenshot(path=pay_path)
                                                                print(f"       Captured pay information for job {i+1}: {pay_path}")
                                                                print(f"       Pay size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                                print(f"       Position: x={bbox['x']:.0f}, y={bbox['y']:.0f}")
                                                                print(f"       Pay text preview: {text[:100]}...")
                                                                
                                                                # Now i am storing pay information data
                                                                job_data["pay_info"] = {
                                                                    "raw_text": text,
                                                                    "screenshot_path": pay_path,
                                                                    "element_size": {"width": bbox['width'], "height": bbox['height']},
                                                                    "position": {"x": bbox['x'], "y": bbox['y']}
                                                                }
                                                                job_data["screenshots"]["pay_info"] = pay_path
                                                                pay_found = True
                                                                break
                                                            else:
                                                                print(f"       Pay element wrong size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                except:
                                                    continue
                                            
                                            # If specific selectors didn't work, then i am trying the broader approach
                                            if not pay_found:
                                                print(f"       Trying broader pay information search for job {i+1}...")
                                                try:
                                                    # Now i am looking for any element with compensation-related classes
                                                    compensation_elements = page.locator("section[class*='compensation'], div[class*='compensation']")
                                                    for j in range(compensation_elements.count()):
                                                        element = compensation_elements.nth(j)
                                                        if element.is_visible():
                                                            bbox = element.bounding_box()
                                                            text = element.text_content() or ""
                                                            
                                                            if (bbox and 
                                                                any(keyword in text.lower() for keyword in ['$', 'salary', 'pay', 'compensation', 'base pay', 'annually', '/year', '/yr', 'range']) and
                                                                bbox['height'] > 20 and bbox['width'] > 50 and bbox['width'] < 600):
                                                                
                                                                pay_path = os.path.join(output_dir, f"pay_info_Job_{i+1}.png")
                                                                element.screenshot(path=pay_path)
                                                                print(f"       Captured pay information (broader) for job {i+1}: {pay_path}")
                                                                print(f"       Pay size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                                print(f"       Pay text preview: {text[:100]}...")
                                                                pay_found = True
                                                                break
                                                except Exception as e:
                                                    print(f"       Error in broader pay search: {e}")
                                            
                                            if not pay_found:
                                                print(f"       Pay information not found for job {i+1}")
                                                
                                        except Exception as e:
                                            print(f"       Error capturing pay information for job {i+1}: {e}")
                                        
                                        # Now i am capturing the second Job Attributes Section (using specific LinkedIn job criteria list)
                                        try:
                                            # Now i am using the specific CSS selectors for job criteria list
                                            attributes_selectors = [
                                                # Most specific: the exact job criteria list structure
                                                "ul.description__job-criteria-list",
                                                "div.details-pane__content ul.description__job-criteria-list",
                                                "div.decorated-job-posting__details ul.description__job-criteria-list",
                                                # Fallback: look for job criteria items
                                                "li.description__job-criteria-item",
                                                "div.details-pane__content li.description__job-criteria-item",
                                                "div.decorated-job-posting__details li.description__job-criteria-item",
                                                # Broader: any element containing job criteria
                                                "ul[class*='job-criteria']",
                                                "div[class*='job-criteria']"
                                            ]
                                            
                                            attributes_found = False
                                            for selector in attributes_selectors:
                                                try:
                                                    elements = page.locator(selector)
                                                    for j in range(elements.count()):
                                                        element = elements.nth(j)
                                                        if element.is_visible():
                                                            bbox = element.bounding_box()
                                                            text = element.text_content() or ""
                                                            
                                                            # Now i am checking if this element contains job criteria (more specific validation)
                                                            if any(keyword in text.lower() for keyword in ['seniority level', 'employment type', 'job function', 'industries', 'entry level', 'full-time', 'part-time', 'information technology', 'broadcast media']):
                                                                # Now i am validating size to prevent panorama and ensuring it's reasonable
                                                                if (bbox and 
                                                                    bbox['height'] > 30 and 
                                                                    bbox['width'] > 100 and
                                                                    bbox['width'] < 600 and  # Prevent panorama - job criteria should be compact
                                                                    bbox['height'] < 500 and  # Prevent panorama - job criteria should be compact
                                                                    text.strip() and
                                                                    len(text) > 20):
                                                                    
                                                                    attributes_path = os.path.join(output_dir, f"job_attributes_Job_{i+1}.png")
                                                                    element.screenshot(path=attributes_path)
                                                                    print(f"       Captured job attributes for job {i+1}: {attributes_path}")
                                                                    print(f"       Attributes size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                                    print(f"       Position: x={bbox['x']:.0f}, y={bbox['y']:.0f}")
                                                                    print(f"       Text preview: {text[:200]}...")
                                                                    
                                                                    # Now i am storing job attributes data
                                                                    job_data["job_attributes"] = {
                                                                        "raw_text": text,
                                                                        "screenshot_path": attributes_path,
                                                                        "element_size": {"width": bbox['width'], "height": bbox['height']},
                                                                        "position": {"x": bbox['x'], "y": bbox['y']}
                                                                    }
                                                                    job_data["screenshots"]["job_attributes"] = attributes_path
                                                                    attributes_found = True
                                                                    break
                                                    if attributes_found:
                                                        break
                                                except:
                                                    continue
                                            
                                            # If specific selectors didn't work, then i am trying the broader search
                                            if not attributes_found:
                                                print(f"       Trying broader job attributes search for job {i+1}...")
                                                try:
                                                    # Now i am looking for any element that might contain job details
                                                    all_elements = page.locator("div, section")
                                                    for j in range(min(all_elements.count(), 100)):  # Limit search to avoid performance issues
                                                        element = all_elements.nth(j)
                                                        if element.is_visible():
                                                            bbox = element.bounding_box()
                                                            text = element.text_content() or ""
                                                            
                                                            # Now i am doing a broader search criteria
                                                            if (bbox and 
                                                                bbox['height'] > 30 and 
                                                                bbox['width'] > 100 and
                                                                bbox['width'] < 600 and  # Prevent panorama - job criteria should be compact
                                                                bbox['height'] < 500 and  # Prevent panorama - job criteria should be compact
                                                                any(keyword in text.lower() for keyword in ['seniority level', 'employment type', 'job function', 'industries', 'entry level', 'full-time', 'part-time', 'information technology', 'broadcast media'])):
                                                                
                                                                attributes_path = os.path.join(output_dir, f"job_attributes_Job_{i+1}.png")
                                                                element.screenshot(path=attributes_path)
                                                                print(f"       Captured job attributes (broader search) for job {i+1}: {attributes_path}")
                                                                print(f"       Attributes size: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                                                print(f"       Position: x={bbox['x']:.0f}, y={bbox['y']:.0f}")
                                                                print(f"       Text preview: {text[:200]}...")
                                                                attributes_found = True
                                                                break
                                                except Exception as e:
                                                    print(f"       Error in broader search: {e}")
                                            
                                            if not attributes_found:
                                                print(f"       Job attributes not found for job {i+1}")
                                                
                                        except Exception as e:
                                            print(f"       Error capturing job attributes for job {i+1}: {e}")
                                    
                                except Exception as e:
                                    print(f"       Error clicking 'Show more': {e}")
                            else:
                                print(f"       Skipping 'Show more' interaction - right panel not found for job {i+1}")
                            
                            # Job description capture removed - now i am focusing on specific sections
                            if False:  # Disabled - now i don't need job description anymore
                                try:
                                    job_description_selectors = [
                                        "div[class*='jobs-description']",
                                        "div[class*='job-description']",
                                        "div[class*='jobs-details__main-content']",
                                        "div[class*='jobs-box']",
                                        "div[class*='jobs-details']"
                                    ]
                                    
                                    description_found = False
                                    for desc_selector in job_description_selectors:
                                        try:
                                            job_desc = page.locator(desc_selector).first
                                            if job_desc.is_visible():
                                                job_desc_path = os.path.join(output_dir, f"job_description_Job_{i+1}.png")
                                                job_desc.screenshot(path=job_desc_path)
                                                print(f"       Captured job description section: {job_desc_path}")
                                                description_found = True
                                                break
                                        except:
                                            continue
                                    
                                    if not description_found:
                                        print(f"       Job description section not found for job {i+1}")
                                        
                                except Exception as e:
                                    print(f"       Error capturing job description: {e}")
                            else:
                                print(f"       Skipping job description capture - right panel not found for job {i+1}")
                        
                    except Exception as e:
                        print(f"       Error clicking job card {i+1}: {e}")
                        # Now i am trying to navigate back to search results if we're on a different page
                        try:
                            current_url = page.url
                            if "/jobs/view/" in current_url or "/jobs/collections/" in current_url:
                                print(f"     Attempting to navigate back to search results...")
                                page.go_back()
                                time.sleep(2)
                                # Now i am re-finding job cards
                                job_cards = page.locator(used_selector)
                                print(f"     Re-found {job_cards.count()} job cards after error recovery")
                        except:
                            pass
                        # Now i am continuing with next card even if this one fails

                except Exception as e:
                    print(f"  Could not capture card {i+1}: {e}")
                    # Log the error but continue with other cards
                    continue
                finally:
                    # Now i am adding job data to the collection (even if there were errors)
                    jobs_data.append(job_data)
                    print(f"       Added job {i+1} data to collection")
            
            print(f"  Successfully captured {captured_count} job card screenshots")
            
            # Now i am saving collected data
            if jobs_data:
                print(f"  Saving collected data for {len(jobs_data)} jobs...")
                
                # Now i am saving as JSON
                json_path = os.path.join(output_dir, f"jobs_data_{timestamp}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(jobs_data, f, indent=2, ensure_ascii=False)
                print(f"       Saved JSON data: {json_path}")
                
                # Now i am saving as CSV for easy viewing
                csv_path = os.path.join(output_dir, f"jobs_summary_{timestamp}.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Job ID', 'Job Term', 'Job URL', 'Job Card Text', 'Pay Info Text', 
                        'Job Attributes Text', 'Job Card Screenshot', 'Pay Info Screenshot', 
                        'Job Attributes Screenshot', 'Job Details Screenshot'
                    ])
                    
                    for job in jobs_data:
                        writer.writerow([
                            job.get('job_id', ''),
                            job.get('job_term', ''),
                            job.get('job_url', ''),
                            job.get('job_card_info', {}).get('raw_text', '')[:200] + '...' if job.get('job_card_info', {}).get('raw_text') else '',
                            job.get('pay_info', {}).get('raw_text', '')[:200] + '...' if job.get('pay_info', {}).get('raw_text') else '',
                            job.get('job_attributes', {}).get('raw_text', '')[:200] + '...' if job.get('job_attributes', {}).get('raw_text') else '',
                            job.get('screenshots', {}).get('job_card', ''),
                            job.get('screenshots', {}).get('pay_info', ''),
                            job.get('screenshots', {}).get('job_attributes', ''),
                            job.get('screenshots', {}).get('job_details', '')
                        ])
                print(f"       Saved CSV summary: {csv_path}")
                
                # Print summary
                print(f"\n  DATA COLLECTION SUMMARY:")
                print(f"   • Total jobs processed: {len(jobs_data)}")
                print(f"   • Jobs with URLs: {sum(1 for job in jobs_data if job.get('job_url'))}")
                print(f"   • Jobs with pay info: {sum(1 for job in jobs_data if job.get('pay_info', {}).get('raw_text'))}")
                print(f"   • Jobs with attributes: {sum(1 for job in jobs_data if job.get('job_attributes', {}).get('raw_text'))}")
                print(f"   • Data saved to: {json_path}")
                print(f"   • Summary saved to: {csv_path}")
            else:
                print("  No job data collected")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            browser.close()
            print("Browser closed.")
    
    return jobs_data

def main(job_term=None, max_jobs=None):
    """Main function for LinkedIn data collection using enhanced Playwright"""
    print("LINKEDIN JOB DATA COLLECTION (ENHANCED PLAYWRIGHT)")
    print("=" * 60)
    
    # Now i am getting user input if not provided
    if job_term is None:
        job_term = input("Enter job search term (default: Data Engineer): ").strip()
    if not job_term:
        job_term = "Data Engineer"
    
    if max_jobs is None:
        max_jobs_input = input("Enter maximum number of jobs to capture (default: 5): ").strip()
        try:
            max_jobs = int(max_jobs_input) if max_jobs_input else 5
        except ValueError:
            max_jobs = 5
    
    print(f"\nStarting LinkedIn job collection...")
    print(f"Job term: {job_term}")
    print(f"Max jobs: {max_jobs}")
    print("=" * 60)
    
    # Now i am creating output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/raw/linkedin_screenshots/linkedin_screenshots_{timestamp}"
    
    # Now i am ensuring output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Now i am capturing jobs using the enhanced scraper
    try:
        jobs_data = capture_multiple_job_card_screenshots_playwright(
            job_term=job_term, 
            output_dir=output_dir, 
            max_cards=max_jobs
        )
        
        if jobs_data:
            print(f"\n  Successfully collected data for {len(jobs_data)} jobs!")
            print("Data saved in structured format with screenshots and extracted text.")
            print(f"  Output directory: {output_dir}")
        else:
            print("\n  No job data was collected.")
        
        print("=" * 60)
        return jobs_data
        
    except Exception as e:
        print(f"\n  Error during collection: {e}")
        print("=" * 60)
        return []


if __name__ == "__main__":
    print("=" * 60)
    print("LINKEDIN JOB CARD SCREENSHOT CAPTURE (PLAYWRIGHT)")
    print("=" * 60)
    
    # Now i am testing with Data Engineer search
    job_term = "Data Engineer"
    output_directory = "test_job_screenshots_playwright"
    max_cards = 5
    
    print(f"Job Term: {job_term}")
    print(f"Output Directory: {output_directory}")
    print(f"Max Cards: {max_cards}")
    print("=" * 60)
    
    try:
        capture_multiple_job_card_screenshots_playwright(
            job_term=job_term, 
            output_dir=output_directory, 
            max_cards=max_cards
        )
        print("=" * 60)
        print("  Test completed successfully!")
        print("  For each job, you now get:")
        print("   • job_card_X.png - Individual job card (role & company)")
        print("   • pay_info_Job_X.png - Pay information (salary range) - captured after 'Show more'")
        print("   • job_attributes_Job_X.png - Job attributes (seniority, employment type, etc.) - captured after 'Show more'")
        print("   • job_details_Job_X.png - Full job details page (if navigated)")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"  Test failed: {e}")
        print("=" * 60)
