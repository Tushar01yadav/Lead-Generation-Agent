import sys
import io
import asyncio
import platform
import time
import os
import socket
import requests
import os
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager  

# ✅ Configure network settings
os.environ['PYTHONHTTPSVERIFY'] = '0'
socket.setdefaulttimeout(60)

# ==================== CRITICAL: Windows Fix MUST be FIRST ====================
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("✓ Windows event loop policy set")

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import zipfile
import subprocess
import shutil
import random
import re

# ==================== IMPORT SELENIUM EXTRACTOR ====================
from selenium_extractor import extract_company_data

# ==================== GLOBAL VARIABLES ====================
selenium_driver = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# ==================== CHROMEDRIVER SETUP ====================
def get_chrome_version():
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ['reg', 'query', 'HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon', '/v', 'version'],
                capture_output=True, text=True
            )
            return result.stdout.split()[-1]
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '--version'],
                capture_output=True, text=True
            )
            return result.stdout.split()[-1].split()[0]
        else:
            result = subprocess.run(['google-chrome', '--version'], capture_output=True, text=True)
            return result.stdout.split()[-1]
    except:
        return None

def setup_selenium_driver():
    """Setup and return Chrome driver with stealth mode"""


    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    selected_ua = random.choice(user_agents)
    
    opts = webdriver.ChromeOptions()
    opts.add_argument(f'--user-agent={selected_ua}')
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--start-maximized")
    
    # Force desktop view with large window size
    opts.add_argument('--window-size=1920,1080')
    
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)
    
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.default_content_setting_values.notifications": 2
    }
    opts.add_experimental_option("prefs", prefs)

    if os.path.exists("/usr/bin/chromium"):  # Streamlit Cloud
      opts.binary_location = "/usr/bin/chromium"
      service = Service("/usr/bin/chromedriver")
    else:  # Local
      from webdriver_manager.chrome import ChromeDriverManager
      service = Service(ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=opts)
    
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    print(f"✓ Browser started with stealth mode")
    
    driver.get("https://duckduckgo.com")
    time.sleep(2)
    print("✓ Browser ready for IPO research")
    
    return driver

def get_selenium_driver():
    global selenium_driver
    if selenium_driver is None:
        selenium_driver = setup_selenium_driver()
    return selenium_driver

def cleanup_selenium_driver():
    global selenium_driver
    if selenium_driver:
        try:
            selenium_driver.quit()
            print("✓ Browser closed successfully")
        except:
            pass
        selenium_driver = None

# ==================== DUCKDUCKGO SEARCH ====================
def search_duckduckgo_selenium(query, max_retries=3):
    """Search DuckDuckGo and return the first relevant Zerodha IPO link"""
    for attempt in range(max_retries):
        try:
            driver = get_selenium_driver()
            print(f"  🔍 Searching DuckDuckGo: {query}")

            driver.get("https://duckduckgo.com/")
            time.sleep(random.uniform(1, 2))

            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_box.clear()
            search_box.send_keys(query + Keys.RETURN)
            
            delay = random.uniform(1, 2)
            time.sleep(delay)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-testid='result']"))
            )

            result_links = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='result'] a[href]")

            for link in result_links:
                try:
                    href = link.get_attribute("href")
                    if not href:
                        continue

                    if any(x in href for x in ["duckduckgo.com", "youtube.com", "webcache"]):
                        continue

                    if any(domain in href for domain in ["zerodha.com/z-connect", "zerodha.com"]):
                        if "ipo" in href.lower():
                            print(f"  ✓ Found Zerodha IPO page: {href}")
                            return href
                except:
                    continue

            for link in result_links:
                try:
                    href = link.get_attribute("href")
                    if href and href.startswith("http") and "duckduckgo.com" not in href:
                        print(f"  ✓ Found (alternative): {href}")
                        return href
                except:
                    continue

            print("  ⚠️ No relevant link found")
            return None

        except Exception as e:
            print(f"  ⚠️ Search error (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(4, 7))
            else:
                return None

    return None

def clean_date_text(date_text):
    """Clean date text by removing ISO date prefix like '2025-11-10'"""
    if not date_text:
        return ""
    
    # Remove ISO date pattern (YYYY-MM-DD) from the beginning
    cleaned = re.sub(r'^\d{4}-\d{2}-\d{2}', '', date_text)
    
    # Clean up any extra whitespace
    cleaned = cleaned.strip()
    
    return cleaned

def parse_table_row(row, idx):
    """Parse a table row (desktop view) and extract company info"""
    try:
        company_name = None
        ipo_date = "Date unavailable"
        listing_date = "Date unavailable"
        
        # Extract company name from <span class="ipo-name"> inside <td class="name">
        name_td = row.find('td', class_='name')
        if name_td:
            # First try to find span with class "ipo-name"
            name_span = name_td.find('span', class_='ipo-name')
            if name_span:
                raw_name = name_span.get_text(strip=True)
                company_name = clean_duplicate_company_name(raw_name)
            
            # Fallback: try to get from link if span not found
            if not company_name:
                link = name_td.find('a')
                if link:
                    # Check if link contains span with ipo-name
                    name_span_in_link = link.find('span', class_='ipo-name')
                    if name_span_in_link:
                        raw_name = name_span_in_link.get_text(strip=True)
                        company_name = clean_duplicate_company_name(raw_name)
                    else:
                        raw_name = link.get_text(strip=True)
                        company_name = clean_duplicate_company_name(raw_name)
        
        # Extract dates from <td class="date">
        date_tds = row.find_all('td', class_='date')
        
        if len(date_tds) >= 1:
            # First date column: Contains IPO open - close dates (like "31st Oct 2025 – 04th Nov 2025")
            first_date_td = date_tds[0]
            
            # Get all text from first date column (this includes the date range)
            raw_date = first_date_td.get_text(strip=True)
            cleaned_date = clean_date_text(raw_date)
            
            if cleaned_date:
                ipo_date = cleaned_date
        
        # Second date column: Listing date
        if len(date_tds) >= 2:
            raw_listing = date_tds[1].get_text(strip=True)
            cleaned_listing = clean_date_text(raw_listing)
            if cleaned_listing:
                listing_date = cleaned_listing
        
        if company_name and company_name.strip():
            return {
                'Company_Name': company_name.strip(),
                'IPO_Date': ipo_date,
                'Listing_Date': listing_date,
                'Country': 'India'
            }
        
        return None
        
    except Exception as e:
        print(f"    ⚠️ Error parsing table row {idx}: {str(e)[:100]}")
        return None

def clean_duplicate_company_name(raw_name):
    """
    Remove duplicate company names like 'CARDEKHOCarDekho' -> 'CarDekho'
    Keeps the properly capitalized version (mixed case) and removes uppercase duplicate
    """
    if not raw_name or len(raw_name) < 3:
        return raw_name
    
    # Find where continuous uppercase/non-letter sequence ends
    uppercase_end = 0
    for i, char in enumerate(raw_name):
        if char.isupper() or not char.isalpha():
            uppercase_end = i + 1
        else:
            break
    
    # If we have an uppercase prefix and remaining text
    if 0 < uppercase_end < len(raw_name):
        uppercase_part = raw_name[:uppercase_end]
        remaining_part = raw_name[uppercase_end:]
        
        # Check if uppercase is duplicate (case-insensitive match)
        if uppercase_part.lower() == remaining_part.lower():
            return remaining_part
        
        # Check if remaining starts with uppercase (e.g., "HEROFINCORPHero Fincorp")
        if remaining_part.lower().startswith(uppercase_part.lower()):
            return remaining_part
    
    return raw_name

def parse_card(card, idx):
    """Parse a card (mobile view) and extract company info"""
    try:
        company_name = None
        ipo_date = "Date unavailable"
        listing_date = "Date unavailable"
        
        # Extract company name from <span class="ipo-name">
        name_span = card.find('span', class_='ipo-name')
        if name_span:
            raw_name = name_span.get_text(strip=True)
            # Clean duplicate names
            company_name = clean_duplicate_company_name(raw_name)
        
        # Extract IPO dates from card-middle
        card_middle = card.find('div', class_='card-middle')
        if card_middle:
            ipo_text = card_middle.get_text(strip=True)
            
            # Remove company name if found
            if company_name:
                ipo_text = ipo_text.replace(company_name, '').strip()
            
            # Split by separator if exists
            if '•' in ipo_text:
                ipo_text = ipo_text.split('•')[0].strip()
            
            # Extract the date range like "31st Oct 2025 – 04th Nov 2025"
            date_match = re.search(r'(\d+\w+\s+\w+\s+\d{4})\s*[-–]\s*(\d+\w+\s+\w+\s+\d{4})', ipo_text)
            if date_match:
                ipo_date = f"{date_match.group(1)} – {date_match.group(2)}"
        
        # Extract listing date from card-bottom
        card_bottom = card.find('div', class_='card-bottom')
        if card_bottom:
            label = card_bottom.find('label')
            if label and 'listing' in label.get_text(strip=True).lower():
                listing_text = card_bottom.get_text(strip=True)
                listing_text = listing_text.replace(label.get_text(strip=True), '').strip()
                
                # Extract date like "10 Nov 2025"
                date_match = re.search(r'\d+\s+\w+\s+\d{4}', listing_text)
                if date_match:
                    listing_date = date_match.group(0)
        
        if company_name and company_name.strip():
            return {
                'Company_Name': company_name.strip(),
                'IPO_Date': ipo_date,
                'Listing_Date': listing_date,
                'Country': 'India'
            }
        
        return None
        
    except Exception as e:
        print(f"    ⚠️ Error parsing card {idx}: {str(e)[:100]}")
        return None

# ==================== SCRAPE ZERODHA IPO TABLES ====================
def scrape_zerodha_ipo_page(url):
    """Scrape the Zerodha IPO page and extract Live + Upcoming IPO data"""
    driver = get_selenium_driver()
    
    print(f"  🌐 Opening Zerodha IPO page: {url}")
    driver.get(url)
    time.sleep(random.uniform(3, 4))
    
    # Maximize window to ensure desktop view
    driver.maximize_window()
    time.sleep(1)
    
    # Scroll to load content
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)
    
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    all_ipos = []
    
    # ==================== EXTRACT LIVE IPO ====================
    print("\n  📋 Looking for 'Live IPO' section...")
    live_section = soup.find('div', {'id': 'live-ipo-table'})
    
    if live_section:
        print("  ✅ Found Live IPO section")
        
        # Try desktop table view first
        live_table = live_section.find('tbody')
        if live_table:
            rows = live_table.find_all('tr')
            print(f"  📊 Found {len(rows)} Live IPO entries (table view)")
            
            for idx, row in enumerate(rows, 0):
                ipo_data = parse_table_row(row, idx)
                if ipo_data:
                    all_ipos.append(ipo_data)
                    print(f"    ✅ [{idx}] {ipo_data['Company_Name']} | IPO: {ipo_data['IPO_Date']} | Listing: {ipo_data['Listing_Date']}")
        else:
            # Try mobile card view
            print("  🔄 Table not found, trying card view...")
            cards = live_section.find_all('div', class_='card')
            if cards:
                print(f"  📊 Found {len(cards)} Live IPO entries (card view)")
                
                for idx, card in enumerate(cards, 0):
                    ipo_data = parse_card(card, idx)
                    if ipo_data:
                        all_ipos.append(ipo_data)
                        print(f"    ✅ [{idx}] {ipo_data['Company_Name']} | IPO: {ipo_data['IPO_Date']} | Listing: {ipo_data['Listing_Date']}")
            else:
                print("  ⚠️ No Live IPO data found in any format")
    else:
        print("  ⚠️ Live IPO section not found")
    
    # ==================== EXTRACT UPCOMING IPO ====================
    print("\n  📋 Looking for 'Upcoming IPO' section...")
    upcoming_section = soup.find('div', {'id': 'upcoming-ipo-table'})
    
    if upcoming_section:
        print("  ✅ Found Upcoming IPO section")
        
        # Try desktop table view first
        upcoming_table = upcoming_section.find('tbody')
        if upcoming_table:
            rows = upcoming_table.find_all('tr')
            print(f"  📊 Found {len(rows)} Upcoming IPO entries (table view)")
            
            for idx, row in enumerate(rows, 0):
                ipo_data = parse_table_row(row, idx)
                if ipo_data:
                    # Override listing date for upcoming IPOs
                    if ipo_data['Listing_Date'] == "Date unavailable":
                        ipo_data['Listing_Date'] = "To be announced"
                    all_ipos.append(ipo_data)
                    print(f"    ✅ [{idx}] {ipo_data['Company_Name']} | IPO: {ipo_data['IPO_Date']} | Listing: {ipo_data['Listing_Date']}")
        else:
            # Try mobile card view
            print("  🔄 Table not found, trying card view...")
            cards = upcoming_section.find_all('div', class_='card')
            if cards:
                print(f"  📊 Found {len(cards)} Upcoming IPO entries (card view)")
                
                for idx, card in enumerate(cards, 0):
                    ipo_data = parse_card(card, idx)
                    if ipo_data:
                        # Override listing date for upcoming IPOs
                        if ipo_data['Listing_Date'] == "Date unavailable":
                            ipo_data['Listing_Date'] = "To be announced"
                        all_ipos.append(ipo_data)
                        print(f"    ✅ [{idx}] {ipo_data['Company_Name']} | IPO: {ipo_data['IPO_Date']} | Listing: {ipo_data['Listing_Date']}")
            else:
                print("  ⚠️ No Upcoming IPO data found in any format")
    else:
        print("  ⚠️ Upcoming IPO section not found")
    
    return all_ipos

# ==================== MAIN FUNCTION ====================
def scrape_zerodha_ipos(llm_provider, api_key_1, api_key_2=None, output_csv=None):
    """
    Main function to scrape Zerodha IPOs and enrich data
    
    Args:
        llm_provider: LLM provider name (e.g., "Mistral", "Claude", "Gemini")
        api_key_1: Primary API key for LLM
        api_key_2: Secondary API key for LLM (optional)
        output_csv: Output CSV file path (optional)
    
    Returns:
        enriched_df: DataFrame with enriched data
        output_file: Path to saved CSV file
    """
    
    print("\n" + "=" * 80)
    print("🎯 ZERODHA IPO SCRAPER WITH DATA ENRICHMENT")
    print("=" * 80)
    print(f"🤖 LLM Provider: {llm_provider}")
    print(f"🔑 API Keys: {'2 keys (with rotation)' if api_key_2 else '1 key'}")
    print("=" * 80)
    
    # Validate API keys
    if not api_key_1:
        print("❌ CRITICAL ERROR: Primary API key is empty!")
        raise ValueError("Primary API key (api_key_1) cannot be empty")
    
    try:
        # Step 1: Search for Zerodha IPO page on DuckDuckGo
        print("\n📍 STEP 1: SEARCHING FOR ZERODHA IPO PAGE")
        print("-" * 80)
        
        search_query = "zerodha upcoming ipo"
        zerodha_url = search_duckduckgo_selenium(search_query)
        
        if not zerodha_url:
            print("❌ Could not find Zerodha IPO page via DuckDuckGo")
            print("   Falling back to direct URL...")
            zerodha_url = "https://zerodha.com/z-connect/ipo"
        
        # Step 2: Scrape IPO data
        print("\n📍 STEP 2: SCRAPING IPO DATA")
        print("-" * 80)
        
        ipo_data = scrape_zerodha_ipo_page(zerodha_url)
        
        if not ipo_data:
            print("❌ No IPO data extracted")
            cleanup_selenium_driver()
            return None, None
        
        # Create DataFrame
        df = pd.DataFrame(ipo_data)
        print(f"\n✅ Extracted {len(df)} IPO entries")
        
        # Close browser immediately after extraction
        print("\n🔒 Closing browser...")
        cleanup_selenium_driver()
        
        # Save initial scraped data as XLSX
        initial_xlsx = output_csv.replace('.xlsx', '_initial.xlsx') if output_csv else 'zerodha_ipos_initial.xlsx'
        df.to_excel(initial_xlsx, index=False, engine='openpyxl')
        print(f"💾 Initial scraped data saved to: {initial_xlsx}")
        
        # Step 3: Enrich data with founder/CEO/LinkedIn/email
        print("\n📍 STEP 3: ENRICHING DATA WITH FOUNDER/CEO/LINKEDIN/EMAIL")
        print("-" * 80)
        
        # Prepare API keys for enrichment
        api_keys = [api_key_1]
        if api_key_2:
            api_keys.append(api_key_2)
        
        # Call selenium_extractor to enrich data
        enriched_output = output_csv if output_csv else 'zerodha_ipos_enriched.xlsx'
        
        enriched_df, enriched_file = extract_company_data(
            csv_path=df,
            llm_provider=llm_provider,
            llm_api_keys=api_keys,
            output_path=enriched_output,
            company_column='Company_Name',
            apollo_api_key=None
        )
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ ZERODHA IPO SCRAPING & ENRICHMENT COMPLETE!")
        print("=" * 80)
        print(f"\n📊 FINAL STATISTICS:")
        print(f"  • Total IPO companies processed: {len(enriched_df)}")
        print(f"  • Initial data XLSX: {initial_xlsx}")
        print(f"  • Enriched data file: {enriched_file}")
        
        if 'Founder_Name' in enriched_df.columns:
            founder_count = len(enriched_df[enriched_df['Founder_Name'].str.len() > 0])
            print(f"\n👥 Companies with Founder Info:")
            print(f"  {founder_count} companies ({founder_count/len(enriched_df)*100:.1f}%)")
        
        if 'LinkedIn_URL' in enriched_df.columns:
            linkedin_count = len(enriched_df[enriched_df['LinkedIn_URL'].str.contains('linkedin.com', na=False)])
            print(f"\n🔗 Companies with LinkedIn Profiles:")
            print(f"  {linkedin_count} companies ({linkedin_count/len(enriched_df)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print("🎉 READY TO USE!")
        print("=" * 80)
        
        return enriched_df, enriched_file
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        cleanup_selenium_driver()

# ==================== CLI ENTRY POINT ====================
if __name__ == "__main__":
    """
    Example usage:
    python zerodha_ipo_scraper.py --llm Mistral --key1 YOUR_API_KEY_1 --key2 YOUR_API_KEY_2 --output zerodha_ipos.xlsx
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Zerodha IPOs and enrich with founder/CEO data')
    parser.add_argument('--llm', type=str, default='Mistral', help='LLM provider (Mistral, Claude, Gemini, etc.)')
    parser.add_argument('--key1', type=str, required=True, help='Primary API key')
    parser.add_argument('--key2', type=str, default=None, help='Secondary API key (optional)')
    parser.add_argument('--output', type=str, default='zerodha_ipos_enriched.xlsx', help='Output file path')
    
    args = parser.parse_args()
    
    enriched_df, output_file = scrape_zerodha_ipos(
        llm_provider=args.llm,
        api_key_1=args.key1,
        api_key_2=args.key2,
        output_csv=args.output
    )
    
    if enriched_df is not None:
        print(f"\n✅ Success! Enriched data saved to: {output_file}")
    else:
        print("\n❌ Scraping failed")
        sys.exit(1)