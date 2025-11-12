import sys
import time
import os
import socket
import platform
import random
import re
from datetime import datetime

# ✅ Configure network settings
os.environ['PYTHONHTTPSVERIFY'] = '0'
socket.setdefaulttimeout(60)

# ==================== CRITICAL: Windows Fix MUST be FIRST ====================
if platform.system() == 'Windows':
    import asyncio
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
import requests

# ==================== GLOBAL VARIABLES ====================
selenium_driver = None

# ==================== HELPER FUNCTIONS ====================
def clean_text(text):
    """Remove markdown formatting, extra spaces, and special characters"""
    if not text:
        return ''
    
    # Remove ** (bold markdown)
    text = text.replace('**', '')
    text = text.replace('*', '')
    
    # Remove other markdown characters if present
    text = text.replace('__', '')
    text = text.replace('~~', '')
    
    # Strip whitespace
    text = text.strip()
    
    return text

# ==================== CHROMEDRIVER SETUP ====================
def get_chrome_version():
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ['reg', 'query', 'HKEY_CURRENT_USER\\Software\\Google\\Chrome\\BLBeacon', '/v', 'version'],
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

def download_chromedriver():
    version = get_chrome_version()
    major = version.split('.')[0] if version else "141"
    os.makedirs("drivers", exist_ok=True)
    binary = "chromedriver.exe" if platform.system() == "Windows" else "chromedriver"
    path = os.path.abspath(os.path.join("drivers", binary))

    if os.path.exists(path):
        print(f"✓ ChromeDriver already exists at: {path}")
        return path

    try:
        print("📥 Downloading ChromeDriver...")
        url_latest = f"https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_{major}"
        r = requests.get(url_latest, timeout=10)
        ver_full = r.text.strip() if r.status_code == 200 else "141.0.7390.65"

        sys_map = {
            "Windows": "win64",
            "Darwin": "mac-arm64" if "arm" in platform.machine().lower() else "mac-x64",
            "Linux": "linux64"
        }
        suffix = sys_map.get(platform.system(), "win64")
        dl = f"https://storage.googleapis.com/chrome-for-testing-public/{ver_full}/{suffix}/chromedriver-{suffix}.zip"

        z = requests.get(dl, timeout=30)
        zip_path = os.path.join("drivers", "chromedriver.zip")
        with open(zip_path, "wb") as f:
            f.write(z.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("drivers")

        os.remove(zip_path)

        for root, dirs, files in os.walk("drivers"):
            for file in files:
                if file.startswith("chromedriver") and (file.endswith(".exe") or file == "chromedriver"):
                    extracted_path = os.path.join(root, file)
                    if root != "drivers":
                        shutil.move(extracted_path, path)

                    if platform.system() != "Windows":
                        os.chmod(path, 0o755)

                    print(f"✓ ChromeDriver downloaded successfully at: {path}")
                    return path

        return path
    except Exception as e:
        print(f"❌ Driver download error: {e}")
        return None

def setup_selenium_driver():
    """Setup and return Chrome driver with stealth mode"""
    driver_path = download_chromedriver()
    if not driver_path or not os.path.exists(driver_path):
        raise Exception(f"ChromeDriver not found at: {driver_path}")

    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    selected_ua = random.choice(user_agents)
    
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless")
    opts.add_argument(f'--user-agent={selected_ua}')
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--start-maximized")
    opts.add_argument('--window-size=1920,1080')
    
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)
    
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.default_content_setting_values.notifications": 2
    }
    opts.add_experimental_option("prefs", prefs)

    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=opts)
    
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    print(f"✓ Browser started with stealth mode")
    
    driver.get("https://duckduckgo.com")
    time.sleep(2)
    print("✓ Browser ready for Inc42 scraping")
    
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
    """Search DuckDuckGo and return the first Inc42 funding galore article link"""
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
            
            delay = random.uniform(2, 3)
            time.sleep(delay)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-testid='result']"))
            )

            result_links = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='result'] a[href]")

            found_url = None
            
            for link in result_links:
                try:
                    href = link.get_attribute("href")
                    if not href:
                        continue

                    if any(x in href for x in ["duckduckgo.com", "youtube.com", "webcache"]):
                        continue

                    # Look for Inc42 links
                    if "inc42.com" in href:
                        found_url = href
                        print(f"  ✓ Found Inc42 link: {href}")
                        break
                except:
                    continue

            if not found_url:
                print("  ⚠️ No Inc42 link found")
                return None
            
            # Check if it's a tag page or article page
            if "/tag/" in found_url or "/category/" in found_url:
                print(f"  📋 This is a tag/category page, need to extract first article...")
                article_url = extract_first_article_from_tag_page(found_url)
                return article_url
            elif "/buzz/" in found_url or any(keyword in found_url.lower() for keyword in ["funding", "galore"]):
                print(f"  ✓ Direct article link found!")
                return found_url
            else:
                print(f"  ⚠️ Unknown Inc42 page type, trying anyway...")
                return found_url

        except Exception as e:
            print(f"  ⚠️ Search error (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(4, 7))
            else:
                return None

    return None

def extract_first_article_from_tag_page(tag_url):
    """Extract the first article link from Inc42 tag/category page"""
    try:
        driver = get_selenium_driver()
        print(f"  🌐 Opening tag page: {tag_url}")
        
        driver.get(tag_url)
        time.sleep(random.uniform(3, 4))
        
        # Scroll a bit to load content
        driver.execute_script("window.scrollTo(0, 500);")
        time.sleep(1)
        
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        # Look for article cards with data-card-id attribute
        article_cards = soup.find_all('div', {'data-card-id': True})
        
        if article_cards:
            print(f"  📰 Found {len(article_cards)} article cards")
            
            for card in article_cards:
                # Find link inside card
                link = card.find('a', href=True)
                if link:
                    article_url = link['href']
                    
                    # Make sure it's a full URL
                    if not article_url.startswith('http'):
                        article_url = 'https://inc42.com' + article_url
                    
                    # Check if it's a funding galore article
                    article_title = link.get('title', '').lower()
                    if any(keyword in article_url.lower() or keyword in article_title for keyword in ['funding', 'galore', 'raised', 'startups']):
                        print(f"  ✅ First funding article: {article_url}")
                        return article_url
            
            # If no funding article found, return first article anyway
            first_link = article_cards[0].find('a', href=True)
            if first_link:
                article_url = first_link['href']
                if not article_url.startswith('http'):
                    article_url = 'https://inc42.com' + article_url
                print(f"  ✅ First article: {article_url}")
                return article_url
        
        # Fallback: try different selectors
        article_links = soup.find_all('a', href=True)
        for link in article_links:
            href = link['href']
            if '/buzz/' in href or '/features/' in href:
                if not href.startswith('http'):
                    href = 'https://inc42.com' + href
                print(f"  ✅ Found article (fallback): {href}")
                return href
        
        print("  ⚠️ Could not find any article on tag page")
        return None
        
    except Exception as e:
        print(f"  ❌ Error extracting article from tag page: {str(e)[:200]}")
        return None

# ==================== POST PROCESS ADDITIONAL INFO ====================
def extract_category_and_amount_from_additional_info(additional_info):
    """
    Extract category and funding amount from Additional Info
    Format: "B2B | $100 Mn | | Goldman Sachs Alternatives, A91 Partners | ..."
    We only need: B2B and $100 Mn
    """
    if not additional_info or additional_info.strip() == '':
        return '', ''
    
    # Clean text first
    additional_info = clean_text(additional_info)
    
    # Split by pipe
    parts = [p.strip() for p in additional_info.split('|')]
    
    category = ''
    funding_amount = ''
    
    # First part is category (B2B, B2C, etc.)
    if len(parts) > 0 and parts[0] in ['B2B', 'B2C', 'B2B2C', 'D2C', 'C2C']:
        category = parts[0]
    
    # Second part is funding amount
    if len(parts) > 1:
        amount_text = parts[1].strip()
        # Check if it has currency symbols or amount indicators
        if re.search(r'[\$₹€£]\s*[\d.,]+|[\d.,]+\s*(Mn|mn|Cr|cr|K|k|B|b|Million|Billion|Crore)', amount_text, re.IGNORECASE):
            funding_amount = amount_text
    
    return category, funding_amount

# ==================== SCRAPE INC42 FUNDING TABLE ====================
def scrape_inc42_funding_table(url):
    """Scrape the Inc42 funding table from the article"""
    driver = get_selenium_driver()
    
    print(f"  🌐 Opening Inc42 article: {url}")
    driver.get(url)
    time.sleep(random.uniform(3, 5))
    
    # Scroll down to load content and find the table
    print("  📜 Scrolling to find funding table...")
    
    # Scroll in steps to ensure content loads
    scroll_pause = 1.5
    screen_height = driver.execute_script("return window.innerHeight")
    scroll_amount = screen_height // 2
    
    max_scrolls = 10
    table_found = False
    
    for scroll in range(max_scrolls):
        driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        time.sleep(scroll_pause)
        
        # Check if we can find the table
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        # Look for table - it should be a standard HTML table
        tables = soup.find_all('table')
        
        if tables:
            print(f"  ✅ Found {len(tables)} table(s) on page")
            table_found = True
            break
    
    if not table_found:
        print("  ⚠️ No table found, getting full page HTML")
    
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    # Find all tables
    tables = soup.find_all('table')
    
    if not tables:
        print("  ❌ No HTML tables found on the page")
        return []
    
    print(f"  📊 Processing {len(tables)} table(s)...")
    
    all_funding_data = []
    
    for table_idx, table in enumerate(tables):
        print(f"\n  📋 Analyzing Table {table_idx + 1}...")
        
        # Get table headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [clean_text(th.get_text(strip=True)) for th in header_row.find_all(['th', 'td'])]
        
        # If no thead, check first row of tbody
        if not headers:
            tbody = table.find('tbody')
            if tbody:
                first_row = tbody.find('tr')
                if first_row:
                    potential_headers = [clean_text(td.get_text(strip=True)) for td in first_row.find_all(['th', 'td'])]
                    # Check if first row looks like headers
                    if any(keyword in ' '.join(potential_headers).lower() for keyword in ['date', 'name', 'sector', 'company']):
                        headers = potential_headers
        
        print(f"    Headers found: {headers}")
        
        # Find tbody rows
        tbody = table.find('tbody')
        if not tbody:
            print("    ⚠️ No tbody found, skipping table")
            continue
        
        rows = tbody.find_all('tr')
        print(f"    Found {len(rows)} data rows")
        
        # Skip first row if it's headers
        start_idx = 1 if not headers else 0
        
        for row_idx, row in enumerate(rows[start_idx:], start_idx):
            try:
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 2:
                    continue
                
                # Extract cell data and clean text
                cell_data = [clean_text(cell.get_text(strip=True)) for cell in cells]
                
                # Skip if row looks like a header
                if any(keyword in ' '.join(cell_data).lower() for keyword in ['date', 'name', 'sector', 'subsector']) and row_idx <= 1:
                    continue
                
                # Initialize
                category = ''
                funding_amount = ''
                
                # Extract based on number of columns
                if len(cell_data) >= 4:
                    # Structure: Date | Name | Sector | Subsector | Category | Amount | Round | Investors | Lead
                    date = cell_data[0]
                    company_name = cell_data[1]
                    industry_sector = cell_data[2]
                    
                    # Cell[4] = Category (B2B/B2C)
                    # Cell[5] = Funding Amount ($100 Mn)
                    if len(cell_data) > 4:
                        category = cell_data[4] if cell_data[4] in ['B2B', 'B2C', 'B2B2C', 'D2C', 'C2C'] else ''
                    
                    if len(cell_data) > 5:
                        funding_amount = clean_text(cell_data[5])
                    
                    # Create entry with correct column order
                    funding_entry = {
                        'Company_Name': company_name,
                        'Funding_Amount': funding_amount,
                        'Country': 'India',
                        'Founder_Name': '',
                        'Important_Person': '',
                        'Email': '',
                        'LinkedIn_URL': '',
                        'Industry_Sector': industry_sector,
                        'Category': category,
                        'Description': '',
                        'Date': date
                    }
                
                elif len(cell_data) == 3:
                    # Date | Name | Sector
                    funding_entry = {
                        'Company_Name': cell_data[1],
                        'Funding_Amount': '',
                        'Country': 'India',
                        'Founder_Name': '',
                        'Important_Person': '',
                        'Email': '',
                        'LinkedIn_URL': '',
                        'Industry_Sector': cell_data[2],
                        'Category': '',
                        'Description': '',
                        'Date': cell_data[0]
                    }
                
                elif len(cell_data) == 2:
                    # Date | Name
                    funding_entry = {
                        'Company_Name': cell_data[1],
                        'Funding_Amount': '',
                        'Country': 'India',
                        'Founder_Name': '',
                        'Important_Person': '',
                        'Email': '',
                        'LinkedIn_URL': '',
                        'Industry_Sector': '',
                        'Category': '',
                        'Description': '',
                        'Date': cell_data[0]
                    }
                
                # Validate company name
                company_name = funding_entry.get('Company_Name', '').strip()
                
                # Skip footer text
                if any(skip_text in company_name.lower() for skip_text in [
                    'part of a larger round',
                    'source:',
                    'note:',
                    'only disclosed',
                    'inc42'
                ]):
                    continue
                
                if funding_entry and company_name and len(company_name) > 1:
                    all_funding_data.append(funding_entry)
                    print(f"    ✅ [{row_idx}] {company_name} | {funding_entry['Category']} | {funding_entry['Funding_Amount']}")
                
            except Exception as e:
                print(f"    ⚠️ Error parsing row {row_idx}: {str(e)[:100]}")
                continue
    
    return all_funding_data

# ==================== MAIN FUNCTION ====================
def scrape_inc42_funding_galore(output_csv=None):
    """
    Main function to scrape Inc42 Funding Galore
    
    Args:
        output_csv: Output CSV file path (optional)
    
    Returns:
        df: DataFrame with funding data
        output_file: Path to saved CSV file
    """
    
    print("\n" + "=" * 80)
    print("🎯 INC42 FUNDING GALORE SCRAPER")
    print("=" * 80)
    
    try:
        # Step 1: Search for Inc42 Funding Galore on DuckDuckGo
        print("\n📍 STEP 1: SEARCHING FOR INC42 FUNDING GALORE")
        print("-" * 80)
        
        search_query = "funding galore inc42"
        inc42_url = search_duckduckgo_selenium(search_query)
        
        if not inc42_url:
            print("❌ Could not find Inc42 Funding Galore via DuckDuckGo")
            print("   Please provide a direct URL to the article")
            cleanup_selenium_driver()
            return None, None
        
        # Step 2: Scrape funding table
        print("\n📍 STEP 2: SCRAPING FUNDING TABLE")
        print("-" * 80)
        
        funding_data = scrape_inc42_funding_table(inc42_url)
        
        if not funding_data:
            print("❌ No funding data extracted")
            cleanup_selenium_driver()
            return None, None
        
        # Create DataFrame
        df = pd.DataFrame(funding_data)
        
        # Ensure correct column order
        column_order = [
            'Company_Name',
            'Funding_Amount', 
            'Country',
            'Founder_Name',
            'Important_Person',
            'Email',
            'LinkedIn_URL',
            'Industry_Sector',
            'Category',
            'Description',
            'Date'
        ]
        
        # Make sure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = ''
        
        df = df[column_order]
        
        # Clean all text columns to remove ** and other formatting
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: clean_text(str(x)) if pd.notna(x) else '')
        
        print(f"\n✅ Extracted {len(df)} funding entries")
        
        # Close browser
        print("\n🔒 Closing browser...")
        cleanup_selenium_driver()
        
        # Save to Excel
        if output_csv:
            output_file = output_csv
        else:
            output_file = 'inc42_funding_galore.xlsx'
        
        if output_file.endswith('.xlsx'):
            df.to_excel(output_file, index=False, engine='openpyxl')
        else:
            df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"💾 Data saved to: {output_file}")
        
        # Display summary
        print("\n" + "=" * 80)
        print("✅ INC42 FUNDING GALORE SCRAPING COMPLETE!")
        print("=" * 80)
        print(f"\n📊 SUMMARY:")
        print(f"  • Total companies: {len(df)}")
        print(f"  • Output file: {output_file}")
        print(f"  • Article URL: {inc42_url}")
        
        if 'Industry_Sector' in df.columns:
            sector_counts = df['Industry_Sector'].value_counts()
            print(f"\n🏢 Top Industries/Sectors:")
            for sector, count in sector_counts.head(5).items():
                if sector:
                    print(f"  • {sector}: {count} companies")
        
        # Funding statistics
        if 'Funding_Amount' in df.columns:
            funded_count = len(df[df['Funding_Amount'].str.len() > 0])
            print(f"\n💰 Companies with Funding Amount: {funded_count}/{len(df)}")
            if funded_count > 0:
                print(f"  Sample amounts:")
                for idx, row in df[df['Funding_Amount'].str.len() > 0].head(5).iterrows():
                    print(f"    • {row['Company_Name']}: {row['Funding_Amount']}")
        
        print("\n" + "=" * 80)
        
        return df, output_file
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        cleanup_selenium_driver()

# ==================== CLI ENTRY POINT ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Inc42 Funding Galore table')
    parser.add_argument('--output', type=str, default='inc42_funding_galore.xlsx', help='Output file path')
    parser.add_argument('--url', type=str, default=None, help='Direct URL to Inc42 article (optional)')
    
    args = parser.parse_args()
    
    if args.url:
        print(f"📌 Using direct URL: {args.url}")
        try:
            get_selenium_driver()
            funding_data = scrape_inc42_funding_table(args.url)
            
            if funding_data:
                df = pd.DataFrame(funding_data)
                
                column_order = [
                    'Company_Name', 'Funding_Amount', 'Country', 'Founder_Name',
                    'Important_Person', 'Email', 'LinkedIn_URL', 'Industry_Sector',
                    'Category', 'Description', 'Date'
                ]
                
                for col in column_order:
                    if col not in df.columns:
                        df[col] = ''
                
                df = df[column_order]
                
                # Clean all text columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(lambda x: clean_text(str(x)) if pd.notna(x) else '')
                
                output_file = args.output
                
                if output_file.endswith('.xlsx'):
                    df.to_excel(output_file, index=False, engine='openpyxl')
                else:
                    df.to_csv(output_file, index=False, encoding='utf-8')
                
                print(f"✅ Success! Data saved to: {output_file}")
            else:
                print("❌ No data extracted")
        finally:
            cleanup_selenium_driver()
    else:
        df, output_file = scrape_inc42_funding_galore(output_csv=args.output)
        
        if df is not None:
            print(f"\n✅ Success! Data saved to: {output_file}")
        else:
            print("\n❌ Scraping failed")
            sys.exit(1)