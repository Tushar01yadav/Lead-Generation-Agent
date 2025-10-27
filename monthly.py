#this is monthly.py
import sys
import io
import time
import re
import os
import json
import random
import platform
import subprocess
import zipfile
import shutil
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from datetime import datetime
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# Get current month for date extraction
this_month = datetime.now().strftime("%B")

# ==================== LLM CONFIGS ====================
LLM_CONFIGS = {
    "Mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "auth_header": "Bearer"
    },
    "Claude": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-sonnet-20241022",
        "auth_header": "x-api-key",
        "anthropic_version": "2023-06-01"
    },
    "Openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "auth_header": "Bearer"
    },
    "Gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.0-flash",
        "auth_header": "Bearer"
    },
    "Deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "auth_header": "Bearer"
    },
    "Qwen": {
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen-plus",
        "auth_header": "Bearer"
    },
    "Perplexity": {
        "url": "https://api.perplexity.ai/chat/completions",
        "model": "sonar",
        "auth_header": "Bearer"
    },
    "Llama": {
        "url": "https://api.llama-api.com/chat/completions",
        "model": "llama-3.3-70b",
        "auth_header": "Bearer"
    }
}

# ==================== CONFIG ====================
FOUNDERSDAY_DIRECT_URL = "https://www.foundersday.co/startup-funding-tracker"

# Global variables for API key management
current_api_key = None
api_key_1 = None
api_key_2 = None
current_key_index = 0
current_llm_name = None

def initialize_api_keys(key1, key2, llm_name):
    """Initialize the API keys for rotation"""
    global api_key_1, api_key_2, current_api_key, current_key_index, current_llm_name
    
    if not key1:
        print("❌ ERROR: Primary API key (key1) is empty!")
        print(f"   key1 value: '{key1}'")
        print(f"   key2 value: '{key2}'")
        raise ValueError("Primary API key cannot be empty!")
    
    api_key_1 = key1
    api_key_2 = key2 if key2 else key1
    current_api_key = key1
    current_key_index = 0
    current_llm_name = llm_name
    
    print(f"✓ API keys initialized for {llm_name}:")
    print(f"  • Primary key: {key1[:10]}...{key1[-4:]}")
    print(f"  • Secondary key: {key2[:10] if key2 else 'Using primary'}...{key2[-4:] if key2 else ''}")
    print(f"  • Starting with key 1")

def rotate_api_key():
    """Rotate to the next API key"""
    global current_api_key, current_key_index
    current_key_index = 1 - current_key_index
    current_api_key = api_key_1 if current_key_index == 0 else api_key_2
    print(f"🔄 Rotated to API key {current_key_index + 1}")
    return current_api_key

# ==================== URL BUILDER ====================
def build_url_config(region, category):
    """
    Build URL configuration for both FoundersDay and GrowthList
    
    Returns: (fd_base_url, fd_max_pages, fd_url_generator, gl_query)
    """
    region = region.lower().strip()
    category = category.lower().strip()
    
    print(f"\n📋 Configuration: Region='{region}', Category='{category}'")
    
    # ========== FOUNDERSDAY CONFIG ==========
    # Case 1: India/Global + All/General → 13 pages from main tracker
    if category in ['all', 'general']:
        fd_base_url = FOUNDERSDAY_DIRECT_URL
        fd_max_pages = 13
        
        def fd_url_generator(page):
            if page == 1:
                return fd_base_url
            else:
                return f"{fd_base_url}?74bb1511_page={page}"
        
        # GrowthList query for Global All
        if region == 'global':
            gl_query = "growthlist latest funded global startups 2025"
        else:
            gl_query = "growthlist latest funded indian startups 2025"
        
        print(f"✓ FoundersDay: main tracker (13 pages)")
        print(f"✓ GrowthList: {gl_query}")
        return fd_base_url, fd_max_pages, fd_url_generator, gl_query
    
    # Case 2: India + Specific Category → 2 pages from category-specific URL
    elif region == 'india':
        fd_base_url = f"https://www.foundersday.co/startupfunding/india-vc-funding-tracker-for-{category}-startups"
        fd_max_pages = 2
        
        def fd_url_generator(page):
            if page == 1:
                return fd_base_url
            else:
                return f"{fd_base_url}?74bb1511_page={page}"
        
        # No GrowthList for India-specific categories
        gl_query = None
        
        print(f"✓ FoundersDay: India-specific category URL (2 pages)")
        print(f"✓ GrowthList: Skipped (India-specific category)")
        return fd_base_url, fd_max_pages, fd_url_generator, gl_query
    
    # Case 3: Global + Specific Category → Skip FoundersDay, use GrowthList only
    else:  # global + specific category
        # Map categories to GrowthList queries
        category_query_map = {
            'healthcare': 'growthlist latest funded healthcare startups 2025',
            'healthtech': 'growthlist latest funded healthcare startups 2025',
            'education': 'growthlist latest funded education startups 2025',
            'edtech': 'growthlist latest funded education startups 2025',
            'ai': 'growthlist latest funded ai startups 2025',
            'artificial intelligence': 'growthlist latest funded ai startups 2025',
            'real estate': 'growthlist latest funded real estate startups 2025',
            'realestate': 'growthlist latest funded real estate startups 2025',
            'proptech': 'growthlist latest funded real estate startups 2025',  # Same as real estate
            'fintech': 'growthlist latest funded fintech startups 2025',
            'finance': 'growthlist latest funded fintech startups 2025',
        }
        
        gl_query = category_query_map.get(category)
        
        if gl_query:
            print(f"✓ FoundersDay: Skipped (Global + specific category)")
            print(f"✓ GrowthList: {gl_query}")
            return None, 0, None, gl_query
        else:
            print(f"⚠️ Unknown category '{category}' - no scraping available")
            return None, 0, None, None
# ==================== SELENIUM SETUP ====================
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
    driver_path = download_chromedriver()
    if not driver_path or not os.path.exists(driver_path):
        raise Exception(f"ChromeDriver not found at: {driver_path}")

    opts = webdriver.ChromeOptions()
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)

    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=opts)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

selenium_driver = None

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
        except:
            pass
        selenium_driver = None

# ==================== HELPER FUNCTIONS ====================
def convert_to_inr(amount_str):
    """Converts USD/other currency amounts to INR approximation."""
    if not amount_str:
        return ""

    usd_rate = 83

    match = re.search(r'\$\s*([\d.]+)\s*([MBK])', amount_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()

        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        numeric_value = value * multipliers.get(unit, 1)
        inr_value = numeric_value * usd_rate

        if inr_value >= 10000000:
            return f"INR {inr_value / 10000000:.2f} Cr"
        elif inr_value >= 100000:
            return f"INR {inr_value / 100000:.2f} Lakh"
        else:
            return f"INR {inr_value:,.0f}"

    if 'INR' in amount_str or '₹' in amount_str or '\u20B9' in amount_str:
        amount_str = re.sub(r'[^\x00-\x7F]+', 'INR ', amount_str)
        return amount_str

    return amount_str

def is_valid_founder_name(name):
    """Validates if extracted text is actually a founder name."""
    if not name or len(name) < 2:
        return False

    name = name.strip()

    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', name):
        return False

    if len(re.findall(r'\d', name)) / len(name) > 0.3:
        return False

    if re.match(r'^[A-Z]{1,4}$', name):
        return False

    if ' ' not in name and len(name) < 3:
        return False

    if not re.search(r'[a-zA-Z]', name):
        return False

    keywords = ['saas', 'b2b', 'b2c', 'fintech', 'healthtech', 'platform', 'startup', 
                'investors', 'founder', 'founders', 'fundraise', 'india', 'company',
                'confidence', 'sharma confidence', 'ceo', 'cto', 'cfo']
    if any(kw in name.lower() for kw in keywords):
        return False

    if ' ' not in name:
        return False

    if '\n' in name:
        return False

    parts = name.split()
    if len(parts) < 2:
        return False

    for part in parts:
        if not part.isalpha():
            return False

    return True

def validate_founder_against_company(company_name, founder_name):
    """Check if founder name is similar to company name and should be removed."""
    if not company_name or not founder_name:
        return True

    company_clean = re.sub(r'[^a-zA-Z0-9\s]', '', company_name.lower().strip())
    founder_clean = re.sub(r'[^a-zA-Z0-9\s]', '', founder_name.lower().strip())

    company_suffixes = ['inc', 'ltd', 'llc', 'corp', 'corporation', 'company', 'co', 'pvt', 'private', 'limited']
    company_words = company_clean.split()
    company_words = [word for word in company_words if word not in company_suffixes]
    company_clean = ' '.join(company_words)

    if company_clean == founder_clean:
        print(f"  🚫 Removing founder '{founder_name}' - matches company name '{company_name}'")
        return False

    if len(founder_clean) > 3:
        if founder_clean in company_clean or company_clean in founder_clean:
            print(f"  🚫 Removing founder '{founder_name}' - too similar to company name '{company_name}'")
            return False

    company_word_set = set(company_words)
    founder_word_set = set(founder_clean.split())

    if len(founder_word_set) > 0:
        similarity = len(company_word_set.intersection(founder_word_set)) / len(founder_word_set)
        if similarity >= 0.8:
            print(f"  🚫 Removing founder '{founder_name}' - {similarity*100:.0f}% similarity to company name '{company_name}'")
            return False

    return True

def filter_category_field(category):
    """Filter category field to only allow B2B/B2C and remove unwanted categories."""
    if not category:
        return ""

    category = category.strip()

    category_parts = []
    for part in category.replace(',', '|').replace('/', '|').replace('&', '|').replace(' and ', '|').split('|'):
        category_parts.extend(part.strip().split())

    allowed_terms = []
    for part in category_parts:
        part_lower = part.lower().strip()

        if part_lower in ['b2b', 'b2c', 'b2g']:
            allowed_terms.append(part_lower.upper())
        elif part_lower in ['business', 'consumer', 'government']:
            if part_lower == 'business':
                allowed_terms.append('B2B')
            elif part_lower == 'consumer':
                allowed_terms.append('B2C')
            elif part_lower == 'government':
                allowed_terms.append('B2G')

    unique_terms = []
    for term in allowed_terms:
        if term not in unique_terms:
            unique_terms.append(term)

    return ', '.join(unique_terms) if unique_terms else ""

def is_noise_text(text):
    """Checks if text is noise/newsletter signup text."""
    noise_keywords = [
        "receive a weekly summary",
        "unsubscribe anytime",
        "filter by sector",
        "newsletter",
        "subscribe",
        "email",
        "sign up",
        "get updates"
    ]
    return any(keyword in text.lower() for keyword in noise_keywords)

# ==================== UNIVERSAL LLM CALLER ====================
def call_llm(llm_name, api_key, prompt, max_tokens=1000):
    """
    Universal LLM caller that works with any configured LLM
    
    Args:
        llm_name: Name of the LLM (case insensitive)
        api_key: API key for the LLM
        prompt: The prompt to send
        max_tokens: Maximum tokens in response
    
    Returns:
        str: The LLM response content or None on error
    """
    
    # Case-insensitive lookup
    matched_key = None
    for key in LLM_CONFIGS.keys():
        if key.lower() == llm_name.lower():
            matched_key = key
            break
    
    if matched_key is None:
        print(f"❌ Error: {llm_name} not supported")
        print(f"Available LLMs: {', '.join(LLM_CONFIGS.keys())}")
        return None
    
    config = LLM_CONFIGS[matched_key]
    
    # Build request based on LLM type
    if matched_key.lower() == "claude":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": config["anthropic_version"]
        }
        data = {
            "model": config["model"],
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{config['auth_header']} {api_key}"
        }
        data = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
    
    try:
        response = requests.post(config["url"], headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            print(f"⚠️ API error {response.status_code} with {matched_key}")
            return None
        
        result = response.json()
        
        # Extract content based on provider
        if matched_key.lower() == "claude":
            content = result['content'][0]['text']
        else:
            content = result['choices'][0]['message']['content']
        
        return content
        
    except Exception as e:
        print(f"❌ Error calling {matched_key}: {e}")
        return None

# ==================== LLM EXTRACTION FUNCTIONS ====================
def extract_with_llm(card_text, retry=0, max_retries=3):
    """Uses selected LLM with key rotation to extract company data."""
    global current_api_key, current_llm_name

    try:
        prompt = f"""You are an expert at extracting startup funding data. Extract ONLY valid JSON from this text.

TEXT:
{card_text}

INSTRUCTIONS:
- Company name: Extract ONLY the actual company/startup name (1-4 words max)
- Funding amount: extract with proper currency (e.g., $4M, INR 50Cr)
  * If BOTH Rs/INR and USD present, ALWAYS pick Rs/INR amount
  * Write rupee symbol as "INR"
- Funding round: Seed, Pre-seed, Series A, etc.
- Founders: Extract ONLY actual founder names (FULL first and last name)
- Industry/sectors: tags like SAAS, AI, healthtech, etc.
- Category: B2B/B2C/B2G
- Description: 1-2 sentence description of what the company does
- Date: Extract any date mentioned (format: 'MMM DD, YYYY' or 'MMM DD-DD, YYYY')

Return ONLY valid JSON (no markdown):
{{
  "company_name": "Short company name only",
  "funding_amount": "INR amount if available, else $USD",
  "funding_round": "Seed/Series A etc",
  "founder_name": "First complete founder name",
  "important_person": "Other complete founder names comma separated",
  "industry": "comma separated",
  "category": "B2B, B2C, or domain",
  "description": "What company does",
  "date": "Extract date in format 'MMM DD, YYYY' or 'MMM DD-DD, YYYY'"
}}"""

        response_text = call_llm(current_llm_name, current_api_key, prompt, max_tokens=1000)

        if not response_text:
            if retry < max_retries:
                rotate_api_key()
                wait_time = 2 ** retry
                print(f"🔄 Retrying with key {current_key_index + 1} in {wait_time}s...")
                time.sleep(wait_time)
                return extract_with_llm(card_text, retry + 1, max_retries)
            else:
                print(f"⚠️ LLM extraction failed after {max_retries} retries")
                return None

        # Clean JSON response
        response_text = re.sub(r'```json\n?|\n?```', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)
        response_text = response_text.strip()

        data = json.loads(response_text)

        # Post-processing validation
        company_name = data.get("company_name", "").strip()

        if company_name.lower() == "investors" or "investor" in company_name.lower():
            print(f"  ⚠️ 'Investors' detected as company name, attempting to fix")
            lines = card_text.split('\n')
            found_company = False
            for line in lines[:10]:
                line = line.strip()
                if not line or len(line) > 100:
                    continue
                if any(keyword in line.lower() for keyword in ['investor', 'investors', 'funding from', 'backed by', 'led by']):
                    continue
                if '.' in line and len(line) < 30:
                    data["company_name"] = line
                    print(f"  ✓ Fixed company name to: '{line}'")
                    found_company = True
                    break
                elif len(line.split()) <= 4 and len(line) > 2:
                    data["company_name"] = line
                    print(f"  ✓ Fixed company name to: '{line}'")
                    found_company = True
                    break

            if not found_company:
                print(f"  ❌ Could not extract valid company name")
                return None

        if len(company_name) > 50 or '. ' in company_name:
            print(f"  ⚠️ Company name looks like description: '{company_name[:50]}...'")
            lines = card_text.split('\n')
            for line in lines[:5]:
                line = line.strip()
                if line and len(line) < 50 and len(line) > 2 and 'investor' not in line.lower():
                    data["company_name"] = line
                    print(f"  ✓ Fixed company name to: '{line}'")
                    break

        funding_amount = data.get("funding_amount", "")
        if funding_amount:
            funding_amount = re.sub(r'[^\x00-\x7F]+', 'INR ', funding_amount)
            funding_amount = re.sub(r'\s+', ' ', funding_amount).strip()
            data["funding_amount"] = funding_amount

        return data

    except json.JSONDecodeError:
        if retry < max_retries:
            rotate_api_key()
            wait_time = 2 ** retry
            print(f"⚠️ JSON parsing error. Retrying with key {current_key_index + 1} in {wait_time}s...")
            time.sleep(wait_time)
            return extract_with_llm(card_text, retry + 1, max_retries)
        else:
            print(f"⚠️ JSON parsing failed after {max_retries} retries")
            return None

    except Exception as e:
        if retry < max_retries:
            rotate_api_key()
            wait_time = 2 ** retry
            print(f"⚠️ Error: {e}. Retrying with key {current_key_index + 1} in {wait_time}s...")
            time.sleep(wait_time)
            return extract_with_llm(card_text, retry + 1, max_retries)
        else:
            print(f"⚠️ Failed after {max_retries} retries: {e}")
            return None

# ==================== FOUNDERSDAY SCRAPER ====================
def scrape_foundersday(region, category, output_csv):
    """Main FoundersDay scraper using selected LLM with dynamic URL/page handling"""
    
    # Build URL configuration based on region and category
    base_url, max_pages, url_generator, gl_query = build_url_config(region, category)
    
    # Skip scraping for unsupported combinations
    if base_url is None:
        print(f"\n⚠️ Skipping scraping for region='{region}' and category='{category}'")
        df = pd.DataFrame(columns=[
            "Company_Name", "Funding_Amount", "Country", "Founder_Name",
            "Important_Person", "Email", "LinkedIn_URL", "Industry_Sector", "Category", "Description", "Date"
        ])
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        return 0
    
    driver = get_selenium_driver()
    all_data = []
    seen_companies = set()

    try:
        print("\n" + "=" * 70)
        print(f"🚀 FoundersDay Scraper (Using {current_llm_name})")
        print(f"📄 Scraping up to {max_pages} pages")
        print(f"🌍 Region: {region.upper()}")
        print(f"📂 Category: {category.upper()}")
        print("=" * 70)

        for page in range(1, max_pages + 1):
            print(f"\n{'=' * 70}")
            print(f"📖 Page {page}/{max_pages}")
            print(f"{'=' * 70}")

            url = url_generator(page)

            print(f"🔗 Loading: {url}")
            driver.get(url)
            time.sleep(5)

            current_url = driver.current_url
            print(f"📍 Current URL: {current_url}")

            driver.execute_script("window.scrollTo(0, 1000);")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

            selectors_to_try = [
                "//div[@role='listitem' and contains(@class, 'w-dyn-item')]",
                "//div[contains(@class, 'w-dyn-item')]",
                "//div[contains(@class, 'div-block-2')]",
                "//div[@role='listitem']"
            ]

            cards = []
            for selector in selectors_to_try:
                try:
                    found = driver.find_elements(By.XPATH, selector)
                    if len(found) >= 1:
                        cards = found
                        print(f"✅ Found {len(cards)} cards using: {selector}")
                        break
                except:
                    continue

            if not cards:
                print("⚠️ No company cards found")
                if page > 1:
                    print("🛑 Stopping pagination")
                    break
                continue

            print(f"📊 Processing {len(cards)} cards with {current_llm_name}...")

            page_companies = []

            for idx, card in enumerate(cards, 1):
                try:
                    card_text = card.text.strip() if card.text else ""

                    if not card_text or len(card_text) < 30 or is_noise_text(card_text):
                        continue

                    print(f"\n🤖 Processing card #{idx}...")

                    extracted = extract_with_llm(card_text)

                    if not extracted or not extracted.get("company_name"):
                        print(f"⚠️ LLM extraction failed for card #{idx}")
                        continue

                    company_name = extracted.get("company_name", "").strip()

                    company_name_lower = company_name.lower()
                    if company_name_lower in seen_companies:
                        print(f"⚠️ Duplicate detected: {company_name}")
                        continue

                    funding_amount = extracted.get("funding_amount", "")
                    inr_amount = convert_to_inr(funding_amount) if funding_amount else ""

                    founder_name = extracted.get("founder_name", "").strip()
                    important_person = extracted.get("important_person", "").strip()

                    if founder_name and not is_valid_founder_name(founder_name):
                        founder_name = ""

                    if founder_name and not validate_founder_against_company(company_name, founder_name):
                        founder_name = ""

                    if important_person:
                        persons = [p.strip() for p in important_person.split(",")]
                        valid_persons = []
                        for person in persons:
                            if is_valid_founder_name(person) and validate_founder_against_company(company_name, person):
                                valid_persons.append(person)
                        important_person = ", ".join(valid_persons)

                    data_row = {
                        "Company_Name": company_name,
                        "Funding_Amount": inr_amount,
                        "Country": region.title(),
                        "Founder_Name": founder_name,
                        "Important_Person": important_person,
                        "Email": "",
                        "LinkedIn_URL": "",
                        "Industry_Sector": extracted.get("industry", "").strip(),
                        "Category": filter_category_field(extracted.get("category", "")),
                        "Description": extracted.get("description", "").strip(),
                        "Date": extracted.get("date", "").strip()
                    }

                    all_data.append(data_row)
                    seen_companies.add(company_name_lower)
                    page_companies.append(company_name)

                    print(f"✅ Extracted: {data_row['Company_Name']}")
                    if data_row["Funding_Amount"]:
                        print(f"   Funding: {data_row['Funding_Amount']}")
                    if data_row["Founder_Name"]:
                        print(f"   Founder: {data_row['Founder_Name']}")

                except Exception as e:
                    print(f"⚠️ Error processing card #{idx}: {e}")
                    continue

            if not page_companies:
                print(f"\n⚠️ No new companies found on page {page}")
                if page > 1:
                    print("🛑 Stopping pagination")
                    break
            else:
                print(f"\n✅ Page {page}: Extracted {len(page_companies)} new companies")

            if page < max_pages:
                print(f"\n⏳ Waiting 5 seconds before loading page {page + 1}...")
                time.sleep(5)

        if all_data:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset=["Company_Name"], keep="first", inplace=True)
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n✅ FoundersDay Complete: {len(df)} companies saved to {output_csv}")
            return len(df)
        else:
            print("\n⚠️ No FoundersDay data extracted")
            df = pd.DataFrame(columns=[
                "Company_Name", "Funding_Amount", "Country", "Founder_Name",
                "Important_Person", "Email", "LinkedIn_URL", "Industry_Sector", "Category", "Description", "Date"
            ])
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            return 0

    except Exception as e:
        print(f"\n❌ Error in FoundersDay scraping: {e}")
        import traceback
        traceback.print_exc()
        return 0
# ==================== GROWTHLIST SCRAPER ====================
def scrape_growthlist(gl_query, region, output_csv, max_pages=10):
    """
    Main GrowthList scraper - searches and extracts funding data.
    NO SOURCE COLUMN ADDED.
    """
    
    if not gl_query:
        print("\n⚠️ No GrowthList query provided - skipping")
        return 0
    
    driver = get_selenium_driver()
    all_data = []
    
    try:
        print("\n" + "=" * 70)
        print(f"🚀 GrowthList Scraper")
        print(f"🔍 Query: {gl_query}")
        print(f"🌍 Region: {region.upper()}")
        print(f"📄 Max Pages: {max_pages}")
        print("=" * 70)
        
        # Search for GrowthList page
        if not search_growthlist_page(driver, gl_query):
            print("❌ Failed to find GrowthList page")
            return 0
        
        # Set entries to 100
        set_growthlist_entries(driver, "100")
        
        # Scrape all pages
        current_page = 1
        while True:
            print(f"\n{'=' * 70}")
            print(f"📖 Page {current_page}/{max_pages}")
            print(f"{'=' * 70}")
            
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            page_data = extract_growthlist_table(driver)
            
            # Add missing fields with proper None values (NO SOURCE FIELD)
            for row in page_data:
                if 'Founder_Name' not in row:
                    row['Founder_Name'] = None
                if 'Important_Person' not in row:
                    row['Important_Person'] = None
                if 'Email' not in row:
                    row['Email'] = None
                if 'LinkedIn_URL' not in row:
                    row['LinkedIn_URL'] = None
                if 'Category' not in row:
                    row['Category'] = None
                if 'Description' not in row:
                    row['Description'] = None
                # DO NOT ADD SOURCE FIELD
            
            all_data.extend(page_data)
            print(f"✅ Page {current_page}: Extracted {len(page_data)} companies")
            
            if current_page >= max_pages or not go_to_next_page(driver, current_page):
                break
            current_page += 1
        
        # Save to CSV
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Replace None with empty strings for CSV compatibility
            df = df.fillna("")
            
            # Ensure all required columns exist (NO SOURCE)
            required_columns = [
                "Company_Name", "Funding_Amount", "Country", "Founder_Name",
                "Important_Person", "Email", "LinkedIn_URL", "Industry_Sector", 
                "Category", "Description", "Date"
            ]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ""
            
            # Reorder columns to match FoundersDay format
            df = df[required_columns]
            
            # Remove duplicates
            df.drop_duplicates(subset=["Company_Name"], keep="first", inplace=True)
            
            # Save to CSV
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n✅ GrowthList Complete: {len(df)} companies saved to {output_csv}")
            
            # Show sample data
            print("\n📊 Sample Data (first 3 rows):")
            print(df[['Company_Name', 'Industry_Sector', 'Country', 'Funding_Amount', 'Date']].head(3).to_string(index=False))
            
            return len(df)
        else:
            print("\n⚠️ No GrowthList data extracted")
            return 0
    
    except Exception as e:
        print(f"\n❌ Error in GrowthList scraping: {e}")
        import traceback
        traceback.print_exc()
        return 0
# ==================== GROWTHLIST HELPER FUNCTIONS ====================
def search_growthlist_page(driver, query):
    """Searches Google for the correct GrowthList link for a given query."""
    print(f"\n🔍 Searching GrowthList for: {query}")
    driver.get("https://www.google.com/")
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "q"))).send_keys(query + Keys.RETURN)
    time.sleep(3)

    links = driver.find_elements(By.XPATH, "//a[contains(@href, 'growthlist.co')]")
    for link in links:
        href = link.get_attribute("href")
        if href and "growthlist.co" in href and "google" not in href:
            print(f"🔗 Opening: {href}")
            driver.get(href)
            time.sleep(5)
            return True

    print("⚠️ Could not find GrowthList link.")
    return False


def set_growthlist_entries(driver, value="100"):
    """Sets dropdown to 100 entries per page on GrowthList."""
    try:
        print(f"📋 Setting dropdown to show {value} entries...")
        dropdown = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select[name$='_length']"))
        )
        driver.execute_script(
            f"arguments[0].value='{value}'; arguments[0].dispatchEvent(new Event('change'));",
            dropdown
        )
        time.sleep(3)
        print(f"✅ Dropdown set to {value} entries.")
        return True
    except Exception as e:
        print(f"⚠️ Could not set dropdown to {value}: {e}")
        return False
# ==================== HELPER FUNCTION FOR FUNDING FORMATTING ====================
def format_funding_amount(amount_str):
    """
    Converts funding amounts to M/B notation (e.g., $3.65M, $2.5B)
    
    Args:
        amount_str: String like "$3,653,876" or "$2,500,000,000"
    
    Returns:
        Formatted string like "$3.65M" or "$2.5B"
    """
    if not amount_str or amount_str.strip() == "":
        return ""
    
    # Remove currency symbols and commas
    clean_amount = amount_str.replace('$', '').replace(',', '').replace('USD', '').strip()
    
    # Try to convert to number
    try:
        value = float(clean_amount)
        
        # Convert to billions or millions
        if value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value / 1_000:.2f}K"
        else:
            return f"${value:.2f}"
    except:
        # If conversion fails, return original
        return amount_str

def extract_growthlist_table(driver):
    """
    Extracts complete data from GrowthList table with CORRECT column mapping:
    
    Column 0: Name → Company_Name ✅
    Column 1: Website → SKIP ❌
    Column 2: Industry → Industry_Sector ✅
    Column 3: Country → Country ✅
    Column 4: Funding Amount (USD) → Funding_Amount ✅
    Column 5: Funding Type → SKIP ❌
    Column 6: Last Funding Date → Date ✅
    """
    all_data = []
    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    
    print(f"🔍 Found {len(rows)} rows in table")
    
    for idx, row in enumerate(rows, 1):
        try:
            cols = row.find_elements(By.TAG_NAME, "td")
            
            # Debug: Print column count and values for first row
            if idx == 1:
                print(f"📊 Table has {len(cols)} columns")
                print(f"📋 Column mapping verification:")
                column_names = ["Name", "Website", "Industry", "Country", "Funding Amount", "Funding Type", "Last Funding Date"]
                for i, col in enumerate(cols[:7]):
                    col_name = column_names[i] if i < len(column_names) else f"Column {i}"
                    print(f"   Column {i} ({col_name}): {col.text.strip()[:50]}")
            
            if len(cols) >= 7:
                # Column 0: Name → Company_Name
                company_name = cols[0].text.strip() if cols[0].text else None
                
                # Column 1: Website → SKIP
                
                # Column 2: Industry → Industry_Sector
                industry = cols[2].text.strip() if cols[2].text else None
                
                # Column 3: Country → Country
                country = cols[3].text.strip() if cols[3].text else None
                
                # Column 4: Funding Amount (USD) → Funding_Amount (with formatting)
                funding_amount_raw = cols[4].text.strip() if cols[4].text else None
                funding_amount = format_funding_amount(funding_amount_raw) if funding_amount_raw else None
                
                # Column 5: Funding Type → SKIP
                
                # Column 6: Last Funding Date → Date
                funding_date = cols[6].text.strip() if cols[6].text else None
                
                # Only add if company name exists
                if company_name:
                    all_data.append({
                        "Company_Name": company_name,
                        "Industry_Sector": industry if industry else "",
                        "Country": country if country else "",
                        "Funding_Amount": funding_amount if funding_amount else "",
                        "Date": funding_date if funding_date else ""
                    })
                    
                    # Debug: Print first 3 entries with correct mapping
                    if idx <= 3:
                        print(f"  ✅ Row {idx}:")
                        print(f"     Company: {company_name}")
                        print(f"     Industry: {industry}")
                        print(f"     Country: {country}")
                        print(f"     Funding: {funding_amount}")
                        print(f"     Date: {funding_date}")
        except Exception as e:
            print(f"⚠️ Error extracting row {idx}: {e}")
            continue
    
    print(f"✅ Extracted {len(all_data)} valid rows from this page.")
    return all_data

def go_to_next_page(driver, current_page):
    """Goes to the next page of GrowthList results."""
    try:
        first_name = ""
        try:
            first_name = driver.find_element(By.CSS_SELECTOR, "table tbody tr:first-child td:first-child").text.strip()
        except:
            pass

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        next_page = str(current_page + 1)
        try:
            next_button = WebDriverWait(driver, 4).until(
                EC.element_to_be_clickable((By.LINK_TEXT, next_page))
            )
            driver.execute_script("arguments[0].click();", next_button)
        except:
            try:
                next_button = WebDriverWait(driver, 4).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'Next') or contains(text(),'»')]"))
                )
                driver.execute_script("arguments[0].click();", next_button)
            except:
                return False

        WebDriverWait(driver, 10).until_not(
            EC.text_to_be_present_in_element((By.CSS_SELECTOR, "table tbody tr:first-child td:first-child"), first_name)
        )
        time.sleep(2)
        return True
    except:
        return False
# ==================== MAIN PROCESS_LEADS FUNCTION ====================
# ==================== MAIN PROCESS_LEADS FUNCTION ====================

# ==================== UPDATED PROCESS_LEADS (REMOVE SOURCE COLUMN) ====================
def process_leads(region, category, api_key_1, api_key_2, llm_name, output_csv):
    """
    Main function to process leads based on region and category with API key rotation.
    Combines FoundersDay and GrowthList data sources.
    NO SOURCE COLUMN in output.
    
    Args:
        region: Region to scrape (India/Global)
        category: Category/sector to scrape (all/fintech/healthtech/etc)
        api_key_1: Primary API key
        api_key_2: Secondary API key (optional, will use key_1 if not provided)
        llm_name: LLM provider name (Mistral/Claude/Openai/etc)
        output_csv: Output CSV filename
    
    Returns:
        tuple: (dataframe, csv_path) or (None, None) on error
    """
    
    # Validate API keys
    print("\n🔍 VALIDATING RECEIVED API KEYS:")
    print(f"  • api_key_1: {api_key_1[:10] if api_key_1 else '❌ EMPTY'}...")
    print(f"  • api_key_2: {api_key_2[:10] if api_key_2 else '⚠️ EMPTY'}...")
    print(f"  • llm_name: {llm_name}")
    print(f"  • region: {region}")
    print(f"  • category: {category}")
    
    if not api_key_1:
        print("❌ CRITICAL ERROR: Primary API key is empty!")
        raise ValueError("Primary API key (api_key_1) cannot be empty")

    # Initialize API keys
    try:
        initialize_api_keys(api_key_1, api_key_2, llm_name)
    except Exception as e:
        print(f"❌ Failed to initialize API keys: {e}")
        raise

    print("\n" + "=" * 80)
    print("🎯 PROCESS LEADS - MONTHLY FUNDING DATA SCRAPER")
    print("=" * 80)
    print(f"🌍 Region: {region}")
    print(f"📂 Category: {category}")
    print(f"🤖 LLM Provider: {llm_name}")
    print(f"🔑 API Keys: Initialized with rotation support")
    print(f"📁 Output File: {output_csv}")
    print("=" * 80)
    print("\n📋 SCRAPING STRATEGY:")
    print("  • FoundersDay scraping with LLM extraction")
    print("  • GrowthList scraping with pagination")
    print("  • Combined data from both sources")
    print("  • Founder name validation against company names")
    print("  • Date extraction from funding announcements")
    print("=" * 80)

    try:
        # ========== BUILD URL CONFIGURATION ==========
        fd_base_url, fd_max_pages, fd_url_generator, gl_query = build_url_config(region, category)
        
        foundersday_count = 0
        growthlist_count = 0
        
        # ========== PHASE 1A: SCRAPE FOUNDERSDAY ==========
        if fd_base_url:
            print("\n" + "=" * 80)
            print("📊 PHASE 1A: SCRAPING FOUNDERSDAY")
            print("=" * 80)
            fd_temp_csv = output_csv.replace('.csv', '_foundersday_temp.csv')
            foundersday_count = scrape_foundersday(region, category, fd_temp_csv)
        else:
            print("\n⚠️ FoundersDay scraping skipped for this region/category combination")
        
        # ========== PHASE 1B: SCRAPE GROWTHLIST ==========
        if gl_query:
            print("\n" + "=" * 80)
            print("📊 PHASE 1B: SCRAPING GROWTHLIST")
            print("=" * 80)
            gl_temp_csv = output_csv.replace('.csv', '_growthlist_temp.csv')
            growthlist_count = scrape_growthlist(gl_query, region, gl_temp_csv, max_pages=10)
        else:
            print("\n⚠️ GrowthList scraping skipped for this region/category combination")
        
        # ========== COMBINE DATA FROM BOTH SOURCES (NO SOURCE COLUMN) ==========
        print("\n" + "=" * 80)
        print("🔄 COMBINING DATA FROM BOTH SOURCES")
        print("=" * 80)
        
        combined_df = pd.DataFrame()
        
        # Load FoundersDay data (NO SOURCE COLUMN)
        if foundersday_count > 0:
            try:
                fd_df = pd.read_csv(fd_temp_csv, encoding='utf-8-sig')
                # DO NOT ADD SOURCE COLUMN
                combined_df = pd.concat([combined_df, fd_df], ignore_index=True)
                print(f"✅ Loaded {len(fd_df)} companies from FoundersDay")
                os.remove(fd_temp_csv)
                print(f"🗑️ Removed temporary file: {fd_temp_csv}")
            except Exception as e:
                print(f"⚠️ Error loading FoundersDay data: {e}")
        
        # Load GrowthList data (NO SOURCE COLUMN)
        if growthlist_count > 0:
            try:
                gl_df = pd.read_csv(gl_temp_csv, encoding='utf-8-sig')
                # DO NOT ADD SOURCE COLUMN
                combined_df = pd.concat([combined_df, gl_df], ignore_index=True)
                print(f"✅ Loaded {len(gl_df)} companies from GrowthList")
                os.remove(gl_temp_csv)
                print(f"🗑️ Removed temporary file: {gl_temp_csv}")
            except Exception as e:
                print(f"⚠️ Error loading GrowthList data: {e}")
        
        # Check if we have any data
        if combined_df.empty:
            print("\n❌ No data was scraped from any source!")
            # Create empty CSV with proper columns (NO SOURCE)
            empty_df = pd.DataFrame(columns=[
                "Company_Name", "Funding_Amount", "Country", "Founder_Name",
                "Important_Person", "Email", "LinkedIn_URL", "Industry_Sector", 
                "Category", "Description", "Date"
            ])
            empty_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            return None, None
        
        # Remove duplicates
        print(f"\n🔍 Checking for duplicates...")
        initial_count = len(combined_df)
        combined_df.drop_duplicates(subset=["Company_Name"], keep="first", inplace=True)
        duplicates_removed = initial_count - len(combined_df)
        if duplicates_removed > 0:
            print(f"🗑️ Removed {duplicates_removed} duplicate companies")
        else:
            print(f"✅ No duplicates found")
        
        # Save combined data
        combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        # ========== DISPLAY STATISTICS (NO SOURCE BREAKDOWN) ==========
        print("\n" + "=" * 80)
        print("✅ DATA COMBINATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📊 COMBINED STATISTICS:")
        print(f"  • FoundersDay Companies: {foundersday_count}")
        print(f"  • GrowthList Companies: {growthlist_count}")
        print(f"  • Total Unique Companies: {len(combined_df)}")
        print(f"  • Duplicates Removed: {duplicates_removed}")
        print(f"  • Output CSV: {output_csv}")
        
        # Country distribution
        print("\n🌍 Country Distribution:")
        print(combined_df['Country'].value_counts().to_string())
        
        # Data completeness
        print("\n📋 Data Completeness:")
        
        def safe_string_count(df, column_name):
            try:
                return len(df[
                    (df[column_name].fillna("").astype(str).str.strip() != "") & 
                    (df[column_name].fillna("").astype(str) != "nan")
                ])
            except Exception as e:
                print(f"⚠️ Error counting {column_name}: {e}")
                return 0
        
        industry_count = safe_string_count(combined_df, 'Industry_Sector')
        founder_count = safe_string_count(combined_df, 'Founder_Name')
        date_count = safe_string_count(combined_df, 'Date')
        description_count = safe_string_count(combined_df, 'Description')
        
        total_companies = len(combined_df)
        print(f"  • Companies with Industry Info: {industry_count} ({industry_count/total_companies*100:.1f}%)")
        print(f"  • Companies with Founder Info: {founder_count} ({founder_count/total_companies*100:.1f}%)")
        print(f"  • Companies with Date Info: {date_count} ({date_count/total_companies*100:.1f}%)")
        print(f"  • Companies with Description: {description_count} ({description_count/total_companies*100:.1f}%)")
        
        # ============================================================
        # PHASE 2: ENRICH DATA WITH FOUNDER/CEO INFO
        # ============================================================
        print("\n" + "=" * 80)
        print("🔍 PHASE 2: ENRICHING DATA WITH FOUNDER/CEO INFORMATION")
        print("=" * 80)
        
        try:
            from selenium_extractor import extract_company_data
            
            # Generate enriched output filename
            enriched_output = output_csv.replace('.csv', '_enriched.xlsx')
            
            print(f"\n🤖 Starting enrichment process...")
            print(f"  • Input: {len(combined_df)} companies")
            print(f"  • Output will be saved to: {enriched_output}")
            
            # Call enrichment with the scraped DataFrame
            enriched_df, enriched_file = extract_company_data(
                csv_path=combined_df,
                llm_provider=llm_name,
                llm_api_keys=[api_key_1, api_key_2],
                output_path=enriched_output,
                company_column='Company_Name',
                apollo_api_key=None
            )
            
            if enriched_df is not None and not enriched_df.empty:
                print(f"\n✅ ENRICHMENT SUCCESSFUL!")
                print(f"  • Enriched companies: {len(enriched_df)}")
                print(f"  • Enriched file: {enriched_file}")
                
                # Display enrichment statistics
                print("\n📊 ENRICHMENT STATISTICS:")
                if 'Founder/CEO Name' in enriched_df.columns:
                    enriched_founders = len(enriched_df[enriched_df['Founder/CEO Name'].notna() & (enriched_df['Founder/CEO Name'].str.len() > 0)])
                    print(f"  • Companies with Founder/CEO: {enriched_founders} ({enriched_founders/len(enriched_df)*100:.1f}%)")
                
                if 'Founder/CEO LinkedIn' in enriched_df.columns:
                    enriched_linkedin = len(enriched_df[enriched_df['Founder/CEO LinkedIn'].notna() & (enriched_df['Founder/CEO LinkedIn'].str.len() > 0)])
                    print(f"  • Companies with LinkedIn: {enriched_linkedin} ({enriched_linkedin/len(enriched_df)*100:.1f}%)")
                
                if 'Founder/CEO Email' in enriched_df.columns:
                    enriched_emails = len(enriched_df[enriched_df['Founder/CEO Email'].notna() & (enriched_df['Founder/CEO Email'].str.len() > 0)])
                    print(f"  • Companies with Email: {enriched_emails} ({enriched_emails/len(enriched_df)*100:.1f}%)")
                
                print("\n" + "=" * 80)
                print("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
                print("=" * 80)
                print(f"\n📁 FILES GENERATED:")
                print(f"  1. Raw Data: {output_csv}")
                print(f"  2. Enriched Data: {enriched_file}")
                print("\n✨ Ready to use!")
                print("=" * 80)
                
                # Return enriched data
                return enriched_df, enriched_file
            else:
                print(f"\n⚠️ Enrichment returned empty data, returning raw scraped data")
                print(f"📁 Raw data available at: {output_csv}")
                return combined_df, output_csv
                
        except ImportError:
            print(f"\n⚠️ selenium_extractor module not found, skipping enrichment")
            print("   Returning raw scraped data")
            print(f"📁 Raw data available at: {output_csv}")
            return combined_df, output_csv
            
        except Exception as e:
            print(f"\n⚠️ Enrichment failed: {e}")
            print("   Returning raw scraped data instead")
            print(f"📁 Raw data available at: {output_csv}")
            import traceback
            traceback.print_exc()
            return combined_df, output_csv

    except Exception as e:
        print(f"\n❌ PROCESS LEADS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    finally:
        cleanup_selenium_driver()
# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Example usage
    result = process_leads(
        region="Global",
        category="AI",
        api_key_1="rL9fnBvN4URssy6kEb0GU3enQWCqS85f",
        api_key_2="IKn4XPXL6US8G1Hfr1WhetwhGKzEleNT",  # Optional
        llm_name="Mistral",
        output_csv="India_fintech_monthly.csv"
    )
    
    if result:
        df, csv_path = result
        print(f"\n✅ Successfully processed leads!")
        print(f"   Data saved to: {csv_path}")
    else:
        print(f"\n❌ Failed to process leads")