#this is main.py
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import re

# Your LLM configurations
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


def call_llm(llm_name, api_key, prompt, max_tokens=200):
   
    # ✅ FIX: Case-insensitive lookup
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
        # Claude has special format
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
        # OpenAI-compatible format for all others
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
        print(f"🤖 Calling {matched_key} API...")
        response = requests.post(config["url"], headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Extract content based on provider
        if matched_key.lower() == "claude":
            content = result['content'][0]['text']
        else:
            content = result['choices'][0]['message']['content']
        
        print(f"✅ {matched_key} responded successfully")
        return content
        
    except Exception as e:
        print(f"❌ Error calling {matched_key}: {e}")
        return None


def get_timeframe_dates(frequency, timeframe):
    """
    Convert frequency to actual date ranges
    Returns formatted date string based on frequency
    """
    today = datetime.now()
    
    if frequency == "WEEKLY":
        # Parse specific week requests from timeframe
        timeframe_lower = timeframe.lower()
        
        # Extract month and year if mentioned
        target_month = today.month
        target_year = today.year
        
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in timeframe_lower:
                target_month = month_num
                break
        
        # Extract year if mentioned
        year_match = re.search(r'20\d{2}', timeframe)
        if year_match:
            target_year = int(year_match.group())
        
        # Determine which week is requested
        if 'first week' in timeframe_lower or '1st week' in timeframe_lower:
            # First complete Monday-Sunday of the month
            first_day = datetime(target_year, target_month, 1)
            days_until_monday = (7 - first_day.weekday()) % 7
            if days_until_monday == 0 and first_day.weekday() == 0:
                first_monday = first_day
            else:
                first_monday = first_day + timedelta(days=days_until_monday)
            last_sunday = first_monday + timedelta(days=6)
            
        elif 'second week' in timeframe_lower or '2nd week' in timeframe_lower:
            # Second complete Monday-Sunday of the month
            first_day = datetime(target_year, target_month, 1)
            days_until_monday = (7 - first_day.weekday()) % 7
            if days_until_monday == 0 and first_day.weekday() == 0:
                first_monday = first_day + timedelta(days=7)
            else:
                first_monday = first_day + timedelta(days=days_until_monday + 7)
            last_sunday = first_monday + timedelta(days=6)
            
        elif 'third week' in timeframe_lower or '3rd week' in timeframe_lower:
            # Third complete Monday-Sunday of the month
            first_day = datetime(target_year, target_month, 1)
            days_until_monday = (7 - first_day.weekday()) % 7
            if days_until_monday == 0 and first_day.weekday() == 0:
                first_monday = first_day + timedelta(days=14)
            else:
                first_monday = first_day + timedelta(days=days_until_monday + 14)
            last_sunday = first_monday + timedelta(days=6)
            
        elif 'last week' in timeframe_lower or 'fourth week' in timeframe_lower or '4th week' in timeframe_lower:
            # Last complete Monday-Sunday of the month
            if target_month == 12:
                next_month = datetime(target_year + 1, 1, 1)
            else:
                next_month = datetime(target_year, target_month + 1, 1)
            
            last_day = next_month - timedelta(days=1)
            days_from_sunday = (last_day.weekday() + 1) % 7
            last_sunday = last_day - timedelta(days=days_from_sunday)
            first_monday = last_sunday - timedelta(days=6)
            
        elif 'second last week' in timeframe_lower or 'penultimate week' in timeframe_lower:
            # Second last complete Monday-Sunday of the month
            if target_month == 12:
                next_month = datetime(target_year + 1, 1, 1)
            else:
                next_month = datetime(target_year, target_month + 1, 1)
            
            last_day = next_month - timedelta(days=1)
            days_from_sunday = (last_day.weekday() + 1) % 7
            last_sunday = last_day - timedelta(days=days_from_sunday + 7)
            first_monday = last_sunday - timedelta(days=6)
            
        else:
            # Default: Get previous complete week
            current_weekday = today.weekday()
            if today.day < 7 or current_weekday < 3:
                first_monday = today - timedelta(days=current_weekday + 7)
            else:
                first_monday = today - timedelta(days=current_weekday + 7)
            last_sunday = first_monday + timedelta(days=6)
        
        # Format the date range
        if first_monday.month != last_sunday.month:
            return f"{first_monday.strftime('%d %b')} - {last_sunday.strftime('%d %b %Y')}"
        else:
            return f"{first_monday.strftime('%d')}-{last_sunday.strftime('%d %B %Y')}"
    
    elif frequency == "MONTHLY":
        # Parse specific month from timeframe
        timeframe_lower = timeframe.lower()
        target_year = today.year
        
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        target_month = None
        for month_name, month_num in months.items():
            if month_name in timeframe_lower:
                target_month = month_num
                break
        
        # Extract year if mentioned
        year_match = re.search(r'20\d{2}', timeframe)
        if year_match:
            target_year = int(year_match.group())
        
        if target_month:
            return datetime(target_year, target_month, 1).strftime('%B %Y')
        else:
            # Default behavior: If less than 7 days into month, use previous month
            if today.day < 7:
                first_day_current = today.replace(day=1)
                last_day_previous = first_day_current - timedelta(days=1)
                return last_day_previous.strftime('%B %Y')
            else:
                return today.strftime('%B %Y')
    
    elif frequency == "YEARLY":
        year_match = re.search(r'20\d{2}', timeframe)
        if year_match:
            return year_match.group()
        return str(today.year)
    
    return "Date range not determined"


def analyze_intent(timeframe, sector, region, llm_name, api_key):
    """
    Analyze timeframe to determine if user wants weekly, monthly, or yearly data
    Uses the specified LLM dynamically
    
    Args:
        timeframe: The timeframe string to analyze
        sector: The sector/industry
        region: The geographic region
        llm_name: Name of the LLM to use (mistral, claude, openai, etc.)
        api_key: API key for the specified LLM
    
    Returns: dict with 'frequency', 'region_category', and 'intent'
    """
    
    prompt = f"""Analyze this query for a funding report:
Timeframe: "{timeframe}"
Sector: "{sector}"
Region: "{region}"

Determine:
1. frequency: Is this a WEEKLY, MONTHLY, or YEARLY data request?
   - WEEKLY: If the query explicitly mentions "weekly" or requests a specific week
   - MONTHLY: If the query mentions a specific month (like September, October, etc.) or "monthly"
   - YEARLY: If the query mentions a year without a specific month or "yearly/annual"

2. region_category: Classify the region as either "Indian" or "Global"
   - Indian: If region mentions India, Indian, or specific Indian cities/states
   - Global: For ANY other region (USA, Europe, Asia, worldwide, global, etc.)

3. intent: What is the user trying to accomplish? Include the region category and sector.

Examples:
- Region "India" -> region_category: "Indian"
- Region "USA" -> region_category: "Global"
- Region "Europe" -> region_category: "Global"
- Region "Global" -> region_category: "Global"
- Region "Mumbai" -> region_category: "Indian"

Return ONLY a JSON object:
{{"frequency": "WEEKLY/MONTHLY/YEARLY", "region_category": "Indian/Global", "intent": "brief description"}}"""

    print(f"🤖 Using {llm_name} for intent analysis...")
    
    try:
        # Use the dynamic call_llm function
        content = call_llm(llm_name, api_key, prompt, max_tokens=200)
        
        if not content:
            raise Exception(f"No response from {llm_name}")
        
        # Parse JSON from response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)
        
        # Add fallback logic in case LLM doesn't classify correctly
        region_lower = region.lower()
        if 'india' in region_lower or 'indian' in region_lower or \
           'mumbai' in region_lower or 'delhi' in region_lower or \
           'bangalore' in region_lower or 'bengaluru' in region_lower:
            result['region_category'] = 'Indian'
        else:
            result['region_category'] = 'Global'
        
        print(f"✅ Analysis result: {result}")
        return result
        
    except Exception as e:
        print(f"⚠️ Error analyzing intent with {llm_name}: {e}")
        # Fallback: Try to determine frequency from timeframe string
        timeframe_lower = timeframe.lower()
        if 'week' in timeframe_lower:
            frequency = 'WEEKLY'
        elif any(month in timeframe_lower for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                          'july', 'august', 'september', 'october', 'november', 'december']):
            frequency = 'MONTHLY'
        else:
            frequency = 'YEARLY'
        
        region_lower = region.lower()
        region_category = 'Indian' if 'india' in region_lower else 'Global'
        
        return {
            "frequency": frequency, 
            "region_category": region_category,
            "intent": f"Fallback: {frequency} report for {sector} in {region}"
        }


def sanitize_filename(text):
    """Helper function for filename sanitization"""
    if not text:
        return "unknown"
    text = re.sub(r'[<>:"/\\|?*]', '', str(text))
    text = text.replace(" ", "_")
    text = re.sub(r'[^\w\-_]', '', text)
    return text.lower().strip() or "unknown"


def start(sector, region, timeframe, mistral_key_1, mistral_key_2, llm_config="mistral"):
    """
    Main processing function with dynamic LLM support
    
    Args:
        llm_config: Either a string like "mistral" OR a dict with 'provider' key
    """
    
    print("\n" + "="*80)
    print("🎯 START FUNCTION CALLED IN MAIN.PY")
    print("="*80)
    
    print("\n📦 RECEIVED PARAMETERS:")
    print(f"  • Sector: {sector}")
    print(f"  • Region: {region}")
    print(f"  • Timeframe: {timeframe}")
    print(f"  • LLM: {llm_config}")
    
    # ✅ FIX: Extract provider name and keep original case
    if isinstance(llm_config, dict):
        llm_name = llm_config.get('provider', 'Mistral')  
        if 'api_key' in llm_config:
            mistral_key_1 = llm_config['api_key']
    else:
        llm_name = str(llm_config)
    
    # ✅ FIX: Case-insensitive validation
    matched_llm = None
    for key in LLM_CONFIGS.keys():
        if key.lower() == llm_name.lower():
            matched_llm = key
            break
    
    if matched_llm is None:
        print(f"\n❌ ERROR: LLM '{llm_name}' not supported")
        print(f"Available LLMs: {', '.join(LLM_CONFIGS.keys())}")
        return None
    
    # Use the correctly-cased key from dictionary
    llm_name = matched_llm
    
    
    # ✅ ADD VALIDATION
    print(f"\n🔑 API KEY VALIDATION:")
    print(f"  • Key 1 received: {'✓ Yes' if mistral_key_1 else '✗ EMPTY'}")
    print(f"  • Key 1 length: {len(mistral_key_1) if mistral_key_1 else 0} chars")
    if mistral_key_1:
     print(f"  • Key 1 preview: {mistral_key_1[:10]}...{mistral_key_1[-4:]}")
    print(f"  • Key 2 received: {'✓ Yes' if mistral_key_2 else '⚠️ EMPTY'}")
    print(f"  • Key 2 length: {len(mistral_key_2) if mistral_key_2 else 0} chars")
    if mistral_key_2:
     print(f"  • Key 2 preview: {mistral_key_2[:10]}...{mistral_key_2[-4:]}")

    if not mistral_key_1:
     print("\n❌ CRITICAL ERROR: Primary API key is empty in main.start()!")
     raise ValueError("mistral_key_1 cannot be empty")
    
    print(f"\n🔄 Analyzing intent with {llm_name.upper()} LLM...")
    
    # Use llm_name (the matched key with correct case)
    analysis = analyze_intent(timeframe, sector, region, llm_name, mistral_key_1)
    
    # Get actual date range based on frequency
    date_range = get_timeframe_dates(analysis['frequency'], timeframe)
    analysis['date_range'] = date_range
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"  • Frequency: {analysis['frequency']}")
    print(f"  • Region Category: {analysis['region_category']}")
    print(f"  • Date Range: {date_range}")
    print(f"  • Intent: {analysis['intent']}")
    print(f"  • 🔍 Researching for: {date_range}")
    print("="*80 + "\n")
    
    timeframe = date_range
    csv_file = None
    df_result = None
    
    # ============================================================
    # WEEKLY FREQUENCY HANDLING
    # ============================================================
    if analysis['frequency'] == 'WEEKLY':
        print("\n" + "="*80)
        print("📅 WEEKLY DATA COLLECTION")
        print("="*80)
        
        
        from weekly import process_leads

        print("\n🔄 CALLING weekly.process_leads()...")
        print(f"  • Region: {region}")
        print(f"  • Passing mistral_key_1: {mistral_key_1[:10] if mistral_key_1 else '❌ EMPTY'}...")
        print(f"  • Passing mistral_key_2: {mistral_key_2[:10] if mistral_key_2 else '⚠️ EMPTY'}...")
        print(f"  • LLM name: {llm_name}")

        result = process_leads(
    region=region,
    api_key_1=mistral_key_1,
    api_key_2=mistral_key_2,
    llm_name=llm_name
)
        
        print(f"\n📊 Weekly result type: {type(result)}")
        
        if isinstance(result, tuple) and len(result) == 2:
            df_result, csv_file = result
            
            if df_result is not None and not df_result.empty:
                print(f"\n✅ WEEKLY DATA SUCCESSFULLY PROCESSED & ENRICHED")
                print(f"  • Records: {len(df_result)}")
                print(f"  • Columns: {list(df_result.columns)}")
                print(f"  • Enriched file: {csv_file}")
                
                print(f"\n📋 Sample data (first 3 rows):")
                print(df_result.head(3).to_string())
            else:
                print("\n⚠️ Empty or None DataFrame returned from weekly processing")
                csv_file = None
                df_result = None
        
        elif isinstance(result, pd.DataFrame):
            df_result = result
            if not df_result.empty:
                timeframe_clean = timeframe.replace(' ', '_').replace('/', '_')
                csv_file = f"weekly_enriched_{timeframe_clean}.csv"
                df_result.to_csv(csv_file, index=False)
                print(f"\n✅ Weekly data saved: {csv_file} ({len(df_result)} records)")
            else:
                print("\n⚠️ Empty DataFrame returned")
                csv_file = None
        
        elif isinstance(result, str):
            csv_file = result
            try:
                if csv_file.endswith('.xlsx'):
                    df_result = pd.read_excel(csv_file)
                else:
                    df_result = pd.read_csv(csv_file)
                print(f"\n✅ Weekly data loaded from file: {len(df_result)} records")
            except Exception as e:
                print(f"\n⚠️ Could not read file: {e}")
                df_result = None
                csv_file = None
        
        else:
            print("\n⚠️ No data returned from weekly processing")
            csv_file = None
            df_result = None
    
    # ============================================================
    # MONTHLY FREQUENCY HANDLING
    # ============================================================






    # ============================================================
    elif analysis['frequency'] == 'MONTHLY':
     print("\n" + "="*80)
     print("📅 MONTHLY DATA COLLECTION")
     print("="*80)
    
     from monthly import process_leads

     print("\n🔄 CALLING monthly.process_leads()...")
     print(f"  • Region: {region}")
     print(f"  • Category: {sector}")
     print(f"  • Passing api_key_1: {mistral_key_1[:10] if mistral_key_1 else '❌ EMPTY'}...")
     print(f"  • Passing api_key_2: {mistral_key_2[:10] if mistral_key_2 else '⚠️ EMPTY'}...")
     print(f"  • LLM name: {llm_name}")

    # Call process_leads
     result = process_leads(
        region=region,
        category=sector,
        api_key_1=mistral_key_1,
        api_key_2=mistral_key_2,
        llm_name=llm_name,
        output_csv=f" {region} monthly.csv"
    ) 
     

      # Debug: Print detailed result information
    print(f"\n🔍 DEBUG: Result received from monthly.process_leads()")
    print(f"  • Type: {type(result)}")
    print(f"  • Value: {result if not isinstance(result, pd.DataFrame) else f'DataFrame with {len(result)} rows'}")
    
    if isinstance(result, tuple):
        print(f"  • Tuple length: {len(result)}")
        if len(result) >= 1:
            print(f"  • Element 0 type: {type(result[0])}")
            print(f"  • Element 0 value: {result[0] if not isinstance(result[0], pd.DataFrame) else f'DataFrame with {len(result[0])} rows'}")
        if len(result) >= 2:
            print(f"  • Element 1 type: {type(result[1])}")
            print(f"  • Element 1 value: {result[1]}")
    if isinstance(result, tuple) and len(result) == 2:
        df_result, csv_file = result
        
        print(f"\n✅ Received tuple with 2 elements")
        print(f"  • DataFrame: {df_result is not None} ({len(df_result) if df_result is not None else 0} rows)")
        print(f"  • File: {csv_file}")
        
        if df_result is not None and not df_result.empty and csv_file:
            print(f"\n✅ MONTHLY DATA SUCCESSFULLY PROCESSED & ENRICHED")
            print(f"  • Records: {len(df_result)}")
            print(f"  • Columns: {list(df_result.columns)}")
            print(f"  • File: {csv_file}")
            
            # Verify file exists
            import os
            if os.path.exists(csv_file):
                print(f"  • ✓ File exists on disk")
            else:
                print(f"  • ⚠️ File NOT found on disk, creating it...")
                if csv_file.endswith('.xlsx'):
                    df_result.to_excel(csv_file, index=False)
                else:
                    df_result.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            print(f"\n📋 Sample data (first 3 rows):")
            print(df_result.head(3).to_string())
            
            # If file is .xlsx, also create a .csv version for better compatibility
            if csv_file.endswith('.xlsx'):
                csv_version = csv_file.replace('.xlsx', '.csv')
                df_result.to_csv(csv_version, index=False, encoding='utf-8-sig')
                print(f"\n📁 Also saved as CSV: {csv_version}")
                # Use CSV version as primary
                csv_file = csv_version
        
        elif df_result is not None and not df_result.empty and not csv_file:
            # We have data but no file path - create one
            print(f"\n⚠️ DataFrame received but no file path, creating file...")
            sector_clean = sanitize_filename(sector)
            region_clean = sanitize_filename(region)
            csv_file = f"{region_clean}_{sector_clean}_monthly_enriched.csv"
            df_result.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"  • Created file: {csv_file}")
        
        else:
            print("\n⚠️ Empty or invalid data returned from monthly processing")
            csv_file = None
            df_result = None
    
    # Handle DataFrame only return (backward compatibility)
    elif isinstance(result, pd.DataFrame):
        print(f"\n⚠️ Received bare DataFrame (not tuple) - backward compatibility mode")
        df_result = result
        if not df_result.empty:
            sector_clean = sanitize_filename(sector)
            region_clean = sanitize_filename(region)
            csv_file = f"{region_clean}_{sector_clean}_monthly_enriched.csv"
            df_result.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✅ Monthly data saved: {csv_file} ({len(df_result)} records)")
        else:
            print("\n⚠️ Empty DataFrame returned")
            csv_file = None
            df_result = None
    
    # Handle file path string return (backward compatibility)
    elif isinstance(result, str):
        print(f"\n⚠️ Received string (file path) - backward compatibility mode")
        csv_file = result
        try:
            if csv_file.endswith('.xlsx'):
                df_result = pd.read_excel(csv_file)
                # Create CSV version
                csv_version = csv_file.replace('.xlsx', '.csv')
                df_result.to_csv(csv_version, index=False, encoding='utf-8-sig')
                csv_file = csv_version
                print(f"✅ Monthly data converted to CSV: {csv_file}")
            else:
                df_result = pd.read_csv(csv_file, encoding='utf-8-sig')
            print(f"  • Loaded {len(df_result)} records from file")
        except Exception as e:
            print(f"\n❌ Could not read file '{csv_file}': {e}")
            import traceback
            traceback.print_exc()
            df_result = None
            csv_file = None
    
    # Handle None return (error case)
    elif result is None or (isinstance(result, tuple) and result[0] is None):
        print("\n❌ Monthly processing returned None - no data generated")
        print("   Check the logs above for errors in monthly.py")
        csv_file = None
        df_result = None
    
    else:
        print(f"\n❌ Unexpected return type from monthly processing")
        print(f"   Type: {type(result)}")
        print(f"   Value: {result}")
        csv_file = None
        df_result = None
    
    # Final verification
    print(f"\n🔍 FINAL STATE CHECK:")
    print(f"  • df_result: {df_result is not None} ({len(df_result) if df_result is not None else 0} rows)")
    print(f"  • csv_file: {csv_file}")
    if csv_file:
        import os
        print(f"  • File exists: {os.path.exists(csv_file)}")

   
    # ============================================================
    # YEARLY FREQUENCY HANDLING
    # ============================================================
    elif analysis['frequency'] == 'YEARLY':
        print("\n" + "="*80)
        print("📅 YEARLY DATA COLLECTION")
        print("="*80)
        
        from Grwothlist import growthlist
        
        print(f"\n🔄 Starting yearly data collection...")
        print(f"  • Sector: {sector}")
        print(f"  • Region: {region}")
        print(f"  • Year: {date_range}")
        print(f"  • LLM: {llm_name}")
        
        # ✅ FIX: Pass llm_name (not llm_config)
        result = growthlist(sector, region, "yearly", mistral_key_1, mistral_key_2, llm_name=llm_name)
        
        print(f"\n📊 Yearly result type: {type(result)}")
        
        if isinstance(result, pd.DataFrame):
            df_result = result
            if not df_result.empty:
                sector_clean = sanitize_filename(sector)
                region_clean = sanitize_filename(region)
                csv_file = f"{sector_clean}_{region_clean}_yearly_startups_enhanced.csv"
                
                df_result.to_csv(csv_file, index=False)
                print(f"\n✅ YEARLY DATA SUCCESSFULLY PROCESSED")
                print(f"  • Records: {len(df_result)}")
                print(f"  • File: {csv_file}")
            else:
                print("\n⚠️ Empty DataFrame returned from yearly processing")
                csv_file = None
        
        elif isinstance(result, str):
            csv_file = result
            try:
                if csv_file.endswith('.xlsx'):
                    df_result = pd.read_excel(csv_file)
                    csv_file = csv_file.replace('.xlsx', '.csv')
                    df_result.to_csv(csv_file, index=False)
                else:
                    df_result = pd.read_csv(csv_file)
                print(f"\n✅ Yearly data loaded: {len(df_result)} records")
            except Exception as e:
                print(f"\n⚠️ Could not read file: {e}")
                df_result = None
                csv_file = None
        
        else:
            print("\n⚠️ Unexpected return type from yearly processing")
            csv_file = None
            df_result = None
    
    # ============================================================
    # FINAL SUMMARY AND RETURN
    # ============================================================
    print("\n" + "="*80)
    print("🎉 PROCESSING COMPLETE")
    print("="*80)
    
    if csv_file and df_result is not None:
        print(f"\n✅ FINAL OUTPUT:")
        print(f"  • File: {csv_file}")
        print(f"  • Records: {len(df_result)}")
        print(f"  • Columns: {', '.join(df_result.columns.tolist())}")
        print(f"  • LLM Used: {llm_name}")
        
        # Show statistics
        founder_col = None
        email_col = None
        linkedin_col = None
        
        for col in df_result.columns:
            if 'founder' in col.lower():
                founder_col = col
            if 'email' in col.lower():
                email_col = col
            if 'linkedin' in col.lower():
                linkedin_col = col
        
        if founder_col:
            founder_count = df_result[founder_col].notna().sum()
            print(f"  • Companies with founder info: {founder_count}")
        
        if email_col:
            email_count = df_result[email_col].notna().sum()
            print(f"  • Companies with email: {email_count}")
        
        if linkedin_col:
            linkedin_count = df_result[linkedin_col].notna().sum()
            print(f"  • Companies with LinkedIn: {linkedin_count}")
        
        print(f"\n📁 You can find the enriched data at: {csv_file}")
        print("="*80)
        
        return csv_file
    
    else:
        print("\n⚠️ WARNING: No file was generated")
        print("  Check the logs above for errors")
        print("="*80)
        return None
# ============================================================
if __name__ == "__main__":
    # Test configuration
    test_sector = "Technology"
    test_region = "India"
    test_timeframe = "this week"
    test_key_1 = "Cfjwf8ciml2yvA4g5I4aaYDw553qCX2L"
    test_key_2 = "your_secondary_key_here"
    test_llm = "Mistral"  # Change this to test different LLMs
    
    print("🧪 RUNNING MAIN.PY IN TEST MODE")
    print("="*80)
    
    result_file = start(
        sector=test_sector,
        region=test_region,
        timeframe=test_timeframe,
        mistral_key_1=test_key_1,
        mistral_key_2=test_key_2,
        llm=test_llm
    )
    
    if result_file:
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Result file: {result_file}")
    else:
        print(f"\n❌ Test failed - no result file generated")