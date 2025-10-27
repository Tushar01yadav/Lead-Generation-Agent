import streamlit as st
import requests
import json
import pandas as pd
import os
from datetime import datetime, timedelta
import random
import os
os.environ['STREAMLIT_DEBUG'] = '1'

# Add this helper function
def debug_llm_config(config, label="LLM Config"):
    """Debug helper to inspect LLM config"""
    print(f"\n🔍 {label}:")
    if isinstance(config, dict):
        for key, value in config.items():
            if 'key' in key.lower():
                print(f"  • {key}: {value[:10] if value else 'EMPTY'}...")
            else:
                print(f"  • {key}: {value}")
    else:
        print(f"  • Type: {type(config)}")
        print(f"  • Value: {config}")

# Page config with custom sidebar width
st.set_page_config(
    page_title="Lead Generation Agent", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar width
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        width: 19% !important;
    }
</style>
""", unsafe_allow_html=True)

# ============ UTILITY FUNCTIONS ============

def log_to_debug(message):
    """Add message to debug console"""
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    st.session_state.debug_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def get_default_endpoint(provider):
    """Get default API endpoint for provider"""
    endpoints = {
        "OpenAI": "https://api.openai.com/v1/chat/completions",
        "Claude": "https://api.anthropic.com/v1/messages",
        "Gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        "Mistral": "https://api.mistral.ai/v1/chat/completions",
        "Deepseek": "https://api.deepseek.com/v1/chat/completions",
        "Qwen": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "Perplexity": "https://api.perplexity.ai/chat/completions",
        "Llama": "https://api.together.xyz/v1/chat/completions"
    }
    return endpoints.get(provider, "")

def get_active_endpoint(config, key_num=1):
    """Get the active endpoint for a config - custom if set, otherwise default"""
    if key_num == 1:
        custom = config.get("custom_endpoint", "")
        return custom if custom else get_default_endpoint(config["provider"])
    else:
        custom = config.get("custom_endpoint_2", "")
        return custom if custom else get_default_endpoint(config["provider"])

def test_llm_connection(provider, api_key, endpoint):
    """Make a simple test call to validate LLM API key with specific endpoint"""
    try:
        log_to_debug(f"Testing {provider} connection at {endpoint}...")
        
        if provider == "OpenAI":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Claude":
            response = requests.post(
                endpoint,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Gemini":
            if "?key=" in endpoint:
                test_endpoint = endpoint
            else:
                test_endpoint = f"{endpoint}?key={api_key}"
            response = requests.post(
                test_endpoint,
                json={"contents": [{"parts": [{"text": "test"}]}]},
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Mistral":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "mistral-small",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Deepseek":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Qwen":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "qwen-turbo",
                    "input": {"messages": [{"role": "user", "content": "test"}]}
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("message", "Error"))
        
        elif provider == "Perplexity":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Llama":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "meta-llama/Llama-3-8b-chat-hf",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        return False, "Unknown provider"
    
    except requests.exceptions.Timeout:
        log_to_debug(f"❌ {provider} connection timeout")
        return False, "Request timeout - check your connection"
    except Exception as e:
        log_to_debug(f"❌ {provider} error: {str(e)}")
        return False, str(e)

# ============ GOOGLE SHEETS FUNCTIONS ============

def authenticate_google_sheets(credentials_json):
    """
    Authenticate with Google Sheets using OAuth credentials
    Returns service object for API calls
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        import pickle
        
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        
        creds = None
        
        # Check if token.pickle exists
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Save credentials to temp file
                with open('temp_credentials.json', 'w') as f:
                    json.dump(credentials_json, f)
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    'temp_credentials.json', SCOPES)
                
                # USE PORT 0 for automatic port selection
                creds = flow.run_local_server(
                    port=0,  # Let the OS pick an available port
                    open_browser=True
                )
                
                # Remove temp file
                if os.path.exists('temp_credentials.json'):
                    os.remove('temp_credentials.json')
            
            # Save credentials for next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        service = build('sheets', 'v4', credentials=creds)
        log_to_debug("✅ Google Sheets authentication successful")
        return service
    
    except Exception as e:
        log_to_debug(f"❌ Google Sheets authentication error: {str(e)}")
        raise Exception(f"Authentication failed: {str(e)}")


def upload_to_google_sheets(service, df, sheet_name=None):
    """
    Upload DataFrame to Google Sheets
    Creates a new spreadsheet and returns the URL
    """
    try:
        if sheet_name is None:
            sheet_name = f"Lead Report {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        log_to_debug(f"📤 Creating new Google Sheet: {sheet_name}")
        
        # Create new spreadsheet
        spreadsheet = {
            'properties': {
                'title': sheet_name
            }
        }
        
        spreadsheet = service.spreadsheets().create(
            body=spreadsheet,
            fields='spreadsheetId,spreadsheetUrl'
        ).execute()
        
        spreadsheet_id = spreadsheet.get('spreadsheetId')
        spreadsheet_url = spreadsheet.get('spreadsheetUrl')
        
        log_to_debug(f"✅ Spreadsheet created: {spreadsheet_id}")
        
        # ===== FIX: Handle NaN values and prepare data =====
        # Replace NaN with empty string and convert to proper types
        df_clean = df.copy()
        
        # Replace NaN, None, and inf values with empty string
        df_clean = df_clean.fillna('')
        
        # Replace inf and -inf with empty string
        df_clean = df_clean.replace([float('inf'), float('-inf')], '')
        
        # Convert all values to strings to avoid JSON serialization issues
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
        
        # Replace 'nan' strings that might have been created
        df_clean = df_clean.replace('nan', '')
        df_clean = df_clean.replace('None', '')
        
        log_to_debug(f"✅ Cleaned DataFrame: removed NaN and converted to strings")
        
        # Prepare data for upload (headers + data rows)
        values = [df_clean.columns.tolist()] + df_clean.values.tolist()
        
        log_to_debug(f"📊 Prepared {len(values)} rows (including header) for upload")
        
        body = {
            'values': values
        }
        
        # Upload data
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Sheet1!A1',
            valueInputOption='RAW',
            body=body
        ).execute()
        
        log_to_debug(f"✅ Uploaded {result.get('updatedCells')} cells to Google Sheets")
        
        # Format header row
        requests = [
            {
                'repeatCell': {
                    'range': {
                        'sheetId': 0,
                        'startRowIndex': 0,
                        'endRowIndex': 1
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': {
                                'red': 0.2,
                                'green': 0.2,
                                'blue': 0.8
                            },
                            'textFormat': {
                                'foregroundColor': {
                                    'red': 1.0,
                                    'green': 1.0,
                                    'blue': 1.0
                                },
                                'bold': True
                            }
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat)'
                }
            },
            {
                'autoResizeDimensions': {
                    'dimensions': {
                        'sheetId': 0,
                        'dimension': 'COLUMNS',
                        'startIndex': 0,
                        'endIndex': len(df_clean.columns)
                    }
                }
            }
        ]
        
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': requests}
        ).execute()
        
        log_to_debug(f"✅ Applied formatting to Google Sheet")
        
        return spreadsheet_url
    
    except Exception as e:
        log_to_debug(f"❌ Error uploading to Google Sheets: {str(e)}")
        import traceback
        log_to_debug(f"📋 Full traceback:\n{traceback.format_exc()}")
        raise Exception(f"Upload failed: {str(e)}")

def get_sector_options(timeframe, region):
    """Get sector options based on timeframe and region selection"""
    if "Weekly Report" in timeframe:
        return ["ALL / GENERAL"]
    elif "Monthly Report" in timeframe and region == "India":
        return [
            "ALL / GENERAL",
            "AI",
            "B2B", 
            "B2C",
            "CLEANTECH",
            "D2C BRANDS",
            "DEEPTECH",
            "EDTECH",
            "FINTECH",
            "HEALTHCARE",
            "IOT",
            "PROPTECH",
            "SAAS"
        ]
    else:
        return [
            "All / General", 
            "Healthcare / Healthtech",
            "Real Estate / Proptech",
            "Education / Edtech",
            "AI : Artificial Intelligence"
        ]

def clear_history():
    """Clear all session state"""
    st.session_state.extracted_info = None
    st.session_state.output_logs = None
    st.session_state.df = None
    st.session_state.csv_filename = None
    st.session_state.enriched_df = None

def extract_with_apollo(df, apollo_api_key):
    """
    Extract contact information (email and phone) from LinkedIn URLs using Apollo.io API
    """
    import time
    import re
    
    log_to_debug(f"🔍 Starting email extraction with Apollo API...")
    log_to_debug(f"📊 Processing {len(df)} records")
    
    # Create a copy to avoid modifying original
    enriched_df = df.copy()
    
    # Find the LinkedIn column (case-insensitive search)
    linkedin_col = None
    for col in enriched_df.columns:
        if 'linkedin' in col.lower():
            linkedin_col = col
            break
    
    if not linkedin_col:
        log_to_debug("❌ No LinkedIn column found in CSV")
        raise ValueError("CSV must contain a LinkedIn URL column (e.g., 'LinkedIn_URL', 'LinkedIn Profile URL')")
    
    log_to_debug(f"✅ Found LinkedIn column: {linkedin_col}")
    
    # Add Email and Phone columns if they don't exist
    if 'Email' not in enriched_df.columns:
        enriched_df['Email'] = ''
    if 'Phone' not in enriched_df.columns:
        enriched_df['Phone'] = ''
    
    # Stats tracking
    total_processed = 0
    emails_found = 0
    phones_found = 0
    errors = 0
    
    # Process each row
    for idx, row in enriched_df.iterrows():
        linkedin_url = row[linkedin_col]
        
        # Skip if no LinkedIn URL
        if pd.isna(linkedin_url) or not linkedin_url or linkedin_url == "Profile not found":
            log_to_debug(f"  ⏭️ Row {idx+1}: No LinkedIn URL - skipping")
            continue
        
        total_processed += 1
        log_to_debug(f"  🔍 Row {idx+1}/{len(enriched_df)}: Processing {linkedin_url}")
        
        # Call Apollo API
        url = "https://api.apollo.io/v1/people/match"
        
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "X-Api-Key": apollo_api_key
        }
        
        payload = {
            "api_key": apollo_api_key,
            "linkedin_url": linkedin_url
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                person = data.get('person', {})
                
                # Extract email
                email = person.get('email', '')
                email_status = person.get('email_status', '')
                
                # Extract phone
                phone_numbers = person.get('phone_numbers', [])
                phone = phone_numbers[0].get('sanitized_number', '') if phone_numbers else ''
                
                if not phone:
                    phone = person.get('sanitized_phone', '') or person.get('phone', '')
                
                # Update dataframe
                if email:
                    enriched_df.at[idx, 'Email'] = email
                    emails_found += 1
                    status_icon = "✓" if email_status == 'verified' else "~"
                    log_to_debug(f"    {status_icon} Email: {email[:30]}... ({email_status})")
                
                if phone:
                    enriched_df.at[idx, 'Phone'] = phone
                    phones_found += 1
                    log_to_debug(f"    ✓ Phone: {phone}")
                
                if not email and not phone:
                    log_to_debug(f"    ✗ No contact info found")
                
            elif response.status_code == 404:
                log_to_debug(f"    ✗ Profile not found in Apollo database")
            
            elif response.status_code == 429:
                log_to_debug(f"    ⚠️ Rate limit hit, waiting 5s...")
                time.sleep(5)
                # Retry the same request
                continue
            
            elif response.status_code == 403:
                log_to_debug(f"    ❌ Apollo 403 Forbidden - Check API key")
                errors += 1
            
            elif response.status_code == 401:
                log_to_debug(f"    ❌ Apollo: Invalid API key")
                raise ValueError("Invalid Apollo API key - please check your credentials")
            
            else:
                log_to_debug(f"    ✗ Apollo API error: {response.status_code}")
                errors += 1
        
        except requests.exceptions.Timeout:
            log_to_debug(f"    ✗ Request timeout")
            errors += 1
        
        except requests.exceptions.RequestException as e:
            log_to_debug(f"    ✗ Request failed: {str(e)[:100]}")
            errors += 1
        
        except Exception as e:
            log_to_debug(f"    ✗ Unexpected error: {str(e)[:100]}")
            errors += 1
        
        # Rate limiting: sleep between requests
        time.sleep(0.5)
    
    # Final summary
    log_to_debug(f"\n📊 EXTRACTION SUMMARY:")
    log_to_debug(f"  • Total records: {len(enriched_df)}")
    log_to_debug(f"  • Processed: {total_processed}")
    log_to_debug(f"  • Emails found: {emails_found}")
    log_to_debug(f"  • Phones found: {phones_found}")
    log_to_debug(f"  • Errors: {errors}")
    log_to_debug(f"  • Success rate: {(emails_found/total_processed*100):.1f}%" if total_processed > 0 else "  • Success rate: 0%")
    
    log_to_debug(f"✅ Email extraction completed")
    return enriched_df

# ============ INITIALIZE SESSION STATE ============

if 'extracted_info' not in st.session_state:
    st.session_state.extracted_info = None
if 'output_logs' not in st.session_state:
    st.session_state.output_logs = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'csv_filename' not in st.session_state:
    st.session_state.csv_filename = None
if 'enriched_df' not in st.session_state:
    st.session_state.enriched_df = None
if "llm_config" not in st.session_state:
    st.session_state.llm_config = {
        "provider": "Mistral",
        "api_key": "",
        "api_key_2": "",
        "validated": False,
        "validated_2": False,
        "custom_endpoint": "",
        "custom_endpoint_2": ""
    }
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []
if "show_endpoint_editor" not in st.session_state:
    st.session_state.show_endpoint_editor = {}
if "temp_endpoints" not in st.session_state:
    st.session_state.temp_endpoints = {}
if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = None
if "gsheet_service" not in st.session_state:
    st.session_state.gsheet_service = None
if "oauth_credentials" not in st.session_state:
    st.session_state.oauth_credentials = None

# ============ SIDEBAR CONFIGURATION ============

st.sidebar.title("⚙️ LLM Configuration")

st.sidebar.info("💡 **Tip:** Provide 2 API keys for the same provider for better reliability and load distribution.")

config = st.session_state.llm_config

with st.sidebar.expander(f"🔧 {config['provider']} Configuration", expanded=True):
    previous_provider = config.get("provider", "Mistral")
    provider = st.selectbox(
        "Select LLM Provider",
        ["Claude", "OpenAI", "Gemini", "Mistral", "Deepseek", "Qwen", "Perplexity", "Llama"],
        key="provider_select",
        index=["Claude", "OpenAI", "Gemini", "Mistral", "Deepseek", "Qwen", "Perplexity", "Llama"].index(config["provider"])
    )
    
    if provider != previous_provider:
        st.session_state.llm_config["custom_endpoint"] = ""
        st.session_state.llm_config["custom_endpoint_2"] = ""
        st.session_state.llm_config["validated"] = False
        st.session_state.llm_config["validated_2"] = False
        st.session_state["show_edit_btn_key1"] = False
        st.session_state["show_edit_btn_key2"] = False
        st.session_state["validation_error_key1"] = ""
        st.session_state["validation_error_key2"] = ""
        st.session_state.show_endpoint_editor["key1"] = False
        st.session_state.show_endpoint_editor["key2"] = False
        if "key1" in st.session_state.temp_endpoints:
            del st.session_state.temp_endpoints["key1"]
        if "key2" in st.session_state.temp_endpoints:
            del st.session_state.temp_endpoints["key2"]
        log_to_debug(f"🔄 Provider changed from {previous_provider} to {provider} - endpoints reset to default")
    
    st.session_state.llm_config["provider"] = provider
    
    # First API Key
    api_key = st.text_input(
        f"{provider} API Key",
        value=config.get("api_key", ""),
        type="password",
        key="api_key_input"
    )
    st.session_state.llm_config["api_key"] = api_key
    
    if st.session_state.show_endpoint_editor.get("key1", False):
        if "key1" not in st.session_state.temp_endpoints:
            st.session_state.temp_endpoints["key1"] = get_active_endpoint(config, 1)
        
        st.caption(f"🔗 Current Endpoint:")
        edited_endpoint = st.text_input(
            "Edit Endpoint URL",
            value=st.session_state.temp_endpoints["key1"],
            key="endpoint_edit_key1",
            help="Modify this endpoint if needed. Changes will be saved when you click Save."
        )
        st.session_state.temp_endpoints["key1"] = edited_endpoint
        
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("💾 Save", key="save_endpoint_key1", use_container_width=True):
                st.session_state.llm_config["custom_endpoint"] = edited_endpoint
                log_to_debug(f"📝 Endpoint updated for {provider}: {edited_endpoint}")
                st.session_state.show_endpoint_editor["key1"] = False
                st.session_state["show_edit_btn_key1"] = False
                st.success("✅ Endpoint saved! Please validate again.")
                st.rerun()
        
        with col_cancel:
            if st.button("❌ Cancel", key="cancel_endpoint_key1", use_container_width=True):
                st.session_state.show_endpoint_editor["key1"] = False
                st.session_state["show_edit_btn_key1"] = False
                if "key1" in st.session_state.temp_endpoints:
                    del st.session_state.temp_endpoints["key1"]
                st.rerun()
    
    if api_key and not st.session_state.show_endpoint_editor.get("key1", False):
        show_edit_btn = st.session_state.get("show_edit_btn_key1", False)
        
        if show_edit_btn and st.session_state.get("validation_error_key1"):
            st.error(st.session_state["validation_error_key1"])
        
        if st.button(f"✓ Validate API Key", key="validate_key1", use_container_width=True):
            endpoint = get_active_endpoint(config, 1)
            is_valid, message = test_llm_connection(provider, api_key, endpoint)
            
            if is_valid:
                st.session_state.llm_config["validated"] = True
                st.session_state["show_edit_btn_key1"] = False
                st.session_state["validation_error_key1"] = ""
                log_to_debug(f"✅ {provider} API Key is Valid (Endpoint: {endpoint})")
                st.success(f"✅ API key is Valid")
                st.rerun()
            else:
                st.session_state.llm_config["validated"] = False
                st.session_state["show_edit_btn_key1"] = True
                st.session_state["validation_error_key1"] = f"❌ Authentication failed: {message}"
                log_to_debug(f"❌ {provider} API Key validation failed at {endpoint}: {message}")
                st.rerun()
        
        if show_edit_btn:
            if st.button("🔧 Edit Endpoint", key="edit_endpoint_btn_key1", use_container_width=True):
                st.session_state.show_endpoint_editor["key1"] = True
                st.rerun()
        
        if config.get("validated"):
            active_endpoint = get_active_endpoint(config, 1)
            st.success(f"✅ API key is Valid")
            st.caption(f"🔗 Using: `{active_endpoint}`")
    
    # Second API Key
    api_key_2 = st.text_input(
        f"{provider} API Key (Optional)",
        value=config.get("api_key_2", ""),
        type="password",
        key="api_key_2_input",
        help="Recommended: Add a second key for automatic failover and load distribution"
    )
    st.session_state.llm_config["api_key_2"] = api_key_2
    
    if st.session_state.show_endpoint_editor.get("key2", False):
        if "key2" not in st.session_state.temp_endpoints:
            st.session_state.temp_endpoints["key2"] = get_active_endpoint(config, 2)
        
        st.caption(f"🔗 Current Endpoint (Key 2):")
        edited_endpoint_2 = st.text_input(
            "Edit Endpoint URL",
            value=st.session_state.temp_endpoints["key2"],
            key="endpoint_edit_key2",
            help="Modify this endpoint if needed. Changes will be saved when you click Save."
        )
        st.session_state.temp_endpoints["key2"] = edited_endpoint_2
        
        col_save2, col_cancel2 = st.columns(2)
        with col_save2:
            if st.button("💾 Save", key="save_endpoint_key2", use_container_width=True):
                st.session_state.llm_config["custom_endpoint_2"] = edited_endpoint_2
                log_to_debug(f"📝 Endpoint updated for {provider} (Key 2): {edited_endpoint_2}")
                st.session_state.show_endpoint_editor["key2"] = False
                st.session_state["show_edit_btn_key2"] = False
                st.success("✅ Endpoint saved! Please validate again.")
                st.rerun()
        
        with col_cancel2:
            if st.button("❌ Cancel", key="cancel_endpoint_key2", use_container_width=True):
                st.session_state.show_endpoint_editor["key2"] = False
                st.session_state["show_edit_btn_key2"] = False
                if "key2" in st.session_state.temp_endpoints:
                    del st.session_state.temp_endpoints["key2"]
                st.rerun()
    
    if api_key_2 and not st.session_state.show_endpoint_editor.get("key2", False):
        show_edit_btn_2 = st.session_state.get("show_edit_btn_key2", False)
        
        if show_edit_btn_2 and st.session_state.get("validation_error_key2"):
            st.error(st.session_state["validation_error_key2"])
        
        if st.button(f"✓ Validate API Key", key="validate_key2", use_container_width=True):
            endpoint = get_active_endpoint(config, 2)
            is_valid, message = test_llm_connection(provider, api_key_2, endpoint)
            
            if is_valid:
                st.session_state.llm_config["validated_2"] = True
                st.session_state["show_edit_btn_key2"] = False
                st.session_state["validation_error_key2"] = ""
                log_to_debug(f"✅ {provider} second API Key validated (Endpoint: {endpoint})")
                st.success(f"✅ API key authenticated ")
                st.rerun()
            else:
                st.session_state.llm_config["validated_2"] = False
                st.session_state["show_edit_btn_key2"] = True
                st.session_state["validation_error_key2"] = f"❌ Authentication failed: {message}"
                log_to_debug(f"❌ {provider} second API Key validation failed at {endpoint}: {message}")
                st.rerun()
        
        if show_edit_btn_2:
            if st.button("🔧 Edit Endpoint", key="edit_endpoint_btn_key2", use_container_width=True):
                st.session_state.show_endpoint_editor["key2"] = True
                st.rerun()
        
        if config.get("validated_2"):
            active_endpoint = get_active_endpoint(config, 2)
            st.success(f"✅ API Key is Valid ")
            st.caption(f"🔗 Using: `{active_endpoint}`")

st.sidebar.divider()
st.sidebar.caption("Lead Generation Agent v1.0")

# ============ MAIN CONTENT ============

tab1, tab2 = st.tabs(["📊 Lead Generation", "🔗 LinkedIn Scraper"])

# ============ TAB 1: LEAD GENERATION ============
with tab1:
    st.title("📊 Lead Generation Agent")
    st.markdown("Select your preferences to generate a funding report!")

    st.markdown("### 🔍 Select Report Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        today = datetime.now()
        current_weekday = today.weekday()
        days_since_monday = current_weekday + 7
        last_monday = today - timedelta(days=days_since_monday)
        last_sunday = last_monday + timedelta(days=6)
        last_week = f"{last_monday.strftime('%d %B')} - {last_sunday.strftime('%d %B %Y')}"
        
        first_day_this_month = today.replace(day=1)
        last_month = (first_day_this_month - timedelta(days=1)).strftime("%B %Y")
        current_year = today.year
        
        timeframe = st.selectbox(
            "📅 Timeframe",
            options=[
                f" Weekly Report  ({last_week})",
                f" Monthly Report  ({last_month})",
               
            ],
            key="timeframe_select"
        )

    with col2:
        region = st.selectbox(
            "🌍 Region",
            options=["Global", "India"],
            help="Select the geographical region",
            key="region_select"
        )

    with col3:
        sector_options = get_sector_options(timeframe, region)
        sector = st.selectbox(
            "🏢 Sector",
            options=sector_options,
            help="Select the industry sector",
            key="sector_select"
        )

    auto_query = f"{sector} startups funding in {region} {timeframe}"

    st.markdown("---")
    st.markdown("### 📝 Generated Query")
    st.info(f"**Query:** {auto_query}")

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        extract_button = st.button("🔍 Generate Report", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)

    if clear_button:
        clear_history()
        st.rerun()

    if extract_button:
        config = st.session_state.llm_config
        has_valid_config = config.get("api_key") and config.get("validated")
        
        if not has_valid_config:
            st.error("❌ Please configure and validate at least one LLM API key in the sidebar before generating reports")
            log_to_debug("❌ Generate Report failed: No validated LLM configuration found")
            st.info("💡 Go to the sidebar 'LLM Configuration' section to add and validate your API key")
        else:
            with st.spinner("Processing your request... ( this may take 20-25 minutes ) "):
                extracted_info = {
                    'sector': sector,
                    'region': region,
                    'timeframe': timeframe
                }
                st.session_state.extracted_info = extracted_info

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    from main import start

                    status_text.text("🔍 Analyzing intent...")
                    progress_bar.progress(10)

                    import sys
                    from io import StringIO

                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()

                    status_text.text("🔍 Searching for companies and enriching data...")
                    progress_bar.progress(30)

                    primary_key = config.get("api_key", "")
                    secondary_key = config.get("api_key_2", "")
                    
                    selected_llm = {
                        "provider": config["provider"],
                        "api_key": primary_key,
                        "endpoint": get_active_endpoint(config, 1),
                        "validated": True
                    }
                    
                    st.session_state.selected_llm = selected_llm
                    
                    log_to_debug(f"🔍 Using LLM provider: {selected_llm.get('provider')}")
                    log_to_debug(f"🔑 Primary key: {primary_key[:10] if primary_key else 'EMPTY'}...")
                    log_to_debug(f"🔑 Secondary key: {secondary_key[:10] if secondary_key else 'None'}...")

                    csv_filename = start(
                        sector=sector,
                        region=region,
                        timeframe=timeframe,
                        mistral_key_1=primary_key,
                        mistral_key_2=secondary_key,
                        llm_config=selected_llm
                    )

                    sys.stdout = old_stdout
                    output = captured_output.getvalue()

                    st.session_state.output_logs = output
                    st.session_state.csv_filename = csv_filename

                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")

                    if csv_filename:
                        with st.expander("🔍 Debug Information", expanded=False):
                            st.write("CSV/Excel file returned:", csv_filename)
                            st.write("File exists:", os.path.exists(csv_filename) if csv_filename else False)

                            if csv_filename and os.path.exists(csv_filename):
                                try:
                                    if csv_filename.endswith('.xlsx') or csv_filename.endswith('.xls'):
                                        df = pd.read_excel(csv_filename)
                                        st.info(f"📊 Loaded Excel file: {csv_filename}")
                                    else:
                                        df = pd.read_csv(csv_filename)
                                        st.info(f"📊 Loaded CSV file: {csv_filename}")
                                    
                                    st.session_state.df = df
                                    st.session_state.csv_filename = csv_filename
                                    st.success(f"✅ Successfully loaded: {csv_filename} ({len(df)} rows)")
                                except Exception as e:
                                    st.error(f"❌ Error loading file: {str(e)}")
                                    import traceback
                                    with st.expander("🔍 Error Details"):
                                        st.code(traceback.format_exc())
                            elif csv_filename:
                                st.warning(f"⚠️ File not found: {csv_filename}")
                                with st.expander("🔍 Available files in directory"):
                                    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.xls'))]
                                    st.write("Files found:", files)
                            else:
                                st.warning("⚠️ No filename returned from processing")
                                all_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.xls'))]
                                if all_files:
                                    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                                    csv_filename = all_files[0]
                                    st.info(f"🔍 Found most recent file: {csv_filename}")
                                    try:
                                        if csv_filename.endswith('.xlsx') or csv_filename.endswith('.xls'):
                                            df = pd.read_excel(csv_filename)
                                        else:
                                            df = pd.read_csv(csv_filename)
                                        st.session_state.df = df
                                        st.session_state.csv_filename = csv_filename
                                        st.success(f"✅ Loaded file: {csv_filename} ({len(df)} rows)")
                                    except Exception as e:
                                        st.error(f"❌ Error loading file: {str(e)}")
                                else:
                                    st.error("❌ No CSV/Excel files found in directory")
                    else:
                        st.error("❌ No result returned from processing")

                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"❌ Error during processing: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())

    if st.session_state.extracted_info:
        st.success("✅ Information extracted successfully!")

        sector_val = st.session_state.extracted_info.get('sector', 'Not specified')
        region_val = st.session_state.extracted_info.get('region', 'Not specified')
        timeframe_val = st.session_state.extracted_info.get('timeframe', 'Not specified')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="📅 Timeframe",
                value=timeframe_val
            )

        with col2:
            st.metric(
                label="🌍 Region",
                value=region_val
            )

        with col3:
            st.metric(
                label="🏢 Sector",
                value=sector_val
            )

        st.markdown("---")

        if st.session_state.output_logs:
            log_expander = st.expander("📋 View Detailed Logs", expanded=False)
            with log_expander:
                st.code(st.session_state.output_logs, language="text")

        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### 📄 Generated Report")

            df = st.session_state.df

            st.success(f"✅ Report generated successfully! ({len(df)} companies)")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Companies", len(df))
            
            with col2:
                email_col = None
                for col in df.columns:
                    if col.lower() == 'email':
                        email_col = col
                        break
                email_count = df[email_col].notna().sum() if email_col else 0
                st.metric("With Email", email_count)
            
            with col3:
                linkedin_col = None
                for col in df.columns:
                    if 'linkedin' in col.lower():
                        linkedin_col = col
                        break
                linkedin_count = df[linkedin_col].notna().sum() if linkedin_col else 0
                st.metric("With LinkedIn", linkedin_count)

            st.markdown("#### 📊 Data Preview")
            st.dataframe(
                df,
                use_container_width=True,
                height=400
            )

            csv_data = df.to_csv(index=False).encode('utf-8')
            
            download_filename = st.session_state.csv_filename
            if download_filename and download_filename.endswith('.xlsx'):
                download_filename = download_filename.replace('.xlsx', '.csv')
            
            st.download_button(
                label="⬇️ Download Full Report (CSV)",
                data=csv_data,
                file_name=download_filename if download_filename else "funding_report.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv"
            )

            # ============ EXTRACT CONTACT SECTION ============
            st.markdown("---")
            st.markdown("### 📧 Extract Contact")
            st.caption("Enrich your data with contact information using Apollo.io API")
            
            # CSV Selection Options
            csv_selection = st.radio(
                "Select CSV file to enrich:",
                options=["Use Generated CSV", "Upload Local CSV"],
                horizontal=True,
                key="csv_selection_radio"
            )
            
            csv_to_enrich = None
            
            if csv_selection == "Use Generated CSV":
                if st.session_state.df is not None:
                    csv_to_enrich = st.session_state.df
                    st.info(f"📊 Using generated CSV with {len(csv_to_enrich)} rows")
                else:
                    st.warning("⚠️ No generated CSV available. Please generate a report first or upload a local CSV.")
            
            else:  # Upload Local CSV
                uploaded_csv = st.file_uploader(
                    "Upload CSV file",
                    type=["csv"],
                    key="upload_csv_for_extraction"
                )
                
                if uploaded_csv:
                    try:
                        csv_to_enrich = pd.read_csv(uploaded_csv)
                        st.info(f"📊 Uploaded CSV with {len(csv_to_enrich)} rows")
                        
                        # Show preview of uploaded CSV
                        with st.expander("📋 Preview Uploaded CSV", expanded=False):
                            st.dataframe(csv_to_enrich.head(10), use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error reading CSV file: {str(e)}")
                        csv_to_enrich = None
            
            # Apollo API Key Input
            apollo_api_key = st.text_input(
                "🔑 Apollo.io API Key",
                type="password",
                key="apollo_api_key_input",
                help="Enter your Apollo.io API key to enrich contact data"
            )
            
            # Extract Emails Button
            if st.button("🚀 Extract Emails", type="primary", use_container_width=True, key="extract_emails_btn"):
                if not apollo_api_key:
                    st.error("❌ Please provide Apollo.io API key")
                    log_to_debug("❌ Email extraction failed: No Apollo API key provided")
                
                elif csv_to_enrich is None:
                    st.error("❌ Please select or upload a CSV file")
                    log_to_debug("❌ Email extraction failed: No CSV file selected")
                
                else:
                    log_to_debug(f"🚀 Starting email extraction for {len(csv_to_enrich)} records")
                    
                    with st.spinner(f"Extracting contact information for {len(csv_to_enrich)} companies... This may take a few minutes"):
                        try:
                            # Call the extraction function
                            enriched_df = extract_with_apollo(csv_to_enrich, apollo_api_key)
                            
                            # Store enriched dataframe in session state
                            st.session_state.enriched_df = enriched_df
                            
                            log_to_debug(f"✅ Email extraction completed successfully")
                            st.success(f"✅ Contact extraction completed! Processed {len(enriched_df)} records")
                        
                        except Exception as e:
                            st.error(f"❌ Error during contact extraction: {str(e)}")
                            log_to_debug(f"❌ Email extraction error: {str(e)}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
            
            # Display Enriched Data Preview
            if st.session_state.enriched_df is not None:
                st.markdown("---")
                st.markdown("### 📊 Enriched Data Preview")
                
                enriched_df = st.session_state.enriched_df
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(enriched_df))
                
                with col2:
                    email_col = None
                    for col in enriched_df.columns:
                        if col.lower() == 'email':
                            email_col = col
                            break
                    email_count = enriched_df[email_col].notna().sum() if email_col else 0
                    st.metric("Emails Found", email_count)
                
                with col3:
                    enrichment_rate = (email_count / len(enriched_df) * 100) if len(enriched_df) > 0 else 0
                    st.metric("Enrichment Rate", f"{enrichment_rate:.1f}%")
                
                # Show dataframe
                st.dataframe(
                    enriched_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download button for enriched data
                enriched_csv_data = enriched_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="⬇️ Download Enriched Report (CSV)",
                    data=enriched_csv_data,
                    file_name="enriched_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_enriched_csv"
                )

            # ============ GOOGLE SHEETS INTEGRATION ============
            st.markdown("---")
            st.markdown("### ☁️ Google Sheets Integration")
            st.caption("Upload your generated report to Google Sheets directly!")
            
            oauth_file = st.file_uploader(
                "📁 Upload Google OAuth Credentials (JSON)",
                type=["json"],
                help="Upload your OAuth 2.0 Client ID JSON file from Google Cloud Console",
                key="oauth_uploader_main"
            )
            
            if oauth_file:
                try:
                    oauth_data = json.load(oauth_file)
                    log_to_debug(f"OAuth JSON file uploaded: {oauth_file.name}")
                    
                    if "installed" in oauth_data or "web" in oauth_data:
                        st.session_state.oauth_credentials = oauth_data
                        log_to_debug("✅ Valid OAuth credentials structure detected")
                        st.success("✅ Valid OAuth credentials file detected")
                        
                        # Authenticate button
                        if st.button("🔐 Authenticate with Google", type="primary", use_container_width=True, key="auth_google_main"):
                            log_to_debug("Attempting Google OAuth authentication...")
                            
                            try:
                                with st.spinner("Authenticating with Google..."):
                                    service = authenticate_google_sheets(oauth_data)
                                    st.session_state.gsheet_service = service
                                    st.session_state.gsheet_authenticated = True
                                    log_to_debug("✅ Google OAuth authentication successful")
                                    st.success("✅ Successfully authenticated with Google Sheets!")
                                    st.rerun()
                            except Exception as e:
                                log_to_debug(f"❌ OAuth authentication failed: {str(e)}")
                                st.error(f"❌ Authentication failed: {str(e)}")
                    else:
                        log_to_debug("❌ Invalid OAuth JSON structure")
                        st.error("❌ Invalid OAuth credentials file. Please upload a valid JSON from Google Cloud Console.")
                
                except json.JSONDecodeError:
                    log_to_debug("❌ Failed to parse OAuth JSON file")
                    st.error("❌ Invalid JSON file. Please upload a valid OAuth credentials file.")
                except Exception as e:
                    log_to_debug(f"❌ Error reading OAuth file: {str(e)}")
                    st.error(f"❌ Error reading file: {str(e)}")
            
            # Show upload section if authenticated
            if st.session_state.get("gsheet_authenticated", False) and st.session_state.get("gsheet_service"):
                st.success("✅ Authenticated with Google Sheets")
                
                st.divider()
                
                # Choose CSV option
                upload_option = st.radio(
                    "Choose CSV to upload:",
                    ["Use Generated CSV", "Use Enriched CSV", "Upload Local CSV"],
                    horizontal=True,
                    key="upload_option_radio"
                )
                
                df_to_upload = None
                sheet_name = None
                
                if upload_option == "Use Generated CSV":
                    if st.session_state.df is not None:
                        df_to_upload = st.session_state.df
                        sheet_name = f"Lead Report {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        st.info(f"📊 Using generated CSV with {len(df_to_upload)} rows")
                    else:
                        st.warning("⚠️ No generated CSV available. Please generate a report first.")
                
                elif upload_option == "Use Enriched CSV":
                    if st.session_state.enriched_df is not None:
                        df_to_upload = st.session_state.enriched_df
                        sheet_name = f"Enriched Report {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        st.info(f"📊 Using enriched CSV with {len(df_to_upload)} rows")
                    else:
                        st.warning("⚠️ No enriched CSV available. Please extract emails first.")
                
                else:  # Upload Local CSV
                    uploaded_file_local = st.file_uploader(
                        "Choose a local CSV file",
                        type="csv",
                        key="local_csv_upload_gsheet"
                    )
                    
                    if uploaded_file_local:
                        try:
                            df_to_upload = pd.read_csv(uploaded_file_local)
                            sheet_name = f"{uploaded_file_local.name.replace('.csv', '')} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                            st.info(f"📊 Using uploaded CSV with {len(df_to_upload)} rows")
                            with st.expander("📋 Preview"):
                                st.dataframe(df_to_upload.head(10), use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ Error reading CSV: {str(e)}")
                
                # Upload button
                if df_to_upload is not None:
                    custom_sheet_name = st.text_input(
                        "Customize Sheet Name (optional):",
                        value=sheet_name,
                        key="custom_sheet_name"
                    )
                    
                    if st.button("📤 Upload to Google Sheets", type="primary", use_container_width=True, key="upload_to_gsheet"):
                        log_to_debug(f"Starting upload to Google Sheets: {custom_sheet_name}")
                        
                        try:
                            with st.spinner("Uploading to Google Sheets..."):
                                service = st.session_state.gsheet_service
                                spreadsheet_url = upload_to_google_sheets(service, df_to_upload, custom_sheet_name)
                                
                                log_to_debug(f"✅ Successfully uploaded to Google Sheets: {spreadsheet_url}")
                                st.success(f"✅ Successfully uploaded to Google Sheets!")
                                st.markdown(f"🔗 **[Open Spreadsheet]({spreadsheet_url})**")
                        
                        except Exception as e:
                            log_to_debug(f"❌ Upload to Google Sheets failed: {str(e)}")
                            st.error(f"❌ Upload failed: {str(e)}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())

# ============ TAB 2: LINKEDIN SCRAPER ============
with tab2:
    st.title("🔗 LinkedIn Scraper")
    st.markdown("Configure and scrape LinkedIn profiles based on your criteria")
    
    st.markdown("### 🎯 Scraper Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        industry = st.selectbox(
            "🏭 Industry",
            options=[
                "Technology",
                "Finance",
                "Healthcare",
                "Education",
                "Manufacturing",
                "Retail",
                "Consulting",
                "Real Estate",
                "Marketing",
                "Legal"
            ],
            key="linkedin_industry"
        )
    
    with col2:
        job_title = st.selectbox(
            "💼 Job Title",
            options=[
                "CEO",
                "CTO",
                "CFO",
                "VP Engineering",
                "VP Sales",
                "VP Marketing",
                "Director",
                "Manager",
                "Engineer",
                "Consultant"
            ],
            key="linkedin_job_title"
        )
    
    with col3:
        location = st.selectbox(
            "📍 Location",
            options=[
                "United States",
                "India",
                "United Kingdom",
                "Canada",
                "Germany",
                "Singapore",
                "Australia",
                "France",
                "Netherlands",
                "Global"
            ],
            key="linkedin_location"
        )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        fetch_button = st.button("🚀 Start Fetch", type="primary", use_container_width=True, key="linkedin_fetch")
    with col2:
        clear_linkedin_button = st.button("🗑️ Clear History", use_container_width=True, key="linkedin_clear")
    
    if clear_linkedin_button:
        st.success("✅ LinkedIn scraper history cleared!")
        log_to_debug("LinkedIn scraper history cleared")
    
    if fetch_button:
        st.info(f"🔍 Fetching LinkedIn profiles for **{job_title}** in **{industry}** industry located in **{location}**...")
        log_to_debug(f"LinkedIn fetch initiated: {job_title} | {industry} | {location}")
        
        with st.spinner("Scraping LinkedIn profiles... This may take a few moments"):
            import time
            time.sleep(2)
            st.success("✅ Fetch completed! (This is a placeholder - implement actual scraping logic)")
            log_to_debug("LinkedIn fetch completed (simulated)")

# Footer
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🚀 Lead Generation Agent | Powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
