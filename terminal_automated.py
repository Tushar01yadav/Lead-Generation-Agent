import subprocess
import requests
import re
import time
import threading
import sqlite3
from datetime import datetime
import base64
import logging
from flask import Flask, request, Response
from flask_cors import CORS

# ---------------------------
# Configuration
# ---------------------------
MISTRAL_API_KEY = "0tUk4hmSZWchWcS8Fkl2coFsAPh4KDC6"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
FLASK_HOST = '127.0.0.1'  # Changed from 0.0.0.0 to localhost
FLASK_PORT = 5001

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1x1 transparent pixel (base64 encoded GIF)
TRACKING_PIXEL = base64.b64decode(
    'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
)

@app.route('/track/<lead_id>')
def track_email_open(lead_id):
    """Track email open via 1x1 pixel"""
    try:
        user_agent = request.headers.get('User-Agent', 'Unknown')
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        referer = request.headers.get('Referer', 'None')
        
        print(f"\n{'='*70}")
        print(f"📧 EMAIL OPENED!")
        print(f"{'='*70}")
        print(f"  Lead ID: {lead_id}")
        print(f"  IP Address: {ip_address}")
        print(f"  Referer: {referer}")
        print(f"  User Agent: {user_agent[:80] if len(user_agent) > 80 else user_agent}")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        logger.info(f"Tracking pixel requested for lead_id: {lead_id} from IP: {ip_address}")
        
        conn = sqlite3.connect("leads.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, email, opened_at, founder_name, company_name 
            FROM leads 
            WHERE id = ?
        """, (lead_id,))
        result = cursor.fetchone()
        
        if result is not None:
            lead_db_id, email, opened_at, founder_name, company_name = result
            
            if opened_at is None:
                current_time = datetime.now().isoformat()
                cursor.execute("""
                    UPDATE leads 
                    SET opened_at = ? 
                    WHERE id = ?
                """, (current_time, lead_id))
                
                conn.commit()
                print(f"✅ Database updated: Lead {lead_id} ({email}) marked as OPENED")
                print(f"   Founder: {founder_name}")
                print(f"   Company: {company_name}")
                print(f"   Timestamp: {current_time}")
                logger.info(f"Lead {lead_id} ({email}) marked as opened in database")
            else:
                print(f"ℹ️  Lead {lead_id} ({email}) was already opened at: {opened_at}")
                logger.info(f"Duplicate open for lead {lead_id}")
        else:
            print(f"⚠️  WARNING: Lead {lead_id} not found in database!")
            logger.warning(f"Lead {lead_id} not found in database")
        
        conn.close()
    
    except sqlite3.Error as e:
        print(f"❌ DATABASE ERROR: {str(e)}")
        logger.error(f"Database error tracking email: {str(e)}")
    except Exception as e:
        print(f"❌ ERROR tracking email open: {str(e)}")
        logger.error(f"Error tracking email: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    response = Response(TRACKING_PIXEL, mimetype='image/gif')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    
    return response

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "service": "email-tracking",
        "timestamp": datetime.now().isoformat()
    }

@app.route('/stats')
def get_stats():
    """Get tracking statistics"""
    try:
        conn = sqlite3.connect("leads.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM leads")
        total_sent = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM leads WHERE opened_at IS NOT NULL")
        total_opened = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM leads WHERE replied = 1")
        total_replied = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT id, email, founder_name, company_name, opened_at 
            FROM leads 
            WHERE opened_at IS NOT NULL 
            ORDER BY opened_at DESC 
            LIMIT 10
        """)
        recent_opens = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_sent": total_sent,
            "total_opened": total_opened,
            "total_replied": total_replied,
            "open_rate": f"{(total_opened/total_sent*100):.1f}%" if total_sent > 0 else "0%",
            "reply_rate": f"{(total_replied/total_sent*100):.1f}%" if total_sent > 0 else "0%",
            "recent_opens": [{
                "id": id, 
                "email": email, 
                "founder_name": founder_name,
                "company_name": company_name,
                "opened_at": opened_at
            } for id, email, founder_name, company_name, opened_at in recent_opens]
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"error": str(e)}, 500

@app.route('/debug/leads')
def debug_leads():
    """Debug endpoint to see all leads"""
    try:
        conn = sqlite3.connect("leads.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, email, founder_name, company_name, sent_at, opened_at, replied
            FROM leads 
            ORDER BY sent_at DESC 
            LIMIT 20
        """)
        leads = cursor.fetchall()
        
        conn.close()
        
        return {
            "leads": [{
                "id": id,
                "email": email,
                "founder_name": founder_name,
                "company_name": company_name,
                "sent_at": sent_at,
                "opened_at": opened_at,
                "replied": bool(replied)
            } for id, email, founder_name, company_name, sent_at, opened_at, replied in leads]
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return {"error": str(e)}, 500

# ---------------------------
# Cloudflared functions
# ---------------------------
def extract_tunnel_url_mistral(text):
    """Ask Mistral only if regex fails."""
    prompt = f"""
Extract only the https URL of the Cloudflare quick tunnel from this text.

Output format: only the URL, nothing else.

Text:
{text}
"""

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        resp = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=20)
        result = resp.json()
        content = result["choices"][0]["message"]["content"].strip()
        if re.match(r"^https://[A-Za-z0-9\-\.]+\.trycloudflare\.com", content):
            return content
        else:
            return None
    except Exception as e:
        print("Mistral error:", e)
        return None


def run_cloudflared(local_url, timeout=40):
    """
    Launch cloudflared tunnel and attempt to capture the public URL.
    Returns the URL string if found, otherwise None.
    """
    cmd = ["cloudflared", "tunnel", "--url", local_url]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        print("❌ cloudflared not found. Make sure it's installed and in PATH.")
        return None

    lines = []
    start = time.time()
    print(f"⏳ Waiting for cloudflared to establish tunnel to {local_url}...\n")

    url_pattern = re.compile(r"https://[A-Za-z0-9\-.]+\.trycloudflare\.com(?:/[^\s]*)?")

    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
                lines.append(line)
                m = url_pattern.search(line)
                if m:
                    found = m.group(0)
                    print(f"\n✅ Found tunnel URL (live): {found}")
                    # Don't terminate - let it keep running
                    return found, process
            if time.time() - start > timeout:
                print("⚠️ Timeout — no tunnel URL seen in cloudflared output.")
                break
    except Exception as e:
        print(f"Error reading cloudflared output: {e}")

    text = "\n".join(lines)
    print("\n🤖 Trying Mistral for extraction from captured output...")
    url = extract_tunnel_url_mistral(text)
    if url:
        print(f"\n✅ Tunnel URL (Mistral): {url}")
        return url, process
    else:
        print("\n❌ Could not find tunnel URL. Dumping captured output:")
        print(text[:2000])
    return None, process


def start_flask_in_thread(host=FLASK_HOST, port=FLASK_PORT):
    """Start Flask in background thread"""
    def run_app():
        app.run(host=host, port=port, debug=False, use_reloader=False)
    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    return thread


def wait_for_server_ready(url=f'http://{FLASK_HOST}:{FLASK_PORT}/health', timeout=20, retries=40):
    """Wait for Flask server to be ready with multiple retries"""
    print(f"Waiting for server at {url} to be ready...")
    for i in range(retries):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"✅ Server responded successfully after {i+1} attempts")
                return True
        except requests.exceptions.ConnectionError:
            pass  # Expected while server is starting
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
        time.sleep(0.5)
    return False
def get_tunnel_url():
    """
    Start Flask server and cloudflared tunnel, return the public URL
    Returns: (tunnel_url: str, flask_thread: Thread, cloudflare_process: Popen)
    """
    print("\n" + "="*70)
    print("🚀 EMAIL TRACKING SERVER - STARTING")
    print("="*70 + "\n")

    # Start Flask in background
    print("Step 1: Starting Flask server...")
    flask_thread = start_flask_in_thread()
    
    # Wait for Flask to be ready
    print("Step 2: Waiting for Flask to be ready...")
    if not wait_for_server_ready():
        raise Exception("❌ Flask server did not start within timeout")
    
    print("✅ Flask server is ready")
    time.sleep(1)
    
    # Start cloudflared and get URL
    print("\nStep 3: Launching cloudflared tunnel...")
    local_url = f"http://{FLASK_HOST}:{FLASK_PORT}"
    result = run_cloudflared(local_url, timeout=40)
    
    if result and result[0]:
        tunnel_url, cloudflare_process = result
        print(f"\n{'='*70}")
        print(f"✅ TUNNEL ACTIVE")
        print(f"{'='*70}")
        print(f"🌐 Public URL: {tunnel_url}")
        print(f"📍 Local URL:  {local_url}")
        print(f"{'='*70}\n")
        
        return tunnel_url, flask_thread, cloudflare_process
    else:
        raise Exception("❌ Failed to establish cloudflared tunnel")

# ---------------------------
# Main entry
# ---------------------------
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 EMAIL TRACKING SERVER - ENHANCED (with cloudflared)")
    print("="*70)
    print(f"📍 Local: http://{FLASK_HOST}:{FLASK_PORT}")
    print("="*70 + "\n")

    # Start Flask in background
    print("Step 1: Starting Flask server...")
    start_flask_in_thread()
    
    # Wait for Flask to be ready with better timeout
    print("Step 2: Waiting for Flask to be ready...")
    if not wait_for_server_ready():
        print("❌ Server did not start within timeout.")
        print("   Try running Flask manually first to debug:")
        print(f"   python -c 'from {__name__} import app; app.run(port={FLASK_PORT})'")
        exit(1)
    
    # Verify server is actually responding
    print("\nStep 3: Verifying server health...")
    try:
        test_response = requests.get(f'http://{FLASK_HOST}:{FLASK_PORT}/health', timeout=5)
        print(f"✅ Health check response: {test_response.json()}")
    except Exception as e:
        print(f"⚠️ Warning: Could not verify health endpoint: {e}")
    
    # Give it one more second to stabilize
    time.sleep(1)
    
    # Now start cloudflared
    print("\nStep 4: Launching cloudflared tunnel...")
    local_url = f"http://{FLASK_HOST}:{FLASK_PORT}"
    result = run_cloudflared(local_url, timeout=40)
    
    if result and result[0]:
        tunnel_url, process = result
        print(f"\n{'='*70}")
        print(f"✅ TUNNEL ACTIVE")
        print(f"{'='*70}")
        print(f"🌐 Public URL: {tunnel_url}")
        print(f"📍 Local URL:  {local_url}")
        print(f"{'='*70}")
        print("\n📝 Update TRACKING_SERVER_URL in your email sender:")
        print(f'   TRACKING_SERVER_URL = "{tunnel_url}"')
        print("\n⚠️  Press Ctrl+C to stop both server and tunnel")
        print("="*70 + "\n")
        
        # Keep both running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            if process and process.poll() is None:
                process.terminate()
            print("✅ Cleanup complete")
    else:
        print("\n❌ Failed to establish tunnel")
        print("   Troubleshooting steps:")
        print("   1. Ensure cloudflared is installed: cloudflared --version")
        print("   2. Test manual tunnel: cloudflared tunnel --url http://localhost:5001")
        print("   3. Check firewall settings")
        print("   4. Verify network connectivity")