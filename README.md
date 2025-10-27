# 🚀 Lead Generation Agent

📊 Lead Generation Agent

> An intelligent AI-powered lead generation platform that automates the collection, enrichment, and analysis of startup funding data from multiple sources.

---

## 📸 Screenshots


### Dashboard 
<img width="959" height="419" alt="LGA1" src="https://github.com/user-attachments/assets/5b071eba-f2b2-4aa5-8699-06b0d4f18951" />

*Main dashboard showing report generation interface*

### LLM Configuration
<img width="671" height="421" alt="LGA5" src="https://github.com/user-attachments/assets/fcb710e9-50c8-4ac3-83cc-8f308f054624" />

*Multi-LLM provider configuration panel*

### Report Preview
<img width="956" height="418" alt="LGA2" src="https://github.com/user-attachments/assets/89c18fb5-318e-43f5-8853-da11a2237797" />

*Generated report with enriched company data*

### Contact Extraction Via Apollo Integration 
<img width="957" height="423" alt="LGA3" src="https://github.com/user-attachments/assets/f4eecc10-d85b-4e28-88c4-de395e6f7195" />

* Email Extraction Via Apollo API Integration*

### Google Sheets Integration 
<img width="950" height="413" alt="LGA4" src="https://github.com/user-attachments/assets/d547d31e-a802-47f3-8cf4-7596fdd48ea2" />

* Integrating Google Sheets via Oauth *
---

## 📖 Description

The **Lead Generation Agent** is an enterprise-grade automation platform designed to streamline the discovery, enrichment, and qualification of startup funding data across global markets. By combining web scraping, multi-LLM intelligence, and contact enrichment APIs, this tool eliminates the manual overhead associated with market research and lead generation for investors, accelerators, and B2B marketing teams.

The platform intelligently adapts to user requirements by analyzing intent through AI—automatically determining whether weekly, monthly, or yearly data collection is needed based on natural language inputs. It then orchestrates parallel data extraction from curated sources like FoundersDay.co and GrowthList, while enriching records with LinkedIn profiles and contact information via Apollo.io integration.

Built with a **modular architecture** supporting 8 LLM providers (Mistral, Claude, OpenAI, Gemini, Deepseek, Qwen, Perplexity, Llama), the system offers flexibility, reliability through dual API key rotation, and resilience against rate limits. The intuitive Streamlit interface provides real-time monitoring, debug logging, and seamless export to CSV, Excel, or Google Sheets—making it accessible to both technical and non-technical users.

Key differentiators include **intelligent validation** that filters low-quality data (e.g., invalid founder names, duplicates), **anti-detection mechanisms** using undetected-chromedriver for sustained scraping operations, and **sector-specific optimization** across 12+ industries including AI, Fintech, Healthcare, and PropTech.

---

## ✨ Features

### 🤖 Multi-LLM Support
- Support for **8 different LLM providers**: Mistral, Claude, OpenAI, Gemini, Deepseek, Qwen, Perplexity, and Llama
- **Dual API key rotation** for reliability and load distribution
- **Custom endpoint configuration** with validation and testing
- Automatic failover between API keys

### 📊 Automated Data Collection
- **Weekly Reports**: Scrapes FoundersDay.co for the latest funding announcements
- **Monthly Reports**: Aggregates data from FoundersDay and GrowthList with intelligent pagination
- **Sector-Specific Filtering**: Supports 12+ industry sectors (AI, Fintech, Healthcare, EdTech, etc.)
- **Geographic Targeting**: India and Global coverage with region-specific optimizations

### 🔍 Intelligent Enrichment
- **LinkedIn Profile Discovery**: Automated founder/CEO LinkedIn profile search using Selenium
- **Contact Information Extraction**: Apollo.io API integration for email and phone numbers
- **Company Intelligence**: Extracts industry, category, description, and key personnel
- **Duplicate Detection**: Smart deduplication across scraping sessions

### 🎨 User-Friendly Interface
- **Interactive Streamlit Dashboard**: Clean, responsive UI with real-time progress tracking
- **Google Sheets Integration**: Direct export to Google Sheets with OAuth authentication
- **CSV/Excel Support**: Download enriched data in multiple formats
- **Debug Console**: Comprehensive logging for troubleshooting

---

## 🏗️ Architecture
Lead-Generation-Agent/
├── app.py # Streamlit UI and main application logic
├── main.py # Core orchestration and LLM routing
├── weekly.py # Weekly report scraper (FoundersDay)
├── monthly.py # Monthly report scraper (FoundersDay + GrowthList)
├── selenium_extractor.py # LinkedIn and company data enrichment
├── requirements.txt # Python dependencies
└── README.md


### 🔄 Data Flow

1. **User Input** → Streamlit interface (sector, region, timeframe)
2. **Intent Analysis** → LLM determines report type (weekly/monthly/yearly)
3. **Data Collection** → Selenium scrapes funding sources
4. **LLM Extraction** → Structured data extraction from raw HTML
5. **Enrichment** → LinkedIn search + Apollo.io contact extraction
6. **Export** → CSV/Excel/Google Sheets with enriched data

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Google Chrome browser (for Selenium)
- API keys for at least one LLM provider
- Apollo.io API key (optional, for contact enrichment)

### Setup

1. **Clone the repository**
2. **Install dependencies**
3.  **Configure API keys** (via Streamlit UI)
   - Add your LLM API key in the sidebar
   - Optionally add Apollo.io API key for contact extraction
4. **Run the application**


---

## 🎯 Usage

### Generating a Report

1. **📅 Select Report Parameters**
   - **Timeframe**: Choose between Weekly or Monthly reports
   - **Region**: India or Global
   - **Sector**: Select from 12+ industry categories

2. **🤖 Configure LLM Provider**
   - Choose your preferred LLM provider from the sidebar
   - Add and validate API keys
   - Optionally configure custom endpoints

3. **🚀 Generate Report**
   - Click "Generate Report" to start the automated pipeline
   - Processing takes 20-25 minutes for comprehensive reports
   - Monitor progress in real-time

4. **📧 Enrich Contact Data** (Optional)
   - Upload your Apollo.io API key
   - Select generated CSV or upload local file
   - Extract emails and phone numbers automatically

### LinkedIn Scraper

Use the LinkedIn Scraper tab for custom CSV enrichment:
- Upload CSV with company names
- Automated LinkedIn profile discovery
- Founder/CEO information extraction
- Integration with Apollo.io for contact details

---

## ⚙️ Configuration

### Supported LLM Providers

| Provider | Model | Authentication |
|----------|-------|----------------|
| 🟣 Mistral | mistral-small-latest | Bearer Token |
| 🟪 Claude | claude-3-5-sonnet-20241022 | x-api-key |
| 🟢 OpenAI | gpt-4o | Bearer Token |
| 🔵 Gemini | gemini-2.0-flash | Bearer Token |
| 🟡 Deepseek | deepseek-chat | Bearer Token |
| 🟠 Qwen | qwen-plus | Bearer Token |
| 🔷 Perplexity | sonar | Bearer Token |
| 🦙 Llama | llama-3.3-70b | Bearer Token |

---

## 🔑 Key Technologies

- **Streamlit**: Web application framework
- **Selenium**: Browser automation for web scraping
- **Undetected ChromeDriver**: Anti-detection for reliable scraping
- **BeautifulSoup**: HTML parsing
- **Pandas**: Data manipulation and CSV handling
- **Google Sheets API**: Direct spreadsheet integration
- **Apollo.io API**: Contact information enrichment
- **Multi-LLM Integration**: Flexible AI provider support

---

## 🎨 Advanced Features

### 🔄 API Key Rotation

The system automatically rotates between two API keys to:
- Bypass rate limits
- Ensure high availability
- Distribute load across multiple keys

### ✅ Intelligent Validation

- **Founder Name Validation**: Filters out invalid entries (e.g., company names, generic terms)
- **Duplicate Detection**: Prevents processing the same company multiple times
- **Data Quality Checks**: Validates emails, LinkedIn URLs, and funding amounts

### 📊 Google Sheets Integration

1. Upload OAuth credentials JSON
2. Authenticate via browser
3. Automatically create formatted spreadsheets
4. Color-coded headers and auto-resized columns

---

## 🐛 Troubleshooting

### ChromeDriver Issues
The system automatically downloads the correct ChromeDriver version for your Chrome browser. If issues persist:

### LLM API Errors
- ✅ Check API key validity in the sidebar
- 🔄 Try rotating to secondary key
- 🔧 Verify custom endpoint configuration
- 📋 Check debug logs for detailed error messages

### LinkedIn Scraping Blocked
- 🛡️ The system uses undetected-chromedriver to avoid detection
- ⏱️ Add random delays between requests
- 🌐 Use residential proxies if needed

---

## 📈 Performance

- **Weekly Reports**: ~10-15 minutes (2 pages from FoundersDay)
- **Monthly Reports**: ~20-25 minutes (13 pages from FoundersDay + GrowthList)
- **Enrichment**: ~0.5 seconds per company for Apollo.io API
- **LinkedIn Search**: ~5-10 seconds per company

---

## ⚠️ Limitations

- **Rate Limits**: Depends on LLM provider and Apollo.io subscription
- **Geographic Coverage**: Optimized for India and Global; limited granularity for other regions
- **Data Sources**: Relies on FoundersDay.co and GrowthList availability
- **LinkedIn Access**: Requires active internet and may face CAPTCHA challenges

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

---

## 🙏 Acknowledgments

- 🌐 FoundersDay.co for funding data
- 📊 GrowthList for startup information
- 📧 Apollo.io for contact enrichment
- 🎨 Streamlit for the UI framework


---

## 🎓 Conclusion

The **Lead Generation Agent** represents a paradigm shift from manual research workflows to **AI-powered automation** that scales effortlessly while maintaining data quality. By automating repetitive tasks like web scraping, data extraction, and contact enrichment, the platform achieves what would take teams days or weeks in under 30 minutes—delivering enriched, actionable intelligence ready for immediate outreach.

This project demonstrates the practical application of **multi-modal AI orchestration**—where LLM reasoning, browser automation, and API integrations work in concert to solve complex business problems. The dual API key rotation strategy ensures 99%+ uptime, while the modular design allows easy expansion to new data sources or LLM providers as the ecosystem evolves.

### 🎯 Use Cases

For **investors and venture capitalists**, this tool provides real-time visibility into funding trends and emerging startups. For **B2B marketers and sales teams**, it offers a pipeline of qualified leads with verified contact information, reducing customer acquisition costs by up to 70% through precision targeting. For **developers and data scientists**, it serves as a reference implementation for building production-grade AI agents with robust error handling, logging, and scalability.

As AI continues to transform lead generation—with 87% of organizations recognizing its competitive advantage—this platform positions users at the forefront of **intelligent automation**. Future enhancements could include predictive lead scoring, sentiment analysis from news articles, and integration with CRM systems like Salesforce or HubSpot for closed-loop attribution.

Whether you're tracking competitive intelligence, building investor databases, or generating B2B leads, the **Lead Generation Agent** provides the infrastructure to do it at scale—**smarter, faster, and more efficiently** than ever before.

---
## Author 

- ** Tushar Yadav **
- **Email**: tusharyadav61900@gmail.conm
- **LinkedIn**: [Tushar Yadav](https://www.linkedin.com/in/tushar-yadav-5829bb353/)





<div align="center">

**⭐ Ready to transform your lead generation strategy?**

**Star this repository and contribute to the future of AI-powered market intelligence!**

Made with ❤️ for the startup ecosystem

</div>




