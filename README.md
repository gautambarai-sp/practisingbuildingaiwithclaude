# 📦 AI Inventory Optimization System

An intelligent inventory management system powered by Machine Learning that helps e-commerce businesses optimize stock levels, prevent stockouts, and identify slow-moving inventory.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### 📈 Demand Forecasting
- **ML-Powered Predictions**: Uses Facebook Prophet for accurate time-series forecasting
- **Multi-Horizon Forecasts**: 7, 14, and 30-day demand predictions
- **Confidence Intervals**: Statistical confidence bounds for all predictions
- **Seasonality Detection**: Automatically identifies weekly, monthly, and yearly patterns

### 🚨 Smart Alerts
- **Stockout Prevention**: Early warnings for items at risk
- **Priority-Based Recommendations**: Critical, urgent, and warning levels
- **Lead Time Awareness**: Considers supplier lead times in calculations

### 💰 Inventory Optimization
- **Reorder Point Calculation**: Automatic safety stock and reorder point computation
- **Cash Flow Forecasting**: Project inventory purchasing requirements
- **Slow-Mover Detection**: Identify overstock items requiring action

### 📊 Interactive Dashboard
- **Real-Time Metrics**: Live inventory health monitoring
- **Visual Analytics**: Interactive charts and graphs
- **Category Performance**: Track sales by product category
- **Export Capabilities**: Download reports and recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- GitHub account
- Streamlit Cloud account (free)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-inventory-optimizer.git
cd ai-inventory-optimizer
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
The app will automatically open at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Create a new GitHub repository**
   - Go to [GitHub](https://github.com) and create a new repository
   - Name it `ai-inventory-optimizer`
   - Keep it public (required for free Streamlit Cloud)

2. **Push your code to GitHub**
```bash
git init
git add .
git commit -m "Initial commit: AI Inventory Optimization System"
git branch -M main
git remote add origin https://github.com/yourusername/ai-inventory-optimizer.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure deployment:**
   - **Repository:** `yourusername/ai-inventory-optimizer`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. **Click "Deploy"**

Your app will be live at: `https://yourusername-ai-inventory-optimizer.streamlit.app`

### Step 3: Configure Secrets (Optional)

If you want to add API keys or database credentials:

1. In Streamlit Cloud, go to your app settings
2. Click "Secrets" in the sidebar
3. Add your secrets in TOML format:

```toml
[database]
host = "your-database-host"
username = "your-username"
password = "your-password"

[api_keys]
shopify_key = "your-shopify-api-key"
```

Access in code:
```python
import streamlit as st
db_host = st.secrets["database"]["host"]
```

## 📁 Project Structure

```
ai-inventory-optimizer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── config.toml            # Streamlit configuration (optional)
├── data/                  # Sample data directory
│   ├── sample_sales.csv
│   └── sample_inventory.csv
├── models/                # ML model utilities (optional)
│   └── forecasting.py
└── utils/                 # Helper functions (optional)
    └── data_processing.py
```

## 📊 Data Format

### Sales Data CSV Format

Your sales data should be in CSV format with these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| date | Date (YYYY-MM-DD) | Transaction date | 2024-01-15 |
| sku | String | Product SKU/ID | TRD-BLK-M |
| product_name | String | Product name | Black Dress Medium |
| category | String | Product category | Dresses |
| quantity_sold | Integer | Units sold | 5 |
| revenue | Float | Total revenue | 399.95 |
| price | Float | Unit price | 79.99 |

**Example CSV:**
```csv
date,sku,product_name,category,quantity_sold,revenue,price
2024-01-15,TRD-BLK-M,Black Dress Medium,Dresses,5,399.95,79.99
2024-01-15,JNS-BLU-32,Blue Jeans 32,Jeans,3,179.97,59.99
```

### Inventory Data CSV Format

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| sku | String | Product SKU/ID | TRD-BLK-M |
| product_name | String | Product name | Black Dress Medium |
| current_stock | Integer | Current units in stock | 145 |
| lead_time_days | Integer | Supplier lead time | 7 |
| category | String | Product category | Dresses |
| price | Float | Unit price | 79.99 |

**Example CSV:**
```csv
sku,product_name,current_stock,lead_time_days,category,price
TRD-BLK-M,Black Dress Medium,145,7,Dresses,79.99
JNS-BLU-32,Blue Jeans 32,89,5,Jeans,59.99
```

## 🎯 Usage Guide

### Using Sample Data

1. Launch the app
2. In the sidebar, select "Use Sample Data"
3. Click "Load Sample Data"
4. Explore the dashboard with pre-loaded demo data

### Uploading Your Own Data

1. Prepare your CSV files in the format above
2. In the sidebar, select "Upload Your Data"
3. Upload both Sales Data and Inventory Data files
4. The system will automatically process and analyze your data

### Navigating the Dashboard

#### Overview Tab
- View key metrics: Total SKUs, Critical Reorders, Inventory Value
- See 7-day demand forecast chart
- Check inventory health distribution
- Review urgent reorder recommendations
- Analyze sales by category

#### Forecasts Tab
- Select any SKU for detailed forecasting
- Choose forecast horizon (7-90 days)
- View confidence intervals
- Export forecast data

#### Reorders Tab
- Filter recommendations by status and category
- View detailed reorder analysis for each product
- Generate purchase orders with one click
- See lead time and stock information

#### Slow Movers Tab
- Identify overstocked items
- View health scores (0-100)
- Get recommended actions (discounts, bundles, clearance)
- Calculate carrying costs

## 🔧 Configuration

### Streamlit Configuration (Optional)

Create a `.streamlit/config.toml` file for custom settings:

```toml
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
```

### Environment Variables

Create a `.env` file for environment-specific settings:

```env
# Database (if using external database)
DATABASE_URL=postgresql://user:password@host:5432/dbname

# API Keys (if integrating with e-commerce platforms)
SHOPIFY_API_KEY=your_key_here
SHOPIFY_API_SECRET=your_secret_here

# ML Model Settings
FORECAST_CONFIDENCE_LEVEL=0.85
MIN_HISTORICAL_DAYS=365
```

## 🧪 Testing

### Manual Testing Checklist

- [ ] Sample data loads successfully
- [ ] CSV upload works for both files
- [ ] Forecast charts display correctly
- [ ] Reorder recommendations calculate properly
- [ ] Slow mover detection works
- [ ] All tabs are accessible
- [ ] Export functions work
- [ ] Mobile responsive design works

### Sample Data Generation

The app includes built-in sample data generation. To test with your own sample data:

```python
python -c "from app import generate_sample_data; sales, products, inventory = generate_sample_data(); sales.to_csv('sample_sales.csv', index=False); inventory.to_csv('sample_inventory.csv', index=False)"
```

## 🔐 Security Best Practices

1. **Never commit sensitive data** to GitHub
2. **Use `.gitignore`** to exclude data files and secrets
3. **Use Streamlit Secrets** for API keys and credentials
4. **Validate uploaded files** before processing
5. **Sanitize user inputs** in production
6. **Use HTTPS** for production deployments

## 📈 Performance Optimization

### For Large Datasets

If you have more than 10,000 SKUs or 2+ years of data:

1. **Use data sampling** for initial analysis
2. **Implement caching**:
```python
@st.cache_data
def load_data(file):
    return pd.read_csv(file)
```

3. **Optimize forecasting**:
   - Forecast only top-performing SKUs daily
   - Forecast others weekly/monthly
   - Use parallel processing for multiple SKUs

4. **Consider database integration**:
   - PostgreSQL for structured data
   - Redis for caching
   - S3 for file storage

## 🛠️ Troubleshooting

### Common Issues

**Issue: "ModuleNotFoundError: No module named 'prophet'"**
```bash
# Solution: Install Prophet dependencies
pip install pystan==2.19.1.1
pip install prophet==1.1.5
```

**Issue: "Cannot connect to Streamlit Cloud"**
- Check your GitHub repository is public
- Verify `requirements.txt` is in root directory
- Check Streamlit Cloud build logs for errors

**Issue: "Forecast taking too long"**
- Reduce forecast horizon
- Decrease number of SKUs being processed
- Implement caching with `@st.cache_data`

**Issue: "Memory error on large datasets"**
- Use data sampling/filtering
- Increase Streamlit Cloud resources (paid plan)
- Optimize data loading with `chunksize` parameter

### Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/ai-inventory-optimizer/issues)
- **Streamlit Community**: [Streamlit Forum](https://discuss.streamlit.io/)
- **Email Support**: support@example.com

## 🚀 Advanced Features (Coming Soon)

- [ ] **Real-time Shopify integration**
- [ ] **QuickBooks API connection**
- [ ] **Email alert notifications**
- [ ] **Multi-user authentication**
- [ ] **Advanced ML models (LSTM, XGBoost)**
- [ ] **Mobile app version**
- [ ] **Supplier portal integration**
- [ ] **Automated PO generation and sending**
- [ ] **Historical accuracy tracking**
- [ ] **A/B testing for pricing strategies**

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ai-inventory-optimizer.git
cd ai-inventory-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies including dev tools
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if you create this

# Run tests
pytest tests/

# Run app
streamlit run app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👏 Acknowledgments

- **Facebook Prophet** - Time series forecasting library
- **Streamlit** - Web app framework
- **Plotly** - Interactive visualization library
- **Pandas** - Data manipulation library

## 📞 Contact

**Your Name**
- Email: your.email@example.com
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-profile)
- GitHub: [@yourusername](https://github.com/yourusername)

**Project Link**: [https://github.com/yourusername/ai-inventory-optimizer](https://github.com/yourusername/ai-inventory-optimizer)

---

⭐ If you find this project helpful, please give it a star on GitHub!

Made with ❤️ for e-commerce businesses worldwide
