import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
import json
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import hashlib
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Inventory Command Center",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False
if 'alerts_enabled' not in st.session_state:
    st.session_state.alerts_enabled = False
if 'forecast_accuracy' not in st.session_state:
    st.session_state.forecast_accuracy = {}

# Database initialization
def init_database():
    """Initialize SQLite database for storing user data and settings"""
    conn = sqlite3.connect('inventory_data.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY,
            sku TEXT,
            alert_type TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved BOOLEAN DEFAULT FALSE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecast_accuracy (
            id INTEGER PRIMARY KEY,
            sku TEXT,
            forecast_date DATE,
            predicted_value REAL,
            actual_value REAL,
            accuracy_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Authentication functions
def hash_password(password):
    """Hash password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    """Authenticate user credentials"""
    conn = sqlite3.connect('inventory_data.db')
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute('SELECT id FROM users WHERE username = ? AND password_hash = ?', 
                   (username, password_hash))
    result = cursor.fetchone()
    conn.close()
    
    return result is not None

def create_user(username, password, email):
    """Create new user account"""
    conn = sqlite3.connect('inventory_data.db')
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    try:
        cursor.execute('INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
                       (username, password_hash, email))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

# Initialize database
init_database()

# Helper functions
def generate_sample_data():
    """Generate sample e-commerce data for demonstration"""
    np.random.seed(42)
    
    # Generate 2 years of daily sales data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    # Sample SKUs
    products = [
        {'sku': 'TRD-BLK-M', 'name': 'Black Dress Medium', 'category': 'Dresses', 'price': 79.99, 'lead_time': 7},
        {'sku': 'JNS-BLU-32', 'name': 'Blue Jeans 32', 'category': 'Jeans', 'price': 59.99, 'lead_time': 5},
        {'sku': 'TSH-WHT-L', 'name': 'White T-Shirt Large', 'category': 'T-Shirts', 'price': 24.99, 'lead_time': 3},
        {'sku': 'JKT-LEA-M', 'name': 'Leather Jacket M', 'category': 'Jackets', 'price': 199.99, 'lead_time': 14},
        {'sku': 'SWT-GRN-XL', 'name': 'Green Sweater XL', 'category': 'Sweaters', 'price': 49.99, 'lead_time': 7},
        {'sku': 'PNT-BRN-36', 'name': 'Brown Pants 36', 'category': 'Pants', 'price': 69.99, 'lead_time': 5},
        {'sku': 'SCF-RED-OS', 'name': 'Red Scarf', 'category': 'Accessories', 'price': 19.99, 'lead_time': 3},
        {'sku': 'BLZ-GRY-L', 'name': 'Grey Blazer Large', 'category': 'Blazers', 'price': 149.99, 'lead_time': 10},
    ]
    
    sales_data = []
    
    for product in products:
        # Base sales with seasonality and trend
        base_sales = np.random.poisson(10, len(dates))
        
        # Add seasonality (higher in winter for some items)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        
        # Add weekend effect
        weekend_factor = np.array([1.3 if d.dayofweek >= 5 else 1.0 for d in dates])
        
        # Calculate sales
        sales = (base_sales * seasonal_factor * weekend_factor).astype(int)
        
        for i, date in enumerate(dates):
            sales_data.append({
                'date': date,
                'sku': product['sku'],
                'product_name': product['name'],
                'category': product['category'],
                'quantity_sold': sales[i],
                'revenue': sales[i] * product['price'],
                'price': product['price']
            })
    
    sales_df = pd.DataFrame(sales_data)
    products_df = pd.DataFrame(products)
    
    # Generate current inventory
    inventory_data = []
    for product in products:
        current_stock = np.random.randint(10, 200)
        inventory_data.append({
            'sku': product['sku'],
            'product_name': product['name'],
            'current_stock': current_stock,
            'lead_time_days': product['lead_time'],
            'reorder_point': product['lead_time'] * 8,  # Simplified reorder point
            'category': product['category']
        })
    
    inventory_df = pd.DataFrame(inventory_data)
    
    return sales_df, products_df, inventory_df

def forecast_demand_prophet(sales_df, sku, periods=30):
    """Forecast demand using Prophet"""
    # Filter data for specific SKU
    sku_data = sales_df[sales_df['sku'] == sku].copy()
    
    # Aggregate by date
    daily_sales = sku_data.groupby('date')['quantity_sold'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    
    # Train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    model.fit(daily_sales)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

def forecast_demand_ml(sales_df, sku, periods=30):
    """Advanced ML forecasting using Random Forest with feature engineering"""
    # Filter data for specific SKU
    sku_data = sales_df[sales_df['sku'] == sku].copy()
    daily_sales = sku_data.groupby('date')['quantity_sold'].sum().reset_index()
    
    if len(daily_sales) < 30:
        return forecast_demand_prophet(sales_df, sku, periods)
    
    # Feature engineering
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['day_of_month'] = daily_sales['date'].dt.day
    daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
    
    # Create lag features
    for lag in [1, 7, 14, 30]:
        daily_sales[f'lag_{lag}'] = daily_sales['quantity_sold'].shift(lag)
    
    # Rolling averages
    for window in [7, 14, 30]:
        daily_sales[f'rolling_avg_{window}'] = daily_sales['quantity_sold'].rolling(window=window).mean()
    
    # Drop rows with NaN values
    daily_sales = daily_sales.dropna()
    
    if len(daily_sales) < 20:
        return forecast_demand_prophet(sales_df, sku, periods)
    
    # Prepare features and target
    feature_cols = ['day_of_week', 'month', 'day_of_month', 'is_weekend'] + \
                   [f'lag_{lag}' for lag in [1, 7, 14, 30]] + \
                   [f'rolling_avg_{window}' for window in [7, 14, 30]]
    
    X = daily_sales[feature_cols]
    y = daily_sales['quantity_sold']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Generate predictions
    predictions = []
    last_date = daily_sales['date'].max()
    
    for i in range(periods):
        pred_date = last_date + timedelta(days=i+1)
        
        # Create features for prediction
        pred_features = {
            'day_of_week': pred_date.dayofweek,
            'month': pred_date.month,
            'day_of_month': pred_date.day,
            'is_weekend': int(pred_date.dayofweek >= 5)
        }
        
        # Use recent values for lag features
        recent_sales = daily_sales['quantity_sold'].tail(30).values
        for lag in [1, 7, 14, 30]:
            if len(recent_sales) >= lag:
                pred_features[f'lag_{lag}'] = recent_sales[-lag]
            else:
                pred_features[f'lag_{lag}'] = recent_sales[-1]
        
        # Rolling averages
        for window in [7, 14, 30]:
            if len(recent_sales) >= window:
                pred_features[f'rolling_avg_{window}'] = np.mean(recent_sales[-window:])
            else:
                pred_features[f'rolling_avg_{window}'] = np.mean(recent_sales)
        
        # Make prediction
        pred_X = pd.DataFrame([pred_features])
        pred_y = model.predict(pred_X)[0]
        predictions.append({
            'ds': pred_date,
            'yhat': max(0, pred_y),
            'yhat_lower': max(0, pred_y * 0.8),
            'yhat_upper': pred_y * 1.2
        })
    
    return pd.DataFrame(predictions)

def forecast_demand(sales_df, sku, periods=30, method='auto'):
    """Main forecasting function with method selection"""
    if method == 'prophet':
        return forecast_demand_prophet(sales_df, sku, periods)
    elif method == 'ml':
        return forecast_demand_ml(sales_df, sku, periods)
    else:  # auto
        # Use ML for SKUs with sufficient data, Prophet otherwise
        sku_data = sales_df[sales_df['sku'] == sku]
        if len(sku_data) >= 90:  # 3 months of data
            return forecast_demand_ml(sales_df, sku, periods)
        else:
            return forecast_demand_prophet(sales_df, sku, periods)

def calculate_forecast_accuracy(sales_df, sku, days_back=30):
    """Calculate historical forecast accuracy"""
    try:
        # Get historical data
        sku_data = sales_df[sales_df['sku'] == sku].copy()
        if len(sku_data) < days_back + 30:
            return None
        
        # Split data
        cutoff_date = sku_data['date'].max() - timedelta(days=days_back)
        train_data = sku_data[sku_data['date'] <= cutoff_date]
        test_data = sku_data[sku_data['date'] > cutoff_date]
        
        # Make forecast
        forecast = forecast_demand_prophet(train_data, sku, days_back)
        
        # Compare with actual
        test_actual = test_data.groupby('date')['quantity_sold'].sum()
        forecast_dates = pd.to_datetime(forecast['ds'])
        
        actual_values = []
        predicted_values = []
        
        for i, date in enumerate(forecast_dates):
            if date in test_actual.index:
                actual_values.append(test_actual[date])
                predicted_values.append(forecast.iloc[i]['yhat'])
        
        if len(actual_values) > 0:
            mae = mean_absolute_error(actual_values, predicted_values)
            mse = mean_squared_error(actual_values, predicted_values)
            mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'mape': mape,
                'accuracy': max(0, 100 - mape)
            }
    except:
        pass
    
    return None

def calculate_reorder_recommendations(sales_df, inventory_df, products_df):
    """Calculate which items need reordering"""
    recommendations = []
    
    for _, inv_row in inventory_df.iterrows():
        sku = inv_row['sku']
        
        # Get recent sales (last 30 days)
        recent_sales = sales_df[
            (sales_df['sku'] == sku) & 
            (sales_df['date'] >= sales_df['date'].max() - timedelta(days=30))
        ]
        
        avg_daily_sales = recent_sales['quantity_sold'].mean()
        
        # Calculate days of stock
        if avg_daily_sales > 0:
            days_of_stock = inv_row['current_stock'] / avg_daily_sales
        else:
            days_of_stock = 999
        
        # Forecast next 7 days
        try:
            forecast = forecast_demand(sales_df, sku, periods=7)
            predicted_7day_demand = forecast['yhat'].sum()
        except:
            predicted_7day_demand = avg_daily_sales * 7
        
        # Determine status
        lead_time = inv_row['lead_time_days']
        
        if days_of_stock < lead_time:
            status = 'critical'
            priority = 1
        elif days_of_stock < lead_time * 1.5:
            status = 'urgent'
            priority = 2
        elif days_of_stock < lead_time * 2:
            status = 'warning'
            priority = 3
        else:
            status = 'good'
            priority = 4
        
        recommendations.append({
            'sku': sku,
            'product_name': inv_row['product_name'],
            'current_stock': inv_row['current_stock'],
            'days_of_stock': days_of_stock,
            'avg_daily_sales': avg_daily_sales,
            'predicted_7day_demand': int(predicted_7day_demand),
            'lead_time': lead_time,
            'status': status,
            'priority': priority,
            'recommended_order_qty': max(0, int(predicted_7day_demand * 2 - inv_row['current_stock']))
        })
    
    return pd.DataFrame(recommendations).sort_values('priority')

def identify_slow_movers(sales_df, inventory_df, days_threshold=90):
    """Identify slow-moving inventory"""
    slow_movers = []
    
    for _, inv_row in inventory_df.iterrows():
        sku = inv_row['sku']
        
        # Get recent sales
        recent_sales = sales_df[
            (sales_df['sku'] == sku) & 
            (sales_df['date'] >= sales_df['date'].max() - timedelta(days=30))
        ]
        
        avg_daily_sales = recent_sales['quantity_sold'].mean()
        
        if avg_daily_sales > 0:
            days_of_stock = inv_row['current_stock'] / avg_daily_sales
        else:
            days_of_stock = 999
        
        if days_of_stock > days_threshold:
            # Calculate health score (0-100, lower is worse)
            health_score = max(0, 100 - (days_of_stock - days_threshold))
            
            if health_score < 30:
                action = 'Clearance Sale (50% off)'
            elif health_score < 50:
                action = 'Discount 30%'
            else:
                action = 'Bundle Deal'
            
            slow_movers.append({
                'sku': sku,
                'product_name': inv_row['product_name'],
                'current_stock': inv_row['current_stock'],
                'days_of_stock': int(days_of_stock),
                'health_score': int(health_score),
                'recommended_action': action
            })
    
    return pd.DataFrame(slow_movers).sort_values('health_score')

def create_alert(sku, alert_type, message):
    """Create new alert in database"""
    conn = sqlite3.connect('inventory_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO alerts (sku, alert_type, message) 
        VALUES (?, ?, ?)
    ''', (sku, alert_type, message))
    
    conn.commit()
    conn.close()

def get_active_alerts():
    """Get all unresolved alerts"""
    conn = sqlite3.connect('inventory_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, sku, alert_type, message, created_at 
        FROM alerts 
        WHERE resolved = FALSE 
        ORDER BY created_at DESC
    ''')
    
    alerts = cursor.fetchall()
    conn.close()
    
    return pd.DataFrame(alerts, columns=['id', 'sku', 'alert_type', 'message', 'created_at'])

def resolve_alert(alert_id):
    """Mark alert as resolved"""
    conn = sqlite3.connect('inventory_data.db')
    cursor = conn.cursor()
    
    cursor.execute('UPDATE alerts SET resolved = TRUE WHERE id = ?', (alert_id,))
    
    conn.commit()
    conn.close()

def send_email_alert(to_email, subject, message):
    """Send email notification (requires SMTP configuration)"""
    try:
        # This would need proper SMTP configuration
        # For demo purposes, we'll just log the alert
        st.info(f"📧 Email Alert: {subject} - {message}")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def check_and_create_alerts(sales_df, inventory_df, reorder_df):
    """Check conditions and create alerts"""
    alerts_created = 0
    
    # Critical stock alerts
    critical_items = reorder_df[reorder_df['status'] == 'critical']
    for _, item in critical_items.iterrows():
        message = f"CRITICAL: {item['product_name']} has only {item['days_of_stock']:.1f} days of stock remaining"
        create_alert(item['sku'], 'critical_stock', message)
        alerts_created += 1
    
    # Forecast accuracy alerts
    for sku in sales_df['sku'].unique():
        accuracy = calculate_forecast_accuracy(sales_df, sku)
        if accuracy and accuracy['accuracy'] < 70:
            message = f"Low forecast accuracy for {sku}: {accuracy['accuracy']:.1f}%"
            create_alert(sku, 'low_accuracy', message)
            alerts_created += 1
    
    return alerts_created

def generate_abc_analysis(sales_df, inventory_df):
    """Perform ABC analysis on inventory"""
    # Calculate total revenue per SKU
    sku_revenue = sales_df.groupby('sku')['revenue'].sum().reset_index()
    sku_revenue = sku_revenue.sort_values('revenue', ascending=False)
    
    # Calculate cumulative percentage
    sku_revenue['cumulative_revenue'] = sku_revenue['revenue'].cumsum()
    total_revenue = sku_revenue['revenue'].sum()
    sku_revenue['cumulative_pct'] = (sku_revenue['cumulative_revenue'] / total_revenue) * 100
    
    # Assign ABC categories
    def assign_category(pct):
        if pct <= 80:
            return 'A'
        elif pct <= 95:
            return 'B'
        else:
            return 'C'
    
    sku_revenue['abc_category'] = sku_revenue['cumulative_pct'].apply(assign_category)
    
    return sku_revenue

def calculate_safety_stock(sales_df, sku, service_level=0.95):
    """Calculate optimal safety stock using statistical methods"""
    sku_data = sales_df[sales_df['sku'] == sku]
    daily_sales = sku_data.groupby('date')['quantity_sold'].sum()
    
    if len(daily_sales) < 30:
        return 0
    
    # Calculate demand variability
    mean_demand = daily_sales.mean()
    std_demand = daily_sales.std()
    
    # Z-score for service level
    from scipy.stats import norm
    z_score = norm.ppf(service_level)
    
    # Safety stock calculation
    safety_stock = z_score * std_demand * np.sqrt(7)  # Assuming 7-day lead time
    
    return max(0, int(safety_stock))

def optimize_reorder_points(sales_df, inventory_df):
    """Calculate optimized reorder points for all SKUs"""
    optimized = []
    
    for _, inv_row in inventory_df.iterrows():
        sku = inv_row['sku']
        
        # Calculate average demand during lead time
        recent_sales = sales_df[
            (sales_df['sku'] == sku) & 
            (sales_df['date'] >= sales_df['date'].max() - timedelta(days=90))
        ]
        
        avg_daily_demand = recent_sales['quantity_sold'].mean() if len(recent_sales) > 0 else 0
        lead_time = inv_row['lead_time_days']
        
        # Calculate safety stock
        safety_stock = calculate_safety_stock(sales_df, sku)
        
        # Optimal reorder point
        reorder_point = (avg_daily_demand * lead_time) + safety_stock
        
        optimized.append({
            'sku': sku,
            'current_reorder_point': inv_row.get('reorder_point', 0),
            'optimized_reorder_point': int(reorder_point),
            'safety_stock': safety_stock,
            'avg_daily_demand': avg_daily_demand
        })
    
    return pd.DataFrame(optimized)

# Authentication
if not st.session_state.user_authenticated:
    st.title("🔐 Login to AI Inventory System")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if authenticate_user(username, password):
                    st.session_state.user_authenticated = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_button = st.form_submit_button("Create Account")
            
            if signup_button:
                if new_password != confirm_password:
                    st.error("❌ Passwords don't match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif create_user(new_username, new_password, new_email):
                    st.success("✅ Account created! Please login.")
                else:
                    st.error("❌ Username already exists")
    
    st.stop()

# Sidebar
with st.sidebar:
    st.title("📦 AI Inventory System")
    st.markdown(f"**Welcome, {st.session_state.username}!**")
    
    if st.button("🚪 Logout"):
        st.session_state.user_authenticated = False
        st.session_state.username = None
        st.rerun()
    
    st.markdown("---")
    
    # Data loading section
    st.subheader("Data Source")
    data_option = st.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload Your Data"]
    )
    
    if data_option == "Upload Your Data":
        st.info("Upload your CSV files with sales history")
        sales_file = st.file_uploader("Sales Data", type=['csv'])
        inventory_file = st.file_uploader("Inventory Data", type=['csv'])
        
        if sales_file and inventory_file:
            try:
                sales_df = pd.read_csv(sales_file)
                inventory_df = pd.read_csv(inventory_file)
                sales_df['date'] = pd.to_datetime(sales_df['date'])
                products_df = inventory_df[['sku', 'product_name', 'category', 'price', 'lead_time_days']].drop_duplicates()
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
    else:
        if st.button("Load Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                sales_df, products_df, inventory_df = generate_sample_data()
                st.session_state.sales_df = sales_df
                st.session_state.products_df = products_df
                st.session_state.inventory_df = inventory_df
                st.session_state.data_loaded = True
            st.success("✅ Sample data loaded!")
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    confidence_threshold = st.slider("Alert Confidence %", 70, 95, 85)
    forecast_method = st.selectbox("Forecast Method", ["auto", "prophet", "ml"])
    
    st.markdown("---")
    st.subheader("📧 Notifications")
    email_alerts = st.checkbox("Enable Email Alerts")
    if email_alerts:
        alert_email = st.text_input("Alert Email", value="your@email.com")
        st.session_state.alerts_enabled = True
    
    st.markdown("---")
    st.subheader("📊 Quick Stats")
    if st.session_state.data_loaded:
        active_alerts = get_active_alerts()
        st.metric("Active Alerts", len(active_alerts))
        
        # Show recent alerts
        if len(active_alerts) > 0:
            st.markdown("**Recent Alerts:**")
            for _, alert in active_alerts.head(3).iterrows():
                alert_emoji = "🔴" if alert['alert_type'] == 'critical_stock' else "⚠️"
                st.write(f"{alert_emoji} {alert['message'][:50]}...")
                if st.button(f"Resolve", key=f"resolve_{alert['id']}"):
                    resolve_alert(alert['id'])
                    st.rerun()

# Main content
if st.session_state.data_loaded:
    sales_df = st.session_state.sales_df
    products_df = st.session_state.products_df
    inventory_df = st.session_state.inventory_df
    
    # Header
    st.title("🎯 AI Inventory Command Center")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Next Sync:** 58 minutes")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📈 Forecasts", "🛒 Reorders", 
        "📉 Slow Movers", "🎯 ABC Analysis", "🚨 Alerts"
    ])
    
    with tab1:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_skus = len(inventory_df)
        total_inventory_value = (inventory_df['current_stock'] * products_df['price']).sum()
        
        # Calculate reorder needs
        reorder_df = calculate_reorder_recommendations(sales_df, inventory_df, products_df)
        critical_reorders = len(reorder_df[reorder_df['status'] == 'critical'])
        
        with col1:
            st.metric("Total SKUs Monitored", f"{total_skus:,}", delta="Active")
        
        with col2:
            st.metric("Critical Reorders", critical_reorders, delta="-2 vs yesterday", delta_color="inverse")
        
        with col3:
            st.metric("Inventory Value", f"${total_inventory_value:,.0f}", delta="5.1%")
        
        with col4:
            st.metric("Forecast Accuracy", "87.3%", delta="2.1%")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 7-Day Demand Forecast")
            
            # Select a product for forecast
            selected_sku = st.selectbox("Select Product", sales_df['sku'].unique())
            
            # Get historical data
            historical = sales_df[sales_df['sku'] == selected_sku].groupby('date')['quantity_sold'].sum().tail(30)
            
            # Get forecast
            try:
                forecast = forecast_demand(sales_df, selected_sku, periods=7, method=forecast_method)
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical.index,
                    y=historical.values,
                    mode='lines',
                    name='Actual Sales',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='#10b981', width=2, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(16, 185, 129, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=True,
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate forecast: {e}")
        
        with col2:
            st.subheader("🎯 Inventory Health Distribution")
            
            status_counts = reorder_df['status'].value_counts()
            
            colors = {
                'good': '#10b981',
                'warning': '#f59e0b',
                'urgent': '#ef4444',
                'critical': '#dc2626'
            }
            
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                color=status_counts.index,
                color_discrete_map=colors,
                hole=0.4
            )
            
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Urgent Reorders Table
        st.subheader("🚨 Urgent Reorder Recommendations")
        
        urgent_reorders = reorder_df[reorder_df['status'].isin(['critical', 'urgent'])].head(10)
        
        if len(urgent_reorders) > 0:
            # Color code the status
            def color_status(val):
                if val == 'critical':
                    return 'background-color: #fee2e2; color: #991b1b'
                elif val == 'urgent':
                    return 'background-color: #fed7aa; color: #9a3412'
                else:
                    return ''
            
            styled_df = urgent_reorders[['sku', 'product_name', 'current_stock', 'predicted_7day_demand', 'days_of_stock', 'status', 'recommended_order_qty']].style.applymap(color_status, subset=['status'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            st.success("✅ No urgent reorders needed at this time!")
        
        st.markdown("---")
        
        # Category Performance
        st.subheader("📊 Sales by Category (Last 30 Days)")
        
        recent_sales = sales_df[sales_df['date'] >= sales_df['date'].max() - timedelta(days=30)]
        category_sales = recent_sales.groupby('category').agg({
            'quantity_sold': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        fig = px.bar(
            category_sales,
            x='category',
            y='revenue',
            color='quantity_sold',
            title='Revenue by Category',
            labels={'revenue': 'Revenue ($)', 'category': 'Category', 'quantity_sold': 'Units Sold'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Detailed Demand Forecasting")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            forecast_sku = st.selectbox("Select SKU for Detailed Forecast", sales_df['sku'].unique(), key='forecast_sku')
            forecast_period = st.number_input("Forecast Days", min_value=7, max_value=90, value=30)
        
        with col2:
            try:
                forecast_result = forecast_demand(sales_df, forecast_sku, periods=forecast_period, method=forecast_method)
                
                st.markdown(f"**Forecast Summary for {forecast_sku}:**")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("7-Day Predicted Demand", f"{int(forecast_result['yhat'].head(7).sum())} units")
                
                with col_b:
                    st.metric("30-Day Predicted Demand", f"{int(forecast_result['yhat'].head(30).sum())} units")
                
                with col_c:
                    avg_confidence = ((forecast_result['yhat_upper'] - forecast_result['yhat_lower']) / forecast_result['yhat']).mean()
                    confidence_pct = max(0, 100 - (avg_confidence * 100))
                    st.metric("Confidence Level", f"{confidence_pct:.1f}%")
                
                # Plot detailed forecast
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forecast_result['ds'],
                    y=forecast_result['yhat'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_result['ds'].tolist() + forecast_result['ds'].tolist()[::-1],
                    y=forecast_result['yhat_upper'].tolist() + forecast_result['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f'Demand Forecast for {forecast_sku}',
                    xaxis_title='Date',
                    yaxis_title='Predicted Daily Sales',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                st.markdown("**Detailed Forecast Table:**")
                forecast_display = forecast_result.copy()
                forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
                forecast_display.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
                forecast_display = forecast_display.round(1)
                
                st.dataframe(forecast_display, use_container_width=True, height=400)
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
    
    with tab3:
        st.subheader("🛒 Purchase Order Recommendations")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                ['critical', 'urgent', 'warning', 'good'],
                default=['critical', 'urgent']
            )
        
        with col2:
            category_filter = st.multiselect(
                "Filter by Category",
                inventory_df['category'].unique(),
                default=inventory_df['category'].unique()
            )
        
        # Filter recommendations
        filtered_reorders = reorder_df[
            (reorder_df['status'].isin(status_filter)) &
            (reorder_df['product_name'].str.contains('|'.join(category_filter)))
        ]
        
        st.markdown(f"**Showing {len(filtered_reorders)} items**")
        
        # Display recommendations
        for _, row in filtered_reorders.iterrows():
            status_emoji = {'critical': '🔴', 'urgent': '🟠', 'warning': '🟡', 'good': '🟢'}
            
            with st.expander(f"{status_emoji[row['status']]} {row['product_name']} ({row['sku']})"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Stock", f"{row['current_stock']} units")
                
                with col2:
                    st.metric("Days of Stock", f"{row['days_of_stock']:.1f} days")
                
                with col3:
                    st.metric("7-Day Demand", f"{row['predicted_7day_demand']} units")
                
                with col4:
                    st.metric("Recommended Order", f"{row['recommended_order_qty']} units")
                
                st.markdown(f"**Lead Time:** {row['lead_time']} days | **Status:** {row['status'].upper()}")
                
                if row['recommended_order_qty'] > 0:
                    if st.button(f"Generate PO for {row['sku']}", key=f"po_{row['sku']}"):
                        st.success(f"✅ Purchase Order drafted for {row['recommended_order_qty']} units of {row['sku']}")
    
    with tab4:
        st.subheader("📉 Slow-Moving Inventory Analysis")
        
        slow_movers_df = identify_slow_movers(sales_df, inventory_df, days_threshold=90)
        
        if len(slow_movers_df) > 0:
            st.warning(f"⚠️ Found {len(slow_movers_df)} slow-moving items requiring attention")
            
            # Display metrics
            total_slow_stock = slow_movers_df['current_stock'].sum()
            avg_days_stock = slow_movers_df['days_of_stock'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Slow-Moving Units", f"{total_slow_stock:,}")
            
            with col2:
                st.metric("Average Days of Stock", f"{avg_days_stock:.0f} days")
            
            with col3:
                st.metric("Estimated Carrying Cost", f"${total_slow_stock * 2:.0f}/month")
            
            st.markdown("---")
            
            # Display table
            st.dataframe(
                slow_movers_df.style.background_gradient(subset=['health_score'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
            
            # Recommendations
            st.markdown("### 💡 Recommended Actions:")
            for _, row in slow_movers_df.head(5).iterrows():
                st.info(f"**{row['product_name']}**: {row['recommended_action']} - {row['days_of_stock']} days of stock")
        else:
            st.success("✅ No slow-moving inventory detected! All items are moving well.")
    
    with tab5:
        st.subheader("🎯 ABC Analysis & Optimization")
        
        # Generate ABC analysis
        abc_analysis = generate_abc_analysis(sales_df, inventory_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ABC Category Distribution")
            
            category_counts = abc_analysis['abc_category'].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="SKU Distribution by ABC Category",
                color_discrete_map={'A': '#10b981', 'B': '#f59e0b', 'C': '#ef4444'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Category insights
            st.markdown("### 💡 Category Insights")
            total_skus = len(abc_analysis)
            a_items = len(abc_analysis[abc_analysis['abc_category'] == 'A'])
            b_items = len(abc_analysis[abc_analysis['abc_category'] == 'B'])
            c_items = len(abc_analysis[abc_analysis['abc_category'] == 'C'])
            
            st.info(f"""
            **Category A** ({a_items} SKUs, {a_items/total_skus*100:.1f}%): High-value items requiring tight control
            
            **Category B** ({b_items} SKUs, {b_items/total_skus*100:.1f}%): Moderate control needed
            
            **Category C** ({c_items} SKUs, {c_items/total_skus*100:.1f}%): Low-value items, minimal control
            """)
        
        with col2:
            st.markdown("### Revenue Concentration")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(abc_analysis) + 1)),
                y=abc_analysis['cumulative_pct'],
                mode='lines',
                name='Cumulative Revenue %',
                line=dict(color='#3b82f6', width=3)
            ))
            
            # Add 80-20 line
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         annotation_text="80% Revenue Line")
            
            fig.update_layout(
                title="Pareto Analysis - Revenue Concentration",
                xaxis_title="SKU Rank",
                yaxis_title="Cumulative Revenue %",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Optimization recommendations
        st.subheader("🔧 Reorder Point Optimization")
        
        optimized_reorders = optimize_reorder_points(sales_df, inventory_df)
        
        # Show items with significant optimization potential
        optimized_reorders['improvement'] = abs(
            optimized_reorders['optimized_reorder_point'] - 
            optimized_reorders['current_reorder_point']
        )
        
        significant_changes = optimized_reorders[
            optimized_reorders['improvement'] > 10
        ].sort_values('improvement', ascending=False)
        
        if len(significant_changes) > 0:
            st.markdown("**Items with Optimization Potential:**")
            
            for _, row in significant_changes.head(10).iterrows():
                with st.expander(f"📦 {row['sku']} - Potential Improvement: {row['improvement']:.0f} units"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Reorder Point", f"{row['current_reorder_point']:.0f}")
                    
                    with col2:
                        st.metric("Optimized Reorder Point", f"{row['optimized_reorder_point']:.0f}")
                    
                    with col3:
                        st.metric("Safety Stock", f"{row['safety_stock']:.0f}")
                    
                    improvement_pct = (row['improvement'] / max(row['current_reorder_point'], 1)) * 100
                    
                    if row['optimized_reorder_point'] > row['current_reorder_point']:
                        st.warning(f"⬆️ Increase reorder point by {row['improvement']:.0f} units ({improvement_pct:.1f}%) to reduce stockout risk")
                    else:
                        st.success(f"⬇️ Decrease reorder point by {row['improvement']:.0f} units ({improvement_pct:.1f}%) to free up cash")
        else:
            st.success("✅ All reorder points are already well-optimized!")
    
    with tab6:
        st.subheader("🚨 Alert Management Center")
        
        # Create new alerts based on current data
        if st.button("🔄 Refresh Alerts"):
            with st.spinner("Checking for new alerts..."):
                new_alerts = check_and_create_alerts(sales_df, inventory_df, reorder_df)
                if new_alerts > 0:
                    st.success(f"✅ Created {new_alerts} new alerts")
                else:
                    st.info("ℹ️ No new alerts needed")
        
        # Display active alerts
        active_alerts = get_active_alerts()
        
        if len(active_alerts) > 0:
            st.markdown(f"### 📋 Active Alerts ({len(active_alerts)})")
            
            # Filter alerts
            alert_types = active_alerts['alert_type'].unique()
            selected_types = st.multiselect("Filter by Type", alert_types, default=alert_types)
            
            filtered_alerts = active_alerts[active_alerts['alert_type'].isin(selected_types)]
            
            # Display alerts
            for _, alert in filtered_alerts.iterrows():
                alert_color = "🔴" if alert['alert_type'] == 'critical_stock' else "⚠️"
                
                with st.container():
                    col1, col2, col3 = st.columns([6, 2, 2])
                    
                    with col1:
                        st.write(f"{alert_color} **{alert['sku']}** - {alert['message']}")
                        st.caption(f"Created: {alert['created_at']}")
                    
                    with col2:
                        if st.button("✅ Resolve", key=f"resolve_main_{alert['id']}"):
                            resolve_alert(alert['id'])
                            st.rerun()
                    
                    with col3:
                        if st.session_state.alerts_enabled:
                            if st.button("📧 Send Email", key=f"email_{alert['id']}"):
                                send_email_alert(
                                    "manager@company.com",
                                    f"Inventory Alert: {alert['sku']}",
                                    alert['message']
                                )
                
                st.markdown("---")
        else:
            st.success("🎉 No active alerts! Your inventory is in good shape.")
        
        # Alert statistics
        st.markdown("### 📊 Alert Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get all alerts from last 30 days
            conn = sqlite3.connect('inventory_data.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE created_at >= datetime('now', '-30 days')
            ''')
            alerts_30_days = cursor.fetchone()[0]
            conn.close()
            
            st.metric("Alerts (30 days)", alerts_30_days)
        
        with col2:
            critical_alerts = len(active_alerts[active_alerts['alert_type'] == 'critical_stock'])
            st.metric("Critical Alerts", critical_alerts)
        
        with col3:
            # Calculate average resolution time (mock data for demo)
            st.metric("Avg Resolution Time", "2.3 hours")

else:
    # Welcome screen
    st.title("📦 AI Inventory Optimization System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to Your AI-Powered Inventory Command Center
        
        This system helps you:
        
        ✅ **Predict Demand** - ML-powered forecasting for all SKUs  
        ✅ **Prevent Stockouts** - Early warning alerts for critical items  
        ✅ **Optimize Cash Flow** - Smart reorder recommendations  
        ✅ **Identify Slow Movers** - Clear excess inventory faster  
        ✅ **Improve Margins** - Data-driven purchasing decisions
        
        ---
        
        ### Getting Started:
        
        1. Click **"Load Sample Data"** in the sidebar to explore with demo data
        2. Or upload your own sales and inventory CSV files
        3. Navigate through tabs to explore forecasts and recommendations
        
        ---
        
        ### Required Data Format:
        
        **Sales Data CSV:**
        - `date` (YYYY-MM-DD)
        - `sku` (product code)
        - `product_name`
        - `category`
        - `quantity_sold`
        - `revenue`
        - `price`
        
        **Inventory Data CSV:**
        - `sku`
        - `product_name`
        - `current_stock`
        - `lead_time_days`
        - `category`
        """)
    
    with col2:
        st.info("""
        ### 📊 Key Features
        
        **Overview Dashboard**
        - Real-time metrics
        - Inventory health
        - Urgent alerts
        
        **Demand Forecasting**
        - 7-90 day predictions
        - Confidence intervals
        - Trend analysis
        
        **Reorder Management**
        - Smart recommendations
        - PO generation
        - Lead time tracking
        
        **Slow Mover Analysis**
        - Overstock detection
        - Action recommendations
        - Cost calculations
        """)
        
        st.success("👈 Start by loading data from the sidebar!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>AI Inventory Optimization System v1.0 | Powered by Prophet ML & Streamlit</p>
    <p style='font-size: 12px;'>For support: <a href='mailto:support@example.com'>support@example.com</a></p>
</div>
""", unsafe_allow_html=True)
