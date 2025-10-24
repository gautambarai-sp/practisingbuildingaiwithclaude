import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
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
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Generate sample data
@st.cache_data
def generate_sample_data():
    """Generate sample e-commerce data for demonstration"""
    np.random.seed(42)
    
    # Generate 6 months of daily sales data
    dates = pd.date_range(start='2024-06-01', end='2024-12-31', freq='D')
    
    # Sample SKUs
    products = [
        {'sku': 'TRD-BLK-M', 'name': 'Black Dress Medium', 'category': 'Dresses', 'price': 79.99, 'lead_time': 7},
        {'sku': 'JNS-BLU-32', 'name': 'Blue Jeans 32', 'category': 'Jeans', 'price': 59.99, 'lead_time': 5},
        {'sku': 'TSH-WHT-L', 'name': 'White T-Shirt Large', 'category': 'T-Shirts', 'price': 24.99, 'lead_time': 3},
        {'sku': 'JKT-LEA-M', 'name': 'Leather Jacket M', 'category': 'Jackets', 'price': 199.99, 'lead_time': 14},
        {'sku': 'SWT-GRN-XL', 'name': 'Green Sweater XL', 'category': 'Sweaters', 'price': 49.99, 'lead_time': 7},
    ]
    
    sales_data = []
    
    for product in products:
        # Base sales with seasonality
        base_sales = np.random.poisson(8, len(dates))
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        sales = (base_sales * seasonal_factor).astype(int)
        
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
        current_stock = np.random.randint(20, 150)
        inventory_data.append({
            'sku': product['sku'],
            'product_name': product['name'],
            'current_stock': current_stock,
            'lead_time_days': product['lead_time'],
            'reorder_point': product['lead_time'] * 8,
            'category': product['category']
        })
    
    inventory_df = pd.DataFrame(inventory_data)
    
    return sales_df, products_df, inventory_df

# Simple forecasting
@st.cache_data
def simple_forecast(sales_df, sku, periods=30):
    """Simple moving average forecast"""
    sku_data = sales_df[sales_df['sku'] == sku].copy()
    daily_sales = sku_data.groupby('date')['quantity_sold'].sum().reset_index()
    
    if len(daily_sales) < 7:
        avg_sales = 5  # Default
        forecast_data = []
        last_date = sales_df['date'].max()
        
        for i in range(periods):
            forecast_data.append({
                'ds': last_date + timedelta(days=i+1),
                'yhat': avg_sales,
                'yhat_lower': avg_sales * 0.8,
                'yhat_upper': avg_sales * 1.2
            })
        
        return pd.DataFrame(forecast_data)
    
    # Moving average
    window = min(14, len(daily_sales))
    recent_avg = daily_sales['quantity_sold'].tail(window).mean()
    
    forecast_data = []
    last_date = daily_sales['date'].max()
    
    for i in range(periods):
        predicted = max(0, recent_avg)
        forecast_data.append({
            'ds': last_date + timedelta(days=i+1),
            'yhat': predicted,
            'yhat_lower': predicted * 0.8,
            'yhat_upper': predicted * 1.2
        })
    
    return pd.DataFrame(forecast_data)

@st.cache_data
def calculate_reorder_recommendations(sales_df, inventory_df, products_df):
    """Calculate reorder recommendations"""
    recommendations = []
    
    for _, inv_row in inventory_df.iterrows():
        sku = inv_row['sku']
        
        # Get recent sales
        recent_sales = sales_df[
            (sales_df['sku'] == sku) & 
            (sales_df['date'] >= sales_df['date'].max() - timedelta(days=30))
        ]
        
        avg_daily_sales = recent_sales['quantity_sold'].mean() if len(recent_sales) > 0 else 0
        
        # Calculate days of stock
        if avg_daily_sales > 0:
            days_of_stock = inv_row['current_stock'] / avg_daily_sales
        else:
            days_of_stock = 999
        
        # Forecast 7 days
        forecast = simple_forecast(sales_df, sku, periods=7)
        predicted_7day_demand = forecast['yhat'].sum()
        
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

@st.cache_data
def identify_slow_movers(sales_df, inventory_df, days_threshold=90):
    """Identify slow-moving inventory"""
    slow_movers = []
    
    for _, inv_row in inventory_df.iterrows():
        sku = inv_row['sku']
        
        recent_sales = sales_df[
            (sales_df['sku'] == sku) & 
            (sales_df['date'] >= sales_df['date'].max() - timedelta(days=30))
        ]
        
        avg_daily_sales = recent_sales['quantity_sold'].mean() if len(recent_sales) > 0 else 0
        
        if avg_daily_sales > 0:
            days_of_stock = inv_row['current_stock'] / avg_daily_sales
        else:
            days_of_stock = 999
        
        if days_of_stock > days_threshold:
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

# Sidebar
with st.sidebar:
    st.title("📦 AI Inventory System")
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
    st.subheader("Settings")
    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)

# Main content
if st.session_state.data_loaded:
    sales_df = st.session_state.sales_df
    products_df = st.session_state.products_df
    inventory_df = st.session_state.inventory_df
    
    # Header
    st.title("🎯 AI Inventory Command Center")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Status:** ✅ Online")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Forecasts", "🛒 Reorders", "📉 Slow Movers"])
    
    with tab1:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_skus = len(inventory_df)
        total_inventory_value = (inventory_df['current_stock'] * products_df['price']).sum()
        
        # Calculate reorder needs
        reorder_df = calculate_reorder_recommendations(sales_df, inventory_df, products_df)
        critical_reorders = len(reorder_df[reorder_df['status'] == 'critical'])
        
        with col1:
            st.metric("Total SKUs", f"{total_skus:,}", delta="Active")
        
        with col2:
            st.metric("Critical Reorders", critical_reorders, delta="-2 vs yesterday", delta_color="inverse")
        
        with col3:
            st.metric("Inventory Value", f"${total_inventory_value:,.0f}", delta="5.1%")
        
        with col4:
            st.metric("System Status", "Online", delta="Optimized")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 7-Day Demand Forecast")
            
            selected_sku = st.selectbox("Select Product", sales_df['sku'].unique())
            
            # Get historical data
            historical = sales_df[sales_df['sku'] == selected_sku].groupby('date')['quantity_sold'].sum().tail(30)
            
            # Get forecast
            forecast = simple_forecast(sales_df, selected_sku, periods=7)
            
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
            
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Inventory Health")
            
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
            st.dataframe(
                urgent_reorders[['sku', 'product_name', 'current_stock', 'predicted_7day_demand', 'days_of_stock', 'status', 'recommended_order_qty']],
                use_container_width=True,
                height=400
            )
        else:
            st.success("✅ No urgent reorders needed!")
        
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
            title='Revenue by Category'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Detailed Demand Forecasting")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            forecast_sku = st.selectbox("Select SKU", sales_df['sku'].unique(), key='forecast_sku')
            forecast_period = st.number_input("Forecast Days", min_value=7, max_value=90, value=30)
        
        with col2:
            forecast_result = simple_forecast(sales_df, forecast_sku, periods=forecast_period)
            
            st.markdown(f"**Forecast Summary for {forecast_sku}:**")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("7-Day Demand", f"{int(forecast_result['yhat'].head(7).sum())} units")
            
            with col_b:
                st.metric("30-Day Demand", f"{int(forecast_result['yhat'].head(30).sum())} units")
            
            with col_c:
                st.metric("Method", "Moving Average")
            
            # Plot forecast
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
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🛒 Purchase Order Recommendations")
        
        for _, row in reorder_df[reorder_df['status'].isin(['critical', 'urgent'])].iterrows():
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
                
                if row['recommended_order_qty'] > 0:
                    if st.button(f"Generate PO for {row['sku']}", key=f"po_{row['sku']}"):
                        st.success(f"✅ Purchase Order drafted for {row['recommended_order_qty']} units")
    
    with tab4:
        st.subheader("📉 Slow-Moving Inventory Analysis")
        
        slow_movers_df = identify_slow_movers(sales_df, inventory_df)
        
        if len(slow_movers_df) > 0:
            st.warning(f"⚠️ Found {len(slow_movers_df)} slow-moving items")
            
            st.dataframe(slow_movers_df, use_container_width=True, height=400)
            
            st.markdown("### 💡 Recommended Actions:")
            for _, row in slow_movers_df.head(3).iterrows():
                st.info(f"**{row['product_name']}**: {row['recommended_action']}")
        else:
            st.success("✅ No slow-moving inventory detected!")

else:
    # Welcome screen
    st.title("📦 AI Inventory Optimization System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to Your AI-Powered Inventory Command Center
        
        **✅ Streamlit Cloud Optimized Version**
        
        This system helps you:
        
        ✅ **Predict Demand** - Smart forecasting for all SKUs  
        ✅ **Prevent Stockouts** - Early warning alerts  
        ✅ **Optimize Cash Flow** - Smart reorder recommendations  
        ✅ **Identify Slow Movers** - Clear excess inventory faster  
        ✅ **Fast Performance** - Optimized for cloud deployment
        
        ---
        
        ### Getting Started:
        
        1. Click **"Load Sample Data"** in the sidebar
        2. Explore the dashboard with demo data
        3. Upload your own CSV files when ready
        """)
    
    with col2:
        st.success("""
        ### 🚀 Cloud Ready
        
        **Fast Deployment**
        - Optimized dependencies
        - Quick loading
        - Reliable performance
        
        **Core Features**
        - Demand forecasting
        - Reorder management
        - Inventory analysis
        - Visual dashboards
        """)
        
        st.info("👈 Start by loading sample data!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>AI Inventory System v2.0 (Cloud Optimized) | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
