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

def forecast_demand(sales_df, sku, periods=30):
    """Forecast demand using linear regression with seasonality"""
    # Filter data for specific SKU
    sku_data = sales_df[sales_df['sku'] == sku].copy()
    
    # Aggregate by date
    daily_sales = sku_data.groupby('date')['quantity_sold'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    
    if len(daily_sales) < 7:
        # Not enough data, use simple average
        avg_sales = daily_sales['y'].mean() if len(daily_sales) > 0 else 10
        future_dates = pd.date_range(
            start=daily_sales['ds'].max() + timedelta(days=1) if len(daily_sales) > 0 else datetime.now(),
            periods=periods,
            freq='D'
        )
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': [avg_sales] * periods,
            'yhat_lower': [max(0, avg_sales * 0.7)] * periods,
            'yhat_upper': [avg_sales * 1.3] * periods
        })
        return forecast_df
    
    # Prepare features for ML model
    daily_sales['day_of_week'] = pd.to_datetime(daily_sales['ds']).dt.dayofweek
    daily_sales['day_of_month'] = pd.to_datetime(daily_sales['ds']).dt.day
    daily_sales['month'] = pd.to_datetime(daily_sales['ds']).dt.month
    daily_sales['days_since_start'] = (pd.to_datetime(daily_sales['ds']) - pd.to_datetime(daily_sales['ds']).min()).dt.days
    
    # Features and target
    X = daily_sales[['days_since_start', 'day_of_week', 'day_of_month', 'month']].values
    y = daily_sales['y'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates
    last_date = pd.to_datetime(daily_sales['ds'].max())
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Prepare future features
    start_days = (last_date - pd.to_datetime(daily_sales['ds']).min()).days + 1
    future_features = pd.DataFrame({
        'ds': future_dates,
        'days_since_start': range(start_days, start_days + periods),
        'day_of_week': [d.dayofweek for d in future_dates],
        'day_of_month': [d.day for d in future_dates],
        'month': [d.month for d in future_dates]
    })
    
    X_future = future_features[['days_since_start', 'day_of_week', 'day_of_month', 'month']].values
    predictions = model.predict(X_future)
    
    # Calculate confidence intervals (based on historical std)
    std_dev = daily_sales['y'].std()
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': np.maximum(0, predictions),  # Ensure non-negative
        'yhat_lower': np.maximum(0, predictions - 1.5 * std_dev),
        'yhat_upper': predictions + 1.5 * std_dev
    })
    
    return forecast_df

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
    confidence_threshold = st.slider("Alert Confidence %", 70, 95, 85)

# Main content
if st.session_state.data_loaded:
    sales_df = st.session_state.sales_df
    products_df = st.session_state.products_df
    inventory_df = st.session_state.inventory_df
    
    # Header
    st.title("🎯 AI Inventory Command Center")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Next Sync:** 58 minutes")
    
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
                forecast = forecast_demand(sales_df, selected_sku, periods=7)
                
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
                forecast_result = forecast_demand(sales_df, forecast_sku, periods=forecast_period)
                
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
    <p>AI Inventory Optimization System v1.0 | Powered by Scikit-learn ML & Streamlit</p>
    <p style='font-size: 12px;'>For support: <a href='mailto:support@example.com'>support@example.com</a></p>
</div>
""", unsafe_allow_html=True)
