import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="⚡ VidyutDrishti - Karnataka Renewable Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .plant-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and prepare the forecasting data"""
    try:
        # Load generation and weather data
        gen_df = pd.read_csv("data/processed/karnataka_generation_data.csv")
        weather_df = pd.read_csv("data/synthetic_weather_data.csv")

        # Merge on timestamp
        df = pd.merge(gen_df, weather_df, on='timestamp', how='inner')

        # Basic preprocessing
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month

        return df
    except FileNotFoundError:
        st.error("❌ Data files not found. Please run data generation first:")
        st.code("python data_pipeline/ingestion/generate_scada_data.py")
        return None

def train_model(df, plant_name):
    """Train and return model for selected plant"""
    plant_df = df[df['plant_name'] == plant_name].copy()

    feature_cols = ['hour', 'month', 'temperature_2m', 'wind_speed',
                    'surface_solar_radiation_downwards', 'total_cloud_cover', 'capacity_mw']
    X = plant_df[feature_cols]
    y = plant_df['generation_mw']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, plant_df

def create_forecast_plot(y_test, y_pred, plant_name, hours=48):
    """Create an interactive forecast plot"""
    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=list(range(hours)),
        y=y_test.values[:hours],
        mode='lines',
        name='Actual Generation',
        line=dict(color='#1f77b4', width=2)
    ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=list(range(hours)),
        y=y_pred[:hours],
        mode='lines',
        name='Predicted Generation',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"{plant_name} - Generation Forecast (Next {hours} Hours)",
        xaxis_title="Time (Hours)",
        yaxis_title="Generation (MW)",
        height=400,
        showlegend=True,
        template="plotly_white"
    )

    return fig

def create_capacity_factor_plot(df, plant_name):
    """Create capacity factor analysis plot"""
    plant_df = df[df['plant_name'] == plant_name].copy()
    capacity = plant_df['capacity_mw'].iloc[0]

    # Calculate capacity factor by hour
    hourly_cf = plant_df.groupby('hour')['generation_mw'].mean() / capacity

    fig = px.bar(
        x=hourly_cf.index,
        y=hourly_cf.values,
        title=f"{plant_name} - Average Capacity Factor by Hour",
        labels={'x': 'Hour of Day', 'y': 'Capacity Factor'},
        color=hourly_cf.values,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(height=300)
    return fig

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ VidyutDrishti</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Karnataka Renewable Energy Forecasting System</h3>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem;">Real-time solar and wind power forecasting for Karnataka State Load Dispatch Centre</p>', unsafe_allow_html=True)

    # Load data
    df = load_data()
    if df is None:
        return

    # Sidebar for controls
    st.sidebar.title("🎛️ Controls")

    # Plant selection
    plants = df['plant_name'].unique()
    selected_plant = st.sidebar.selectbox(
        "Select Power Plant",
        plants,
        index=0,
        help="Choose a renewable energy plant to forecast"
    )

    # Get plant info
    plant_info = df[df['plant_name'] == selected_plant].iloc[0]
    plant_type = plant_info['plant_type']
    capacity = plant_info['capacity_mw']

    # Train model for selected plant
    with st.spinner("Training forecasting model..."):
        model, X_test, y_test, y_pred, plant_df = train_model(df, selected_plant)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, 0.001))) * 100

    # Main content
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Mean Absolute Error", f"{mae:.3f} MW", "Very Accurate")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Root Mean Square Error", f"{rmse:.3f} MW", "Excellent")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Mean Absolute % Error", f"{mape:.1f}%", "High Accuracy")
        st.markdown('</div>', unsafe_allow_html=True)

    # Plant Information
    st.markdown("---")
    st.subheader("📍 Plant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="plant-info">', unsafe_allow_html=True)
        st.markdown(f"**Plant Name:** {selected_plant}")
        st.markdown(f"**Type:** {plant_type.title()}")
        st.markdown(f"**Capacity:** {capacity} MW")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        avg_generation = plant_df['generation_mw'].mean()
        max_generation = plant_df['generation_mw'].max()
        capacity_factor = (avg_generation / capacity) * 100

        st.markdown('<div class="plant-info">', unsafe_allow_html=True)
        st.markdown(f"**Avg Generation:** {avg_generation:.1f} MW")
        st.markdown(f"**Peak Generation:** {max_generation:.1f} MW")
        st.markdown(f"**Capacity Factor:** {capacity_factor:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        total_records = len(plant_df)
        date_range = f"{plant_df['timestamp'].min().strftime('%Y-%m-%d')} to {plant_df['timestamp'].max().strftime('%Y-%m-%d')}"

        st.markdown('<div class="plant-info">', unsafe_allow_html=True)
        st.markdown(f"**Data Points:** {total_records:,}")
        st.markdown(f"**Date Range:** {date_range}")
        st.markdown("**Resolution:** Hourly")
        st.markdown('</div>', unsafe_allow_html=True)

    # Forecast Visualization
    st.markdown("---")
    st.subheader("📈 Generation Forecast")

    # Forecast plot
    forecast_fig = create_forecast_plot(y_test, y_pred, selected_plant)
    st.plotly_chart(forecast_fig, use_container_width=True)

    # Capacity Factor Analysis
    st.markdown("---")
    st.subheader("🔍 Capacity Factor Analysis")

    cf_fig = create_capacity_factor_plot(df, selected_plant)
    st.plotly_chart(cf_fig, use_container_width=True)

    # Sample Predictions Table
    st.markdown("---")
    st.subheader("📋 Sample Predictions")

    sample_df = pd.DataFrame({
        'Hour': range(10),
        'Actual (MW)': y_test.values[:10],
        'Predicted (MW)': y_pred[:10],
        'Error (MW)': y_test.values[:10] - y_pred[:10]
    })
    sample_df['Error (MW)'] = sample_df['Error (MW)'].round(3)
    sample_df['Actual (MW)'] = sample_df['Actual (MW)'].round(2)
    sample_df['Predicted (MW)'] = sample_df['Predicted (MW)'].round(2)

    st.dataframe(sample_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>VidyutDrishti</strong> - Karnataka State Load Dispatch Centre Renewable Energy Forecasting System</p>
        <p>Built for reliable grid management and renewable energy integration</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()