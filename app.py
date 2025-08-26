import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_loader import StockDataLoader
from feature_engineering import FeatureEngineer
from models import StockPredictionModels
from visualization import StockVisualizer

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .prediction-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Stock ticker input
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, GOOGL, MSFT)")
    
    # Time period selection
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    selected_period = st.sidebar.selectbox("Time Period", list(period_options.keys()), index=3)
    period = period_options[selected_period]
    
    # Model selection
    st.sidebar.subheader("🤖 Models")
    use_linear_regression = st.sidebar.checkbox("Linear Regression", value=True)
    use_random_forest = st.sidebar.checkbox("Random Forest", value=True)
    use_lstm = st.sidebar.checkbox("LSTM", value=True)
    
    # Hyperparameter tuning
    st.sidebar.subheader("🔧 Advanced Options")
    enable_hyperparameter_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False)
    test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5) / 100
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        run_analysis(ticker, period, use_linear_regression, use_random_forest, use_lstm, 
                    enable_hyperparameter_tuning, test_size)
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 About
    This dashboard uses machine learning to predict stock prices based on:
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Moving averages and volume analysis
    - Historical price patterns
    - Market sentiment indicators
    
    ### ⚠️ Disclaimer
    This is for educational purposes only. 
    Do not make investment decisions based on these predictions.
    """)
    
    # Main content area
    if 'analysis_complete' not in st.session_state:
        st.info("👈 Use the sidebar to configure and run your stock analysis!")
        
        # Sample data preview
        st.subheader("📊 Sample Data Preview")
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Open': [150.0, 151.2, 149.8, 152.1, 153.5, 152.8, 154.2, 155.0, 154.5, 156.1],
            'High': [152.5, 153.1, 151.9, 153.8, 154.9, 154.2, 155.8, 156.5, 155.9, 157.2],
            'Low': [149.5, 150.8, 148.9, 151.2, 152.8, 152.1, 153.5, 154.2, 153.8, 155.1],
            'Close': [151.2, 149.8, 152.1, 153.5, 152.8, 154.2, 155.0, 154.5, 156.1, 155.8],
            'Volume': [1000000, 1200000, 950000, 1100000, 1300000, 1150000, 1400000, 1250000, 1350000, 1200000]
        })
        st.dataframe(sample_data, use_container_width=True)

def run_analysis(ticker, period, use_linear_regression, use_random_forest, use_lstm, 
                enable_hyperparameter_tuning, test_size):
    """Run the complete stock analysis pipeline"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        status_text.text("🔄 Initializing components...")
        progress_bar.progress(10)
        
        data_loader = StockDataLoader()
        feature_engineer = FeatureEngineer()
        models = StockPredictionModels()
        visualizer = StockVisualizer()
        
        # Step 1: Load data
        status_text.text(f"📊 Fetching {ticker} data for {period}...")
        progress_bar.progress(20)
        
        stock_data = data_loader.fetch_stock_data(ticker, period=period)
        if stock_data is None:
            st.error(f"❌ Failed to fetch data for {ticker}")
            return
        
        # Step 2: Feature engineering
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(40)
        
        stock_data_features = feature_engineer.prepare_features(stock_data)
        if stock_data_features is None or stock_data_features.empty:
            st.error("❌ Feature engineering failed")
            return
        
        # Step 3: Model training
        status_text.text("🤖 Training models...")
        progress_bar.progress(60)
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = models.prepare_data(
            stock_data_features, test_size=test_size
        )
        
        # Train selected models
        trained_models = {}
        
        if use_linear_regression:
            lr_model = models.train_linear_regression(X_train, y_train)
            if lr_model:
                trained_models['linear_regression'] = lr_model
        
        if use_random_forest:
            rf_model = models.train_random_forest(X_train, y_train)
            if rf_model:
                trained_models['random_forest'] = rf_model
                
                # Hyperparameter tuning
                if enable_hyperparameter_tuning:
                    status_text.text("🔧 Tuning Random Forest hyperparameters...")
                    best_rf = models.hyperparameter_tuning_random_forest(X_train, y_train, cv=3)
                    if best_rf:
                        trained_models['random_forest_tuned'] = best_rf
        
        if use_lstm and X_lstm_train is not None and len(X_lstm_train) > 50:
            lstm_model, _ = models.train_lstm(X_lstm_train, y_lstm_train, epochs=20, batch_size=32)
            if lstm_model:
                trained_models['lstm'] = lstm_model
        
        # Step 4: Evaluation
        status_text.text("📊 Evaluating models...")
        progress_bar.progress(80)
        
        results = {}
        for model_name in trained_models.keys():
            if 'tuned' not in model_name:
                metrics, predictions = models.evaluate_model(model_name, X_test, y_test)
                if metrics:
                    results[model_name] = {
                        'metrics': metrics,
                        'predictions': predictions
                    }
        
        # Step 5: Display results
        status_text.text("🎨 Generating visualizations...")
        progress_bar.progress(90)
        
        display_results(ticker, stock_data_features, y_test, results, models)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        st.session_state.analysis_complete = True
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Analysis failed")

def display_results(ticker, stock_data_features, y_test, results, models):
    """Display analysis results and visualizations"""
    
    if not results:
        st.warning("⚠️ No models were successfully trained")
        return
    
    # Results overview
    st.header(f"📊 Analysis Results for {ticker}")
    
    # Model performance metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏆 Model Performance")
        
        # Create metrics dataframe
        metrics_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'RMSE': f"${metrics['RMSE']:.2f}",
                'MAE': f"${metrics['MAE']:.2f}",
                'R²': f"{metrics['R²']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.subheader("💰 Current Stock Info")
        current_price = stock_data_features['Close'].iloc[-1]
        current_volume = stock_data_features['Volume'].iloc[-1]
        
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Current Volume", f"{current_volume:,.0f}")
        
        # Price change
        if len(stock_data_features) > 1:
            prev_price = stock_data_features['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:+.2f}%")
    
    # Visualizations
    st.header("📈 Data Visualizations")
    
    # Stock price trends
    st.subheader("📊 Stock Price Trends")
    fig_trends = visualizer.create_interactive_plot(stock_data_features, ticker)
    if fig_trends:
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # Technical indicators
    st.subheader("🔧 Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI
        if 'RSI' in stock_data_features.columns:
            fig_rsi = px.line(stock_data_features, x='Date', y='RSI', title='RSI (Relative Strength Index)')
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        # MACD
        if 'MACD' in stock_data_features.columns:
            fig_macd = px.line(stock_data_features, x='Date', y=['MACD', 'MACD_signal'], 
                              title='MACD', color_discrete_map={'MACD': 'blue', 'MACD_signal': 'red'})
            st.plotly_chart(fig_macd, use_container_width=True)
    
    # Model predictions
    st.header("🔮 Model Predictions")
    
    if results:
        # Create predictions plot
        fig_predictions = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{name.replace('_', ' ').title()}" for name in list(results.keys())[:4]],
            vertical_spacing=0.1
        )
        
        test_dates = stock_data_features['Date'].iloc[-len(y_test):]
        
        for i, (model_name, pred_data) in enumerate(results.items()):
            if i >= 4:
                break
                
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            predictions = pred_data['predictions']
            
            fig_predictions.add_trace(
                go.Scatter(x=test_dates, y=y_test, name='Actual', line=dict(color='blue')),
                row=row, col=col
            )
            
            fig_predictions.add_trace(
                go.Scatter(x=test_dates, y=predictions, name='Predicted', line=dict(color='red')),
                row=row, col=col
            )
        
        fig_predictions.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_predictions, use_container_width=True)
    
    # Feature importance
    if 'random_forest' in models.models:
        st.header("🔍 Feature Importance")
        
        importance_df = models.get_feature_importance('random_forest')
        if importance_df is not None:
            top_features = importance_df.head(15)
            
            fig_importance = px.bar(
                top_features, 
                x='importance', 
                y='feature', 
                orientation='h',
                title='Top 15 Feature Importance (Random Forest)'
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
    
    # Future predictions
    st.header("🔮 Future Price Prediction")
    
    if results:
        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
        best_model = models.models[best_model_name]
        
        st.info(f"🎯 Using {best_model_name.replace('_', ' ').title()} for predictions (lowest RMSE)")
        
        # Make prediction
        try:
            recent_data = stock_data_features.iloc[-1:][models.feature_columns].values
            
            if best_model_name == 'lstm':
                # For LSTM, we need a sequence
                if hasattr(models, 'X_lstm_test') and models.X_lstm_test is not None and len(models.X_lstm_test) > 0:
                    last_sequence = models.X_lstm_test[-1:]
                    prediction = best_model.predict(last_sequence)[0]
                else:
                    st.warning("⚠️ Cannot make LSTM prediction without sequence data")
                    prediction = None
            else:
                # For other models
                prediction = best_model.predict(recent_data)[0]
            
            if prediction is not None:
                current_price = stock_data_features['Close'].iloc[-1]
                price_change = prediction - current_price
                price_change_pct = (price_change / current_price) * 100
                
                # Display prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    st.metric("Predicted Price", f"${prediction:.2f}")
                
                with col3:
                    st.metric("Predicted Change", f"${price_change:.2f}", f"{price_change_pct:+.2f}%")
                
                # Prediction sentiment
                if price_change > 0:
                    st.success("📈 Bullish prediction - Stock price expected to increase")
                else:
                    st.error("📉 Bearish prediction - Stock price expected to decrease")
                
                # Confidence indicator
                best_metrics = results[best_model_name]['metrics']
                r2_score = best_metrics['R²']
                
                if r2_score > 0.8:
                    confidence = "High"
                    confidence_color = "success"
                elif r2_score > 0.6:
                    confidence = "Medium"
                    confidence_color = "warning"
                else:
                    confidence = "Low"
                    confidence_color = "error"
                
                st.info(f"🎯 Model Confidence: {confidence} (R² = {r2_score:.4f})")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    # Download section
    st.header("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download features data
        csv_features = stock_data_features.to_csv(index=False)
        st.download_button(
            label="📥 Download Features Data (CSV)",
            data=csv_features,
            file_name=f"{ticker}_features.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download model results
        if results:
            results_data = []
            for model_name, result in results.items():
                metrics = result['metrics']
                metrics['model'] = model_name
                results_data.append(metrics)
            
            results_df = pd.DataFrame(results_data)
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Model Results (CSV)",
                data=csv_results,
                file_name=f"{ticker}_model_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
