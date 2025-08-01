import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose

# Set Streamlit page config
st.set_page_config(
    page_title="EV Forecast Pro",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

# === Dark Theme CSS Styling ===
st.markdown("""
    <style>
        :root {
            --primary-color: #3b82f6;
            --secondary-color: #1e40af;
            --accent-color: #60a5fa;
            --background-color: #111827;
            --card-color: #1f2937;
            --text-color: #f3f4f6;
            --border-color: #374151;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text-color);
        }
        
        .stApp {
            background-color: var(--background-color);
        }
        
        .header-title {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            color: var(--accent-color) !important;
            margin-bottom: 0.5rem !important;
            text-align: center;
        }
        
        .header-subtitle {
            font-size: 1.2rem !important;
            color: #9ca3af !important;
            text-align: center;
            margin-bottom: 2rem !important;
        }
        
        .card {
            background-color: var(--card-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            color: white !important;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
            border: 1px solid #3b82f6;
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem;
            margin-top: 3rem;
            color: #6b7280;
            font-size: 0.9rem;
            border-top: 1px solid var(--border-color);
        }
    </style>
""", unsafe_allow_html=True)

# === Load Data and Model ===
@st.cache_resource
def load_model():
    try:
        return joblib.load('forecasting_ev_model.pkl')
    except:
        st.error("Model file not found. Please ensure 'forecasting_ev_model.pkl' is in the directory.")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("preprocessed_ev_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        st.error("Data file not found. Please ensure 'preprocessed_ev_data.csv' is in the directory.")
        st.stop()

model = load_model()
df = load_data()

# === Helper Functions ===
def generate_forecast(county_df, forecast_horizon=36):
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()
    county_code = county_df['county_encoded'].iloc[0]

    future_rows = []
    
    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({
            "Date": forecast_date, 
            "Predicted EV Total": round(pred),
            "Month": forecast_date.month,
            "Year": forecast_date.year
        })

        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)

        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

    forecast_df = pd.DataFrame(future_rows)
    return forecast_df

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

# === Header Section ===
st.markdown("""
    <div class="header-title">
        🚗 EV Adoption Forecaster Pro
    </div>
    <div class="header-subtitle">
        Advanced forecasting tool for Electric Vehicle adoption trends in Washington State
    </div>
""", unsafe_allow_html=True)

# Hero image with dark overlay
st.markdown("""
    <div style="position: relative; margin-bottom: 2rem;">
        <img src="https://images.unsplash.com/photo-1605559424843-9e4c228bf1c2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2000&q=80" 
             style="width: 100%; height: auto; border-radius: 8px; filter: brightness(0.7);">
        <div style="position: absolute; bottom: 20px; left: 20px; color: white;">
            <h3 style="margin: 0; font-size: 1.5rem;">The Future of Transportation is Electric</h3>
            <p style="margin: 0; opacity: 0.8;">Advanced analytics for EV adoption forecasting</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Flag_of_Washington.svg/1200px-Flag_of_Washington.svg.png",
                width=100,
                caption="Washington State")
    except:
        st.warning("Couldn't load state image")
    
    st.markdown("## Settings")
    analysis_type = st.radio(
        "Analysis Type",
        ["Single County Forecast", "Multi-County Comparison", "Growth Rate Analysis"],
        index=0
    )
    

# === Main Content ===
if analysis_type == "Single County Forecast":
    st.markdown("## Single County Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        county_list = sorted(df['County'].dropna().unique().tolist())
        selected_county = st.selectbox("Select a County", county_list, index=county_list.index('King') if 'King' in county_list else 0)
        
        forecast_horizon = st.slider(
            "Forecast Horizon (months)",
            min_value=12,
            max_value=60,
            value=36,
            step=12
        )
        
        show_advanced = st.checkbox("Show Advanced Options")
        
        if show_advanced:
            show_metrics = st.checkbox("Show Model Metrics", value=True)
            show_seasonality = st.checkbox("Show Seasonality Analysis", value=True)
            show_monthly = st.checkbox("Show Monthly Breakdown", value=True)
        else:
            show_metrics = False
            show_seasonality = False
            show_monthly = False
    
    county_df = df[df['County'] == selected_county].sort_values("Date")
    
    if county_df.empty:
        st.warning(f"No data available for {selected_county} county.")
        st.stop()
    
    # Generate forecast
    forecast_df = generate_forecast(county_df, forecast_horizon)
    
    # Combine historical and forecast data
    historical_df = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    historical_df['Type'] = 'Historical'
    historical_df.rename(columns={'Electric Vehicle (EV) Total': 'EV Count'}, inplace=True)
    
    forecast_plot_df = forecast_df[['Date', 'Predicted EV Total']].copy()
    forecast_plot_df['Type'] = 'Forecast'
    forecast_plot_df.rename(columns={'Predicted EV Total': 'EV Count'}, inplace=True)
    
    combined_plot_df = pd.concat([historical_df, forecast_plot_df])
    
    # Calculate cumulative values
    historical_cum = historical_df.copy()
    historical_cum['Cumulative EV'] = historical_cum['EV Count'].cumsum()
    
    forecast_cum = forecast_plot_df.copy()
    forecast_cum['Cumulative EV'] = forecast_cum['EV Count'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]
    
    combined_cum_df = pd.concat([historical_cum, forecast_cum])
    
    # Metrics
    last_year_historical = historical_df[historical_df['Date'] >= (historical_df['Date'].max() - pd.DateOffset(years=1))]
    if len(last_year_historical) >= 12 and show_metrics:
        test_dates = last_year_historical['Date'].values
        test_actual = last_year_historical['EV Count'].values
        
        test_predictions = []
        temp_hist = list(historical_df[historical_df['Date'] < test_dates[0]]['EV Count'].values[-6:])
        temp_cum = list(np.cumsum(temp_hist))
        temp_months = county_df[county_df['Date'] < test_dates[0]]['months_since_start'].max()
        
        for i in range(len(test_dates)):
            months_since_start = temp_months + i + 1
            lag1, lag2, lag3 = temp_hist[-1], temp_hist[-2], temp_hist[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            recent_cum = temp_cum[-6:]
            ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0
            
            new_row = {
                'months_since_start': months_since_start,
                'county_encoded': county_df['county_encoded'].iloc[0],
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_slope
            }
            
            pred = model.predict(pd.DataFrame([new_row]))[0]
            test_predictions.append(pred)
            
            temp_hist.append(test_actual[i])
            if len(temp_hist) > 6:
                temp_hist.pop(0)
                
            temp_cum.append(temp_cum[-1] + test_actual[i])
            if len(temp_cum) > 6:
                temp_cum.pop(0)
        
        mae, rmse, mape = calculate_metrics(np.array(test_actual), np.array(test_predictions))
    
    # Display metrics in cards
    if show_metrics and len(last_year_historical) >= 12:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>MAE</h3>
                    <h1>{mae:.1f}</h1>
                    <p>Mean Absolute Error</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>RMSE</h3>
                    <h1>{rmse:.1f}</h1>
                    <p>Root Mean Squared Error</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>MAPE</h3>
                    <h1>{mape:.1f}%</h1>
                    <p>Mean Absolute Percentage Error</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["EV Adoption Trend", "Cumulative Forecast", "Detailed Analysis"])
    
    with tab1:
        st.markdown(f"### Monthly EV Adoption Trend - {selected_county}")
        
        fig = px.line(
            combined_plot_df,
            x='Date',
            y='EV Count',
            color='Type',
            line_dash='Type',
            markers=True,
            labels={'EV Count': 'Number of EVs', 'Date': ''},
            color_discrete_map={'Historical': '#3b82f6', 'Forecast': '#ef4444'},
            template='plotly_dark'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title=None,
            yaxis_title="Number of EVs",
            legend_title="Data Type",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth metrics
        hist_total = historical_df['EV Count'].sum()
        forecast_total = forecast_plot_df['EV Count'].sum()
        growth_pct = (forecast_total / (hist_total / (len(historical_df) / len(forecast_plot_df))) - 1) * 100
        
        st.markdown(f"""
            <div class="card">
                <h3>Growth Summary</h3>
                <p>Historical monthly average: <strong>{hist_total/len(historical_df):.1f} EVs</strong></p>
                <p>Forecasted monthly average: <strong>{forecast_total/len(forecast_plot_df):.1f} EVs</strong></p>
                <p>Projected growth rate: <strong>{growth_pct:.1f}%</strong> over forecast period</p>
            </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f"### Cumulative EV Forecast - {selected_county}")
        
        fig = px.line(
            combined_cum_df,
            x='Date',
            y='Cumulative EV',
            color='Type',
            line_dash='Type',
            markers=True,
            labels={'Cumulative EV': 'Total EVs', 'Date': ''},
            color_discrete_map={'Historical': '#3b82f6', 'Forecast': '#ef4444'},
            template='plotly_dark'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title=None,
            yaxis_title="Cumulative EV Count",
            legend_title="Data Type",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate growth metrics
        historical_cumulative = historical_cum['Cumulative EV'].iloc[-1]
        forecasted_cumulative = forecast_cum['Cumulative EV'].iloc[-1]
        forecast_growth_pct = ((forecasted_cumulative - historical_cumulative) / historical_cumulative) * 100
        
        st.markdown(f"""
            <div class="card">
                <h3>Cumulative Growth</h3>
                <p>Historical cumulative EVs: <strong>{historical_cumulative:,.0f}</strong></p>
                <p>Forecasted additional EVs: <strong>{forecasted_cumulative - historical_cumulative:,.0f}</strong></p>
                <p>Projected growth: <strong>{forecast_growth_pct:.1f}%</strong> over {forecast_horizon} months</p>
            </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        if show_seasonality or show_monthly:
            cols = st.columns(2 if show_seasonality and show_monthly else 1)
            col_idx = 0
            
            if show_seasonality:
                with cols[col_idx]:
                    st.markdown("### Seasonality Analysis")
                    
                    ts_data = historical_df.set_index('Date')['EV Count']
                    if len(ts_data) >= 24:
                        result = seasonal_decompose(ts_data, model='additive', period=12)
                        
                        fig = make_subplots(
                            rows=4, cols=1,
                            shared_xaxes=True,
                            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual')
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=result.observed.index, y=result.observed, name='Observed'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=result.trend.index, y=result.trend, name='Trend'),
                            row=2, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=result.seasonal.index, y=result.seasonal, name='Seasonal'),
                            row=3, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=result.resid.index, y=result.resid, name='Residual'),
                            row=4, col=1
                        )
                        
                        fig.update_layout(
                            height=800, 
                            showlegend=False,
                            template='plotly_dark',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data (minimum 2 years required) for seasonality analysis")
                
                col_idx += 1
            
            if show_monthly:
                with cols[col_idx]:
                    st.markdown("### Monthly Breakdown")
                    
                    monthly_data = pd.concat([
                        historical_df.assign(Period='Historical'),
                        forecast_plot_df.assign(Period='Forecast')
                    ]).copy()
                    
                    monthly_data['Month'] = monthly_data['Date'].dt.month.apply(lambda x: calendar.month_abbr[x])
                    monthly_data['Year'] = monthly_data['Date'].dt.year
                    
                    monthly_data = monthly_data.dropna(subset=['Month', 'Year', 'EV Count'])
                    
                    if not monthly_data.empty:
                        try:
                            heatmap_data = monthly_data.pivot_table(
                                index='Month',
                                columns='Year',
                                values='EV Count',
                                aggfunc='mean'
                            )
                            
                            month_order = [calendar.month_abbr[i] for i in range(1, 13)]
                            heatmap_data = heatmap_data.reindex(month_order)
                            
                            fig = px.imshow(
                                heatmap_data,
                                labels=dict(x="Year", y="Month", color="EV Count"),
                                aspect="auto",
                                color_continuous_scale='Viridis',
                                template='plotly_dark'
                            )
                            
                            fig.update_layout(
                                title="Monthly EV Adoption Heatmap",
                                xaxis_title="Year",
                                yaxis_title="Month"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not create heatmap: {str(e)}")
                    else:
                        st.warning("No data available for monthly breakdown")
    
        st.markdown("### Data Preview")
        st.dataframe(combined_plot_df.sort_values('Date', ascending=False).head(10))

elif analysis_type == "Multi-County Comparison":
    st.markdown("## Multi-County Comparison")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        county_list = sorted(df['County'].dropna().unique().tolist())
        selected_counties = st.multiselect(
            "Select Counties to Compare",
            county_list,
            default=['King', 'Pierce', 'Snohomish'] if all(c in county_list for c in ['King', 'Pierce', 'Snohomish']) else county_list[:3],
            max_selections=5
        )
        
        comparison_metric = st.radio(
            "Comparison Metric",
            ["Cumulative Count", "Monthly Adoption", "Growth Rate"],
            index=0
        )
        
        forecast_horizon = st.slider(
            "Forecast Horizon (months)",
            min_value=12,
            max_value=60,
            value=36,
            step=12
        )
    
    if not selected_counties:
        st.warning("Please select at least one county")
        st.stop()
    
    comparison_data = []
    growth_rates = []
    
    for county in selected_counties:
        county_df = df[df['County'] == county].sort_values("Date")
        
        if county_df.empty:
            st.warning(f"No data available for {county} county.")
            continue
        
        forecast_df = generate_forecast(county_df, forecast_horizon)
        
        historical_df = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        historical_df['Type'] = 'Historical'
        historical_df.rename(columns={'Electric Vehicle (EV) Total': 'EV Count'}, inplace=True)
        
        forecast_plot_df = forecast_df[['Date', 'Predicted EV Total']].copy()
        forecast_plot_df['Type'] = 'Forecast'
        forecast_plot_df.rename(columns={'Predicted EV Total': 'EV Count'}, inplace=True)
        
        combined_df = pd.concat([historical_df, forecast_plot_df])
        combined_df['County'] = county
        
        combined_df['Cumulative EV'] = combined_df.groupby('County')['EV Count'].cumsum()
        combined_df['Growth Rate'] = combined_df.groupby('County')['EV Count'].pct_change() * 100
        
        comparison_data.append(combined_df)
        
        hist_total = historical_df['EV Count'].sum()
        forecast_total = forecast_plot_df['EV Count'].sum()
        growth_pct = (forecast_total / (hist_total / (len(historical_df) / len(forecast_plot_df))) - 1) * 100
        growth_rates.append(growth_pct)
    
    if not comparison_data:
        st.error("No valid data available for comparison")
        st.stop()
    
    comparison_df = pd.concat(comparison_data)
    
    if comparison_metric == "Cumulative Count":
        st.markdown("### Cumulative EV Adoption Comparison")
        
        fig = px.line(
            comparison_df,
            x='Date',
            y='Cumulative EV',
            color='County',
            line_dash='Type',
            labels={'Cumulative EV': 'Total EVs', 'Date': ''},
            hover_data=['EV Count'],
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template='plotly_dark'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title=None,
            yaxis_title="Cumulative EV Count",
            legend_title="County",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif comparison_metric == "Monthly Adoption":
        st.markdown("### Monthly EV Adoption Comparison")
        
        fig = px.line(
            comparison_df,
            x='Date',
            y='EV Count',
            color='County',
            line_dash='Type',
            labels={'EV Count': 'Monthly EVs', 'Date': ''},
            hover_data=['Cumulative EV'],
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template='plotly_dark'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title=None,
            yaxis_title="Monthly EV Count",
            legend_title="County",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.markdown("### Growth Rate Comparison (%)")
        
        fig = px.line(
            comparison_df[comparison_df['Growth Rate'].notna()],
            x='Date',
            y='Growth Rate',
            color='County',
            line_dash='Type',
            labels={'Growth Rate': 'Growth Rate (%)', 'Date': ''},
            hover_data=['EV Count', 'Cumulative EV'],
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template='plotly_dark'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title=None,
            yaxis_title="Growth Rate (%)",
            legend_title="County",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Comparison Summary")
    
    summary_data = []
    for i, county in enumerate(selected_counties):
        county_data = comparison_df[comparison_df['County'] == county]
        historical = county_data[county_data['Type'] == 'Historical']
        forecast = county_data[county_data['Type'] == 'Forecast']
        
        summary_data.append({
            'County': county,
            'Historical EVs': historical['EV Count'].sum(),
            'Forecasted EVs': forecast['EV Count'].sum(),
            'Growth Rate (%)': growth_rates[i],
            'Current Monthly Avg': historical['EV Count'].mean(),
            'Projected Monthly Avg': forecast['EV Count'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(
        summary_df.style
        .background_gradient(subset=['Growth Rate (%)'], cmap='RdYlGn')
        .format({
            'Historical EVs': '{:,.0f}',
            'Forecasted EVs': '{:,.0f}',
            'Growth Rate (%)': '{:.1f}%',
            'Current Monthly Avg': '{:.1f}',
            'Projected Monthly Avg': '{:.1f}'
        }),
        use_container_width=True
    )
    
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="Download Comparison Data",
        data=csv,
        file_name='ev_comparison_data.csv',
        mime='text/csv'
    )

else:
    st.markdown("## Growth Rate Analysis")
    
    county_list = sorted(df['County'].dropna().unique().tolist())
    selected_counties = st.multiselect(
        "Select Counties to Compare",
        county_list,
        default=['Adams', 'Albemarle'] if all(c in county_list for c in ['Adams', 'Albemarle']) else county_list[:2],
        max_selections=5
    )
    
    forecast_horizon = st.slider(
        "Forecast Horizon (months)",
        min_value=12,
        max_value=60,
        value=36,
        step=12
    )
    
    if not selected_counties:
        st.warning("Please select at least one county")
        st.stop()
    
    growth_data = []
    
    for county in selected_counties:
        county_df = df[df['County'] == county].sort_values("Date")
        
        if county_df.empty:
            st.warning(f"No data available for {county} county.")
            continue
        
        forecast_df = generate_forecast(county_df, forecast_horizon)
        
        hist_avg = county_df['Electric Vehicle (EV) Total'].mean()
        forecast_avg = forecast_df['Predicted EV Total'].mean()
        growth_pct = ((forecast_avg - hist_avg) / hist_avg) * 100 if hist_avg != 0 else 0
        
        hist_total = county_df['Electric Vehicle (EV) Total'].sum()
        forecast_total = forecast_df['Predicted EV Total'].sum()
        cumulative_growth = ((forecast_total - hist_total) / hist_total) * 100 if hist_total != 0 else 0
        
        growth_data.append({
            'County': county,
            'Growth Percentage': growth_pct,
            'Cumulative Growth': cumulative_growth,
            'Historical Average': hist_avg,
            'Forecasted Average': forecast_avg
        })
    
    if not growth_data:
        st.error("No valid data available for growth rate analysis")
        st.stop()
    
    st.markdown("### Projected Growth Percentage by County")
    
    growth_df = pd.DataFrame(growth_data)
    
    fig = px.bar(
        growth_df,
        x='County',
        y='Growth Percentage',
        color='County',
        text='Growth Percentage',
        labels={'Growth Percentage': 'Growth Percentage (%)', 'County': ''},
        template='plotly_dark'
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=None,
        yaxis_title="Growth Percentage (%)",
        showlegend=False,
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Detailed Growth Metrics")
    
    display_df = growth_df.copy()
    
    styled_df = (display_df.style
                .background_gradient(subset=['Growth Percentage', 'Cumulative Growth'], cmap='RdYlGn')
                .format({
                    'Growth Percentage': '{:.1f}%',
                    'Cumulative Growth': '{:.1f}%',
                    'Historical Average': '{:.1f}',
                    'Forecasted Average': '{:.1f}'
                }))
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("""
        <div class="card">
            <h3>Growth Rate Explanation</h3>
            <p><strong>Growth Percentage:</strong> The projected percentage increase in monthly EV adoption compared to historical averages.</p>
            <p><strong>Cumulative Growth:</strong> The total percentage increase in EVs over the forecast period compared to historical totals.</p>
        </div>
    """, unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>Developed for AICTE Internship Cycle 2 | Powered by Streamlit</p>
        <p>Data Source: Washington State Department of Licensing | Last Updated: {}</p>
    </div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)