import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Smart Finance Analyzer",
    page_icon="images/icon.png",
    layout="wide"
)

@st.cache_resource
def load_models():
    categorizer = joblib.load('models/categorizer_v2_multifeature.pkl')
    category_stats = joblib.load('models/anomaly_category_stats.pkl')
    threshold = joblib.load('models/anomaly_threshold.pkl')
    forecasts = joblib.load('models/forecasts.pkl')
    return categorizer, category_stats, threshold, forecasts

categorizer, category_stats, threshold, forecasts = load_models()

col1, col2 = st.columns([1,8])

with col1:
    st.image("images/icon.png", width=60)

with col2:
    st.title("Smart Finance Analyzer")

st.markdown("Upload your transactions to get spending insights, auto-categorization, anomaly alerts, and next month's forecast.")
st.divider()

uploaded_file = st.file_uploader(
    "Upload your transactions CSV or Excel file",
    type=['csv', 'xlsx']
)

## loading the dataset

if uploaded_file is not None:

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df['date'] = pd.to_datetime(df['date'])
    df['clean_description'] = df['description'].str.lower().str.strip()

    ## categorizing the new data

    input_df = df[['clean_description', 'amount', 'payment_mode']].copy()
    input_df = input_df.rename(columns={'clean_description': 'description'})

    df['predicted_category'] = categorizer.predict(input_df)

    ## anomaly detection calculations

    df = df.merge(category_stats, left_on='predicted_category', right_on='category', how='left')
    df['z_score'] = (df['amount'] - df['mean_amount']) / df['std_amount']
    df['is_anomaly_detected'] = df['z_score'] > threshold

    anomaly_count = len(df[df['is_anomaly_detected']])

    st.success(f"Loaded {len(df)} transactions successfully")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", len(df))
    m2.metric("Total Spend", f"₹{df['amount'].sum():,.0f}")
    m3.metric("Avg per Transaction", f"₹{df['amount'].mean():,.0f}")
    m4.metric("Anomalies Detected", anomaly_count)

    ## categorization table

    st.header("Auto Categorization")

    st.dataframe(
        df[['date', 'description', 'amount', 'payment_mode', 'predicted_category']],
        use_container_width=True
    )

    ## spending breakdown

    st.header("Spending Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        cat_spend = df.groupby('predicted_category')['amount'].sum().reset_index()

        fig1 = px.bar(
            cat_spend.sort_values('amount'),
            x='amount',
            y='predicted_category',
            orientation='h',
            title='Total Spend by Category',
            labels={'amount': 'Amount (₹)', 'predicted_category': 'Category'}
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.pie(
            cat_spend,
            values='amount',
            names='predicted_category',
            title='Spending Distribution'
        )

        st.plotly_chart(fig2, use_container_width=True)

    ## monthly trend

    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    monthly = df.groupby('month')['amount'].sum().reset_index()

    fig3 = px.line(
        monthly,
        x='month',
        y='amount',
        title='Monthly Spending Trend',
        markers=True,
        labels={'amount': 'Amount (₹)', 'month': 'Month'}
    )

    st.plotly_chart(fig3, use_container_width=True)

    ## anomaly detection

    st.header("Anomaly Detection")

    anomalies = df[df['is_anomaly_detected'] == True]

    if len(anomalies) > 0:
        st.warning(f"{len(anomalies)} suspicious transactions detected")

        st.dataframe(
            anomalies[['date', 'description', 'predicted_category', 'amount', 'z_score']]
            .sort_values('z_score', ascending=False)
            .style.highlight_max(subset=['amount'], color='#ffcccc'),
            use_container_width=True
        )

    else:
        st.success("No anomalies detected")

    ## forecasting for next month

    st.header("Next Month Forecast")

    forecast_df = pd.DataFrame(forecasts).T.reset_index()
    forecast_df.columns = ['category', 'predicted', 'lower', 'upper']
    forecast_df['predicted'] = forecast_df['predicted'].clip(lower=0)
    forecast_df['lower'] = forecast_df['lower'].clip(lower=0)
    forecast_df = forecast_df.sort_values('predicted', ascending=False)

    fig4 = px.bar(
        forecast_df,
        x='predicted',
        y='category',
        orientation='h',
        error_x=forecast_df['upper'] - forecast_df['predicted'],
        title='Predicted Next Month Spending',
        labels={'predicted': 'Predicted Amount (₹)', 'category': 'Category'}
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.caption("Forecast based on 12 months of data. Wider error bars indicate higher uncertainty.")