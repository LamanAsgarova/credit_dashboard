import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from faker import Faker
import random
from datetime import datetime  # Add this import

# Page config and custom CSS
st.set_page_config(page_title="Credit Risk Analytics Dashboard", layout="wide", page_icon="💳")

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Create sample data
fake = Faker()

credit_risk_probs = [0.7, 0.3]  # 70% good, 30% bad
housing_options = ['own', 'rent', 'free']
housing_probs = [0.55, 0.25, 0.2]

job_levels = ['unskilled', 'skilled', 'highly skilled', 'management']
job_probs = [0.2, 0.5, 0.2, 0.1]

purpose_options = ['car', 'furniture', 'radio/TV', 'education', 'business', 'repairs']
purpose_probs = [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]

data = []

for _ in range(10000):
    row = {
        'status': random.choice(['A11', 'A12', 'A13', 'A14']),
        'duration': random.randint(6, 72),
        'credit_history': random.choice(['no credits/all paid', 'all paid', 'existing paid', 'delayed']),
        'purpose': random.choices(purpose_options, weights=purpose_probs)[0],
        'amount': random.randint(500, 20000),
        'savings': random.choice(['<100', '100<=X<500', '500<=X<1000', '>=1000', 'unknown']),
        'employment': random.choice(['<1', '1<=X<4', '4<=X<7', '>=7']),
        'installment_rate': random.randint(1, 4),
        'personal_status': random.choice(['male single', 'female div/dep/mar', 'male div/sep', 'male mar/wid']),
        'other_debtors': random.choice(['none', 'co-applicant', 'guarantor']),
        'residence_since': random.randint(1, 4),
        'property': random.choice(['real estate', 'savings', 'car', 'unknown']),
        'age': random.randint(18, 75),
        'other_installment_plans': random.choice(['bank', 'stores', 'none']),
        'housing': random.choices(housing_options, weights=housing_probs)[0],
        'existing_credits': random.randint(1, 4),
        'job': random.choices(job_levels, weights=job_probs)[0],
        'people_liable': random.randint(1, 2),
        'telephone': random.choice(['yes', 'no']),
        'foreign_worker': random.choice(['yes', 'no']),
        'credit_risk': random.choices(['good', 'bad'], weights=credit_risk_probs)[0]
    }
    data.append(row)
    
df = pd.DataFrame(data)

df['age_group'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 70, 100], labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71+'])

# Streamlit page setup
st.title("Credit Risk Analytics Dashboard")

# 3. Sidebar filters
def apply_filters(df):
    st.sidebar.header("🔧 Filter Options")
    
    age_min, age_max = st.sidebar.slider(
        "👥 Age Range:",
        min_value=int(df['age'].min()), 
        max_value=int(df['age'].max()), 
        value=(int(df['age'].min()), int(df['age'].max())),
        step=1
    )
    
    job_options = ['All'] + sorted(df['job'].unique().tolist())
    selected_job = st.sidebar.multiselect("💼 Job Type:", options=job_options, default=['All'])
    
    housing_options = ['All'] + sorted(df['housing'].unique().tolist())
    selected_housing = st.sidebar.multiselect("🏠 Housing Status:", options=housing_options, default=['All'])
    
    credit_risk_options = ['All', 'good', 'bad']
    selected_credit_risk = st.sidebar.selectbox("💳 Credit Risk:", options=credit_risk_options, index=0)
    
    # Apply filters
    filtered_df = df.copy()
    
    filtered_df = filtered_df[(filtered_df['age'] >= age_min) & (filtered_df['age'] <= age_max)]
    
    if 'All' not in selected_job:
        filtered_df = filtered_df[filtered_df['job'].isin(selected_job)]
    
    if 'All' not in selected_housing:
        filtered_df = filtered_df[filtered_df['housing'].isin(selected_housing)]
    
    if selected_credit_risk != 'All':
        filtered_df = filtered_df[filtered_df['credit_risk'] == selected_credit_risk]
    
    return filtered_df

# KPI calculation
def calculate_kpis(df):
    if len(df) == 0:
        return dict.fromkeys(['total_applicants', 'total_approved', 'avg_credit_amount', 'avg_age'], 0)
    
    kpis = {
        'total_applicants': len(df),
        'total_approved': df[df['credit_risk'] == 'good'].shape[0],
        'avg_credit_amount': df['amount'].mean(),
        'avg_age': df['age'].mean()
    }
    return kpis

# KPI display
def display_kpis(kpis):
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Applicants", f"{kpis['total_applicants']:,}")
    with col2:
        st.metric("✅ Total Approved", f"{kpis['total_approved']:,}")
    with col3:
        st.metric("💰 Avg Credit Amount", f"${kpis['avg_credit_amount']:,.2f}")
    with col4:
        st.metric("👥 Avg Age", f"{kpis['avg_age']:.0f} years")

# Visualization functions
def create_visualizations(df):
    st.markdown("---")
    st.markdown("### 📊 Data Visualizations")
    
    # Plot 1: Age Distribution
    fig = px.histogram(
        df,
        x="age",
        nbins=20,
        title="Age Distribution of Credit Applicants",
        labels={"age": "Age"},
        opacity=0.7
    )
    st.plotly_chart(fig)

    # Plot 2: Average Credit Amount by Job Type
    avg_amount_by_job = df.groupby('job')['amount'].mean().reset_index()
    fig = px.bar(
        avg_amount_by_job,
        x='job',
        y='amount',
        title="Average Credit Amount by Job Type",
        labels={'job': 'Job Type', 'amount': 'Average Credit Amount'},
        color='job'
    )
    st.plotly_chart(fig)

    # Plot 3: Average Credit Amount by Housing Status
    avg_amount_by_housing = df.groupby('housing')['amount'].mean().reset_index()
    fig = px.bar(
        avg_amount_by_housing,
        x='housing',
        y='amount',
        title='Average Credit Amount by Housing Status',
        labels={'housing': 'Housing Status', 'amount': 'Average Credit Amount'},
        color='housing'
    )
    fig.update_layout(title_x=0.5, showlegend=False)
    st.plotly_chart(fig)

    # Plot 4: Distribution of Applicant Age by Credit Purpose
    fig = px.box(
        df,
        x='purpose',
        y='age',
        title="Distribution of Applicant Age by Credit Purpose",
        labels={'purpose': 'Credit Purpose', 'age': 'Applicant Age'}
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig)

    # Plot 5: Credit Amount by Credit History and Employment Duration
    fig = px.box(
        df,
        x='credit_history',
        y='amount',
        color='employment',
        title="Credit Amount by Credit History and Employment Duration",
        labels={'credit_history': 'Credit History', 'amount': 'Credit Amount', 'employment': 'Employment Duration'},
        template='plotly_white'
    )
    st.plotly_chart(fig)

    # Plot 6: Credit Amount Distribution by Employment Duration
    fig = px.box(
        df,
        x='employment',
        y='amount',
        color='employment',
        title='Credit Amount Distribution by Employment Duration (Boxplot)',
        labels={'employment': 'Employment Duration', 'amount': 'Credit Amount'},
        template='plotly_white'
    )
    st.plotly_chart(fig)

    # Plot 7: Housing Ownership Across Credit Purposes
    fig = px.histogram(
        df,
        x='purpose',
        color='housing',
        title="Distribution of Housing Ownership Across Credit Purposes",
        labels={'purpose': 'Credit Purpose', 'housing': 'Housing Ownership'},
        category_orders={"housing": ["own", "rent", "free"]},
        opacity=0.7
    )
    st.plotly_chart(fig)

    # Plot 8: Savings Level Distribution (Pie Chart)
    fig = px.pie(
        df,
        names='savings',
        title='Savings Level Distribution',
        hole=0.3
    )
    st.plotly_chart(fig)

    # Plot 9: Average Credit Amount by Age Group
    df['age_group'] = pd.cut(
        df['age'],
        bins=[18, 30, 40, 50, 60, 70, 100],
        labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71+']
    )
    avg_credit_by_age = df.groupby('age_group', observed=True)['amount'].mean().reset_index()
    fig = px.line(
        avg_credit_by_age,
        x='age_group',
        y='amount',
        markers=True,
        title='Average Credit Amount by Age Group',
        labels={'amount': 'Average Credit Amount', 'age_group': 'Age Group'}
    )
    st.plotly_chart(fig)

    # Plot 10: Relationship Between Income and Credit Amount
    def generate_income(row):
        if row['employment'] == 'unemployed':
            return np.random.normal(500, 150)
        elif row['employment'] in ['<1', '1<=X<4']:
            return np.random.normal(1200, 300)
        elif row['employment'] in ['4<=X<7', '>=7']:
            return np.random.normal(2500, 600)
        else:
            return np.random.normal(1800, 400)

    df['income'] = df.apply(generate_income, axis=1)
    df['income'] = df['income'].clip(lower=100)
    fig = px.scatter(
        df,
        x='income',
        y='amount',
        title='Relationship Between Income and Credit Amount',
        labels={'income': 'Monthly Income', 'amount': 'Credit Amount'},
        opacity=0.7
    )
    st.plotly_chart(fig)

# Main function
def main():
    st.markdown("""
    <div class="main-header">
        <h1>💳 Credit Risk Analysis Dashboard</h1>
        <p>Analyze credit data, identify risk patterns, and make informed decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        filtered_df = apply_filters(df)
        st.success(f"📋 Showing {len(filtered_df):,} orders out of {len(df):,} total orders")

        kpis = calculate_kpis(filtered_df)
        display_kpis(kpis)

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if len(filtered_df) > 0:
                st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False).encode('utf-8'), file_name=f'ecommerce_data_filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')
        with col2:
            st.download_button("📥 Download Full Dataset", df.to_csv(index=False).encode('utf-8'), file_name=f'ecommerce_data_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')

        if st.sidebar.checkbox("Show Raw Data"):
            st.markdown("### 📄 Raw Data Preview")
            if len(filtered_df) > 0:
                st.dataframe(filtered_df, use_container_width=True, height=300)
            else:
                st.info("No data to display with current filters")

        if len(filtered_df) > 0:
            create_visualizations(filtered_df)
        else:
            st.warning("⚠️ No data available with current filters.")

        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Total Records:** {len(filtered_df):,}")
        with col2:
            st.info(f"**Total Columns:** {len(filtered_df.columns)}")
        with col3:
            st.info(f"**Memory Usage:** {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        st.markdown("---")
        st.markdown("""<div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>Credit Analytics Dashboard v1.0</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.error("Unable to load the dataset. Please check if 'ecommerce.csv' exists in the current directory.")

# Run
if __name__ == "__main__":
    main()
