import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import plotly.express as px
import sqlite3
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()


SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL= os.getenv("SUPABASE_URL")
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)
# -------------------------
# SESSION STATE INIT
# -------------------------
if "auto_analyze" not in st.session_state:
    st.session_state.auto_analyze = False
if "result" not in st.session_state:
    st.session_state.result = None
if "df" not in st.session_state:
    st.session_state.df = None
if "insights" not in st.session_state:
    st.session_state.insights = None

if "chart_type" not in st.session_state:
    st.session_state.chart_type = None
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "llm" not in st.session_state:
    st.session_state.llm = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""    
if "current_question" not in st.session_state:
    st.session_state.current_question = "" 
if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = []
if "root_cause_result" not in st.session_state:
    st.session_state.root_cause_result = ""

if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = []       
            
# generate dashboard metrics
def forecasting_dashboard(df, llm):

    st.subheader("🔮 Forecasting Dashboard")

    try:

        from prophet import Prophet

        forecast_df = df.copy()

        date_cols = forecast_df.select_dtypes(
            include=["datetime64[ns]"]
        ).columns.tolist()

        numeric_cols = forecast_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(date_cols) == 0:

            st.warning("No date column found")
            return

        if len(numeric_cols) == 0:

            st.warning("No numeric column found")
            return

        date_col = st.selectbox(
            "Date Column",
            date_cols,
            key="forecast_date"
        )

        target_col = st.selectbox(
            "Target Column",
            numeric_cols,
            key="forecast_target"
        )

        forecast_days = st.slider(
            "Forecast Days",
            7,
            365,
            30
        )

        if st.button(
            "Generate Forecast",
            key="forecast_btn"
        ):

            prophet_df = forecast_df[
                [date_col, target_col]
            ].copy()

            prophet_df.columns = ["ds", "y"]

            prophet_df.dropna(inplace=True)

            model = Prophet()

            model.fit(prophet_df)

            future = model.make_future_dataframe(
                periods=forecast_days
            )

            forecast = model.predict(future)

            st.dataframe(
                forecast[
                    ["ds", "yhat"]
                ].tail(forecast_days)
            )

            fig = px.line(
                forecast,
                x="ds",
                y="yhat",
                title=f"{target_col} Forecast"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    except Exception as e:

        st.error(str(e))
def find_common_columns(dataframes):

    relationships = []

    names = list(dataframes.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            df1 = dataframes[names[i]]
            df2 = dataframes[names[j]]

            common = list(
                set(df1.columns)
                &
                set(df2.columns)
            )

            if common:

                relationships.append(
                    (
                        names[i],
                        names[j],
                        common
                    )
                )

    return relationships
def load_dataframe_to_sql(df):

    conn = sqlite3.connect(":memory:")

    df.to_sql(
        "data",
        conn,
        index=False,
        if_exists="replace"
    )

    return conn
def generate_dashboard(df):

    st.subheader("📈 Auto Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", len(df))

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicates",
            int(df.duplicated().sum())
        )
        # generate smart dashboard metrics
def generate_auto_dashboard(df):

    st.subheader("📊 Smart Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", len(df))

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicates",
            int(df.duplicated().sum())
        )       
        numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_cols) > 0:

        st.subheader("📈 Numeric Analysis")

        num_col = st.selectbox(
            "Choose Numeric Column",
            numeric_cols,
            key="dashboard_num"
        )

        fig = px.histogram(
            df,
            x=num_col,
            title=f"Distribution of {num_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        ) 
        cat_cols = df.select_dtypes(
        include="object"
    ).columns

    if len(cat_cols) > 0:

        st.subheader("🏷 Category Analysis")

        cat_col = st.selectbox(
            "Choose Category Column",
            cat_cols,
            key="dashboard_cat"
        )

        chart_df = (
            df[cat_col]
            .value_counts()
            .head(10)
            .reset_index()
        )

        chart_df.columns = [
            cat_col,
            "Count"
        ]

        fig = px.bar(
            chart_df,
            x=cat_col,
            y="Count",
            title=f"Top {cat_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )    
def dashboard_summary(llm, df):

    prompt = f"""
Dataset Columns:
{df.columns.tolist()}

Rows:
{len(df)}

Generate 5 business observations.

Keep concise.
"""

    response = llm.invoke(prompt)

    text = response.content

    if "<tool_call>" in text:
        text = text.split("<tool_call>")[-1]

    return text        
# -------------------------
# PDF Generator
# -------------------------
def generate_pdf(question, result, insights):
    pdf_file = "analysis_report.pdf"
    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Data Analyst Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Question:</b> {question}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Result:</b><br/>{str(result)}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>AI Insights:</b><br/>{insights}", styles["Normal"]))

    doc.build(elements)
    return pdf_file


# -------------------------
# AI INSIGHTS
# -------------------------
def generate_insights(llm, question, result):
    prompt = f"""
You are a senior business analyst.

Question:
{question}

Result:
{str(result)}

Provide:
1. Key Findings
2. Trends
3. Business Insights
4. Recommendations

Maximum 5 bullet points.
Do not show reasoning.
"""
    response = llm.invoke(prompt)
    insights = response.content.strip()
    if "<tool_call>" in insights and "<tool_call>" in insights:
        insights = insights.split("<tool_call>")[-1].strip()
    return insights


def get_chart_type(llm, question):
    prompt = f"""
You are a data visualization expert.

Question:
{question}

Choose ONLY one:
bar
line
pie
histogram
scatter
none

Return only the chart name.
"""
    response = llm.invoke(prompt)
    chart_type = response.content.strip().lower()
    if "<tool_call>" in chart_type and "<tool_call>" in chart_type:
        chart_type = chart_type.split("<tool_call>")[-1].strip()
    return chart_type
def generate_suggested_questions(llm, columns):

    prompt = f"""
You are a data analyst.

Dataset Columns:
{columns}

Suggest 5 useful questions a user can ask.

Rules:
1. One question per line
2. No numbering
3. No explanation
"""

    response = llm.invoke(prompt)

    questions = response.content.strip()

    if "<tool_call>" in questions:
        questions = questions.split("<tool_call>")[-1].strip()

    return questions.split("\n")

# -------------------------
# ENV LOAD
# -------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
api_key = os.getenv("GROQ_API_KEY")


# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst Agent")
st.sidebar.title("📝 Analysis History")
for item in reversed(st.session_state.chat_history):

    st.sidebar.markdown(
        f"**Q:** {item['question']}"
    )

    st.sidebar.caption(
        item['result']
    )

    st.sidebar.divider()
    # Clear History Button
if st.sidebar.button("🗑 Clear History"):

    st.session_state.chat_history = []

    st.rerun()
if not api_key:
    st.error("GROQ_API_KEY not found")
    st.stop()


# -------------------------
# LLM INIT (GLOBAL)
# -------------------------
if st.session_state.llm is None:
    st.session_state.llm = ChatGroq(
        model="llama-3.3-70b-versatile", api_key=api_key, temperature=0
    )

llm = st.session_state.llm


# -------------------------
# UPLOAD CSV FILES
# -------------------------

uploaded_files = st.file_uploader(
    "Upload CSV Files",
    type=["csv"],
    accept_multiple_files=True
)
if uploaded_files:

    dataframes = {}

    for file in uploaded_files:
        dataframes[file.name] = pd.read_csv(file)

    st.session_state.dataframes = dataframes
    dataframes = st.session_state.dataframes

else:

    dataframes = st.session_state.get(
        "dataframes",
        {}
    )

# -------------------------
# SELECT DATASET
# -------------------------

if dataframes:

    selected_dataset = st.selectbox(
        "Choose Dataset",
        list(dataframes.keys())
    )

    df = dataframes[selected_dataset]

    st.session_state.df = df
    st.session_state.selected_dataset = selected_dataset

else:

    st.info("Please upload CSV files.")
    st.stop()

## ====================================
# PHASE 18 - AUTHENTICATION
# ====================================

st.sidebar.title("🔐 Authentication")

auth_mode = st.sidebar.radio(
    "Choose",
    ["Login", "Signup"]
)

email = st.sidebar.text_input(
    "Email"
)

password = st.sidebar.text_input(
    "Password",
    type="password"
)

# -------------------------
# SIGNUP
# -------------------------

if auth_mode == "Signup":

    if st.sidebar.button("Create Account"):

        try:

            supabase.auth.sign_up(
                {
                    "email": email,
                    "password": password
                }
            )

            st.sidebar.success(
                "Account Created Successfully"
            )

        except Exception as e:

            st.sidebar.error(str(e))

# -------------------------
# LOGIN
# -------------------------

if auth_mode == "Login":

    if st.sidebar.button("Login"):

        try:

            response = supabase.auth.sign_in_with_password(
                {
                    "email": email,
                    "password": password
                }
            )

            st.session_state.user = email

            st.sidebar.success(
                "Login Successful"
            )

        except Exception as e:

            st.sidebar.error(str(e))

# -------------------------
# LOGIN CHECK
# -------------------------

if "user" not in st.session_state:

    st.warning(
        "🔐 Please Login First"
    )

    st.stop()

st.sidebar.success(
    f"Logged in as {st.session_state.user}"
)

# -------------------------
# LOGOUT
# -------------------------

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# =====================================================
# PHASE 12 - SQL ANALYTICS
# =====================================================

st.subheader("🗄 SQL Analytics Mode")

sql_query = st.text_area(
    "Write SQL Query",
    height=150,
    placeholder="""
SELECT *
FROM data
LIMIT 10
"""
)

natural_query = st.text_input("Ask in English")

if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = ""

if st.button("Generate SQL") and natural_query:

    prompt = f"""
Dataset Columns:
{df.columns.tolist()}

Convert the question into SQLite SQL.

Question:
{natural_query}

Table name is data.

Return ONLY SQL.
"""

    response = llm.invoke(prompt)

    st.session_state.generated_sql = (
        response.content
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

if st.session_state.generated_sql:
    st.code(st.session_state.generated_sql, language="sql")

# =====================================================
# PHASE 13 - DATA CLEANING AGENT
# =====================================================

st.divider()

st.subheader("🧹 Data Cleaning Agent")

if st.button("🧹 Clean Data"):

    cleaned_df = df.copy()

    missing_before = cleaned_df.isnull().sum().sum()

    duplicates_removed = cleaned_df.duplicated().sum()

    cleaned_df.drop_duplicates(inplace=True)

    cleaned_df.replace(
        ["", " ", "N/A", "NULL", "null", "NaN"],
        pd.NA,
        inplace=True
    )

    # Fill numeric columns
    for col in cleaned_df.select_dtypes(include="number").columns:
        cleaned_df[col] = cleaned_df[col].fillna(
            cleaned_df[col].median()
        )

    # Fill text columns
    for col in cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns:

        if cleaned_df[col].isnull().sum() > 0:

            mode_values = cleaned_df[col].mode()

            if not mode_values.empty:
                cleaned_df[col] = cleaned_df[col].fillna(
                    mode_values[0]
                )
            else:
                cleaned_df[col] = cleaned_df[col].fillna(
                    "Unknown"
                )

    missing_after = cleaned_df.isnull().sum().sum()

    st.success("✅ Data Cleaned Successfully")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Duplicates Removed",
            int(duplicates_removed)
        )

    with col2:
        st.metric(
            "Missing Before",
            int(missing_before)
        )

    with col3:
        st.metric(
            "Missing After",
            int(missing_after)
        )

    st.write(f"Rows after cleaning: {len(cleaned_df)}")

    st.dataframe(cleaned_df.head())

    csv = cleaned_df.to_csv(index=False)

    st.download_button(
        label="⬇ Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    st.info(
        f"""
Cleaning Summary:
• Removed {duplicates_removed} duplicate rows
• Missing values before cleaning: {missing_before}
• Missing values after cleaning: {missing_after}
• Numeric columns filled using median
• Text columns filled using mode
"""
    )
  #     # -------------------------
    # PHASE 14 - FORECASTING AGENT
    # -------------------------

    st.subheader("📈 Forecasting Agent")

    try:

        from prophet import Prophet

        forecast_df = df.copy()

        # Detect date columns
        for col in forecast_df.columns:

            try:

                forecast_df[col] = pd.to_datetime(
                    forecast_df[col],
                    errors="ignore"
                )

            except Exception:
                pass

        # Convert possible numeric text columns
        for col in forecast_df.columns:

            try:

                forecast_df[col] = pd.to_numeric(
                    forecast_df[col]
                )

            except Exception:
                pass

        date_cols = forecast_df.select_dtypes(
            include=["datetime64[ns]"]
        ).columns.tolist()

        numeric_cols = forecast_df.select_dtypes(
            include="number"
        ).columns.tolist()

        # Debug information
        st.write("📅 Date Columns:", date_cols)
        st.write("📊 Numeric Columns:", numeric_cols)

        if len(date_cols) == 0:

            st.warning(
                "No date column detected for forecasting."
            )

        elif len(numeric_cols) == 0:

            st.warning(
                "No numeric column detected for forecasting."
            )

        else:

            col1, col2 = st.columns(2)

            with col1:

                date_col = st.selectbox(
                    "📅 Select Date Column",
                    date_cols
                )

            with col2:

                target_col = st.selectbox(
                    "📊 Select Target Column",
                    numeric_cols
                )

            forecast_days = st.slider(
                "Forecast Future Days",
                min_value=7,
                max_value=365,
                value=30
            )

            if st.button("🚀 Generate Forecast"):

                with st.spinner(
                    "Training Forecast Model..."
                ):

                    prophet_df = forecast_df[
                        [date_col, target_col]
                    ].copy()

                    prophet_df.columns = [
                        "ds",
                        "y"
                    ]

                    prophet_df.dropna(
                        inplace=True
                    )

                    prophet_df = prophet_df.sort_values(
                        "ds"
                    )

                    model = Prophet()

                    model.fit(
                        prophet_df
                    )

                    future = model.make_future_dataframe(
                        periods=forecast_days
                    )

                    forecast = model.predict(
                        future
                    )

                    st.success(
                        "Forecast Generated Successfully"
                    )

                    # Forecast Results
                    st.subheader(
                        "📋 Forecast Results"
                    )

                    forecast_result = forecast[
                        [
                            "ds",
                            "yhat",
                            "yhat_lower",
                            "yhat_upper"
                        ]
                    ]

                    st.dataframe(
                        forecast_result.tail(
                            forecast_days
                        )
                    )

                    # Forecast Chart
                    st.subheader(
                        "📈 Forecast Chart"
                    )

                    fig = px.line(
                        forecast,
                        x="ds",
                        y="yhat",
                        title=f"{target_col} Forecast"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # Confidence Interval
                    st.subheader(
                        "📊 Confidence Interval"
                    )

                    fig2 = px.line(
                        forecast,
                        x="ds",
                        y=[
                            "yhat",
                            "yhat_lower",
                            "yhat_upper"
                        ],
                        title="Forecast Confidence Interval"
                    )

                    st.plotly_chart(
                        fig2,
                        use_container_width=True
                    )

                    # Download CSV
                    csv_forecast = forecast_result.to_csv(
                        index=False
                    )

                    st.download_button(
                        label="⬇ Download Forecast CSV",
                        data=csv_forecast,
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )

                    # AI Insights
                    summary_df = forecast_result.tail(
                        10
                    )

                    prompt = f"""
You are a senior business analyst.

Forecast Data:

{summary_df.to_string()}

Provide:
1. Trend
2. Growth/Decline
3. Risks
4. Recommendation

Maximum 5 bullet points.
"""

                    response = llm.invoke(
                        prompt
                    )

                    st.subheader(
                        "🤖 Forecast Insights"
                    )

                    st.write(
                        response.content
                    )

    except ImportError:

        st.info(
            "Install Prophet first using:\n\npip install prophet"
        )
    # =====================================================
# PHASE 15 - EXECUTIVE DASHBOARD
# =====================================================
        
# =====================================================
# PHASE 17 - ROLE BASED DASHBOARD
# =====================================================

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state.df

st.divider()
st.header("📊 Executive Dashboard")

role = st.sidebar.selectbox(
    "👤 Select Role",
    ["Admin", "Manager", "Analyst", "Viewer"],
    key="role_selector"
)

st.sidebar.success(f"Dashboard Role: {role}")

# =====================================================
# ADMIN
# =====================================================

if role == "Admin":

    overview_tab, analytics_tab, forecast_tab, reports_tab, copilot_tab = st.tabs(
        [
            "📊 Overview",
            "📈 Analytics",
            "🔮 Forecasting",
            "📄 Reports",
            "💬 AI Copilot"
        ]
    )

    # OVERVIEW
    with overview_tab:

        st.subheader("Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))

        st.dataframe(df.head())

    # ANALYTICS
    with analytics_tab:

        st.subheader("Analytics Dashboard")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:

            metric = st.selectbox(
                "Select Metric",
                numeric_cols,
                key="admin_metric"
            )

            fig = px.histogram(df, x=metric)

            st.plotly_chart(fig, use_container_width=True)

    # FORECASTING
    with forecast_tab:

        forecasting_dashboard(df, llm)

    # REPORTS
    with reports_tab:

        st.subheader("Executive Report")

        report_text = f"""
Rows: {len(df)}
Columns: {len(df.columns)}
Missing Values: {df.isnull().sum().sum()}
Duplicate Rows: {df.duplicated().sum()}
"""

        st.text(report_text)

        st.download_button(
            "⬇ Download Report",
            report_text,
            "executive_report.txt"
        )

    # COPILOT
    with copilot_tab:

        st.subheader("🤖 AI Copilot")

        question = st.text_input(
            "Ask a business question",
            key="admin_question"
        )

        if st.button("Analyze", key="admin_ai"):

            prompt = f"""
Dataset Columns:
{df.columns.tolist()}

Question:
{question}

Provide business insights.
"""

            response = llm.invoke(prompt)

            st.session_state.copilot_history.append(
                {
                    "question": question,
                    "answer": response.content
                }
            )

            st.write(response.content)

        st.subheader("📝 Copilot History")

        for item in reversed(st.session_state.copilot_history[-5:]):

            with st.expander(item["question"]):

                st.write(item["answer"])

# =====================================================
# MANAGER
# =====================================================

elif role == "Manager":

    overview_tab, analytics_tab, reports_tab = st.tabs(
        [
            "📊 Overview",
            "📈 Analytics",
            "📄 Reports"
        ]
    )

    with overview_tab:
        st.dataframe(df.head())

    with analytics_tab:

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:

            metric = st.selectbox(
                "Metric",
                numeric_cols,
                key="manager_metric"
            )

            fig = px.histogram(df, x=metric)

            st.plotly_chart(fig, use_container_width=True)

    with reports_tab:

        st.subheader("Manager Reports")

        st.write("Access to reports only.")

# =====================================================
# ANALYST
# =====================================================

elif role == "Analyst":

    analytics_tab, forecast_tab, copilot_tab = st.tabs(
        [
            "📈 Analytics",
            "🔮 Forecasting",
            "💬 AI Copilot"
        ]
    )

    with analytics_tab:

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:

            metric = st.selectbox(
                "Metric",
                numeric_cols,
                key="analyst_metric"
            )

            fig = px.histogram(df, x=metric)

            st.plotly_chart(fig, use_container_width=True)

    with forecast_tab:

        forecasting_dashboard(df, llm)

    with copilot_tab:

        st.subheader("🤖 AI Copilot")

        question = st.text_input(
            "Ask a business question",
            key="analyst_question"
        )

        if st.button(
            "Analyze",
            key="analyst_ai"
        ):

            response = llm.invoke(question)

            st.write(response.content)

# =====================================================
# VIEWER
# =====================================================

else:

    st.subheader("📊 Dataset Preview")

    st.dataframe(df.head())

    st.info("Viewer role has read-only access.")

    

# =====================================================
# ROOT CAUSE ANALYSIS
# =====================================================

if "root_cause_result" not in st.session_state:
    st.session_state.root_cause_result = ""

st.divider()

st.subheader("🔍 Root Cause Analysis")

issue = st.text_input(
    "Describe a business issue",
    placeholder="Sales dropped in June"
)

if st.button("Analyze Root Cause") and issue:

    with st.spinner("Finding root causes..."):

        sample_data = df.head(10).to_string()

        prompt = f"""
You are a senior business analyst.

Dataset Columns:
{df.columns.tolist()}

Dataset Sample:
{sample_data}

Problem:
{issue}

Find:
1. Likely Causes
2. Business Impact
3. Suggested Actions

Provide maximum 5 concise bullet points.
"""

        try:

            response = llm.invoke(prompt)

            st.session_state.root_cause_result = (
                response.content
            )

        except Exception as e:

            st.error(f"Analysis Error: {e}")

# Show saved result
if st.session_state.root_cause_result:

    st.success("Analysis Complete")

    st.write(
        st.session_state.root_cause_result
    )

# =====================================================
# SMART KPI DETECTION
# =====================================================

st.divider()

st.subheader("📊 Smart KPI Detection")

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

if len(numeric_cols) > 0:

    cols = st.columns(
        min(4, len(numeric_cols))
    )

    for i, col in enumerate(
        numeric_cols[:4]
    ):

        with cols[i]:

            st.metric(
                label=col,
                value=round(
                    df[col].sum(),
                    2
                )
            )

# =====================================================
# COPILOT HISTORY
# =====================================================

st.divider()

st.subheader("💬 Copilot History")

for item in reversed(
    st.session_state.copilot_history[-5:]
):

    with st.expander(
        item["question"]
    ):

        st.write(
            item["answer"]
        )

# =====================================================
# DATASET RELATIONSHIPS
# =====================================================

st.subheader("🔗 Dataset Relationships")

relations = find_common_columns(
    dataframes
)

for r in relations:

    st.success(
        f"{r[0]} ↔ {r[1]}"
    )

    st.write(
        f"Common Columns: {r[2]}"
    )

# =====================================================
# AUTO JOIN
# =====================================================

if relations:

    relation = relations[0]

    left_df = dataframes[relation[0]]

    right_df = dataframes[relation[1]]

    join_col = relation[2][0]

    merged_df = pd.merge(
        left_df,
        right_df,
        on=join_col
    )

    st.subheader(
        "🔄 Auto Joined Dataset"
    )

    st.dataframe(
        merged_df.head()
    )

# =====================================================
# AI RELATIONSHIP ANALYSIS
# =====================================================

prompt = f"""
Datasets:
{list(dataframes.keys())}

Suggest:
1. Which datasets should be joined
2. Join keys
3. Business insights possible

Keep response concise.
"""

response = llm.invoke(prompt)

st.subheader(
    "🤖 AI Relationship Analysis"
)

st.write(
    response.content
)

# =====================================================
# SESSION STATE INIT
# =====================================================

defaults = {
    "suggested_questions": [],
    "selected_question": "",
    "result": None,
    "chart_type": "bar",
    "insights": "",
    "current_question": "",
    "chat_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# DATASET OVERVIEW
# =====================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", len(df))

with col2:
    st.metric("Columns", len(df.columns))

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

with st.expander("Preview Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Columns"):
    st.write(df.columns.tolist())


# =====================================================
# AI SUGGESTED QUESTIONS
# =====================================================

st.subheader("💡 AI Suggested Questions")

refresh_col1, refresh_col2 = st.columns([1, 5])

with refresh_col1:

    if st.button("🔄 Refresh Questions"):

        st.session_state.suggested_questions = (
            generate_suggested_questions(
                llm,
                df.columns.tolist()
            )
        )

if not st.session_state.suggested_questions:

    st.session_state.suggested_questions = (
        generate_suggested_questions(
            llm,
            df.columns.tolist()
        )
    )

for idx, q in enumerate(
    st.session_state.suggested_questions
):

   if st.button(
    q,
    key=f"suggested_q_{idx}"
):
    st.session_state.selected_question = q
    st.session_state.auto_analyze = True
    st.rerun()

# =====================================================
# QUESTION INPUT
# =====================================================

question = st.text_input(
    "Ask anything about your data",
    value=st.session_state.selected_question
)
# =====================================================
# ANALYZE BUTTON
# =====================================================

analyze_clicked = st.button(
    "🚀 Analyze",
    key="analyze_btn"
)

if analyze_clicked or st.session_state.get("auto_analyze", False):

    if question:

        st.session_state.auto_analyze = False

        with st.spinner("Analyzing Data..."):

            prompt = f"""
You are an expert Pandas analyst.

DataFrame name is df.

Columns:
{df.columns.tolist()}

Question:
{question}

Rules:
1. Return ONLY executable Python code.
2. Store final answer in variable result.
3. Use dataframe df only.
4. No markdown.
5. No explanation.
6. No comments.
"""

            try:

                response = llm.invoke(prompt)

                code = (
                    response.content
                    .replace("```python", "")
                    .replace("```", "")
                    .strip()
                )

                if "</think>" in code:
                    code = code.split("</think>")[-1].strip()

                local_vars = {
                    "df": df,
                    "pd": pd
                }

                exec(code, {}, local_vars)

                result = local_vars.get("result")

                st.session_state.result = result
                st.session_state.current_question = question

                st.session_state.chart_type = (
                    get_chart_type(
                        llm,
                        question
                    )
                )

                st.session_state.insights = (
                    generate_insights(
                        llm,
                        question,
                        result
                    )
                )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "result": str(result)[:300]
                    }
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Analysis Error: {e}"
                )


# =====================================================
# RESULTS SECTION
# =====================================================

result = st.session_state.get("result")
chart_type = st.session_state.get("chart_type", "bar")
insights = st.session_state.get("insights", "")

if result is not None:

    st.divider()
    st.subheader("📌 Answer")

    # -------------------------
    # DATAFRAME RESULT
    # -------------------------

    if isinstance(result, pd.DataFrame):

        st.dataframe(
            result.head(100),
            use_container_width=True
        )

        csv_export = result.to_csv(index=False)

        st.download_button(
            "⬇ Download Result CSV",
            csv_export,
            file_name="analysis_result.csv",
            mime="text/csv"
        )

        numeric_cols = result.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_cols) > 0:

            st.subheader("📊 Visualization")

            try:

                if chart_type == "line":

                    fig = px.line(
                        result,
                        x=result.columns[0],
                        y=numeric_cols[0]
                    )

                elif chart_type == "pie":

                    fig = px.pie(
                        result,
                        names=result.columns[0],
                        values=numeric_cols[0]
                    )

                elif chart_type == "histogram":

                    fig = px.histogram(
                        result,
                        x=numeric_cols[0]
                    )

                elif (
                    chart_type == "scatter"
                    and len(numeric_cols) >= 2
                ):

                    fig = px.scatter(
                        result,
                        x=numeric_cols[0],
                        y=numeric_cols[1]
                    )

                else:

                    fig = px.bar(
                        result,
                        x=result.columns[0],
                        y=numeric_cols[0]
                    )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.warning(
                    f"Chart generation failed: {e}"
                )

    # -------------------------
    # SERIES RESULT
    # -------------------------

    elif isinstance(result, pd.Series):

        st.dataframe(result)

        chart_df = result.reset_index()

        chart_df.columns = [
            "Category",
            "Value"
        ]

        fig = px.bar(
            chart_df,
            x="Category",
            y="Value"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------
    # SCALAR RESULT
    # -------------------------

    else:

        st.success(str(result))

    # -------------------------
    # AI INSIGHTS
    # -------------------------

    if insights:

        st.subheader("📈 AI Insights")

        st.info(insights)

    # -------------------------
    # PDF REPORT
    # -------------------------

    if insights:

        try:

            pdf_path = generate_pdf(
                st.session_state.current_question,
                result,
                insights
            )

            with open(pdf_path, "rb") as pdf:

                st.download_button(
                    "📄 Download PDF Report",
                    pdf,
                    file_name="AI_Report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:

            st.error(
                f"PDF Generation Error: {e}"
            )


# =====================================================
# CHAT WITH DATA
# =====================================================

st.divider()
st.subheader("💬 Chat With Your Data")

follow_up = st.text_input(
    "Ask a follow-up question",
    key="follow_up"
)

if st.button("Ask Follow-up") and follow_up:

    with st.spinner("Thinking..."):

        prompt = f"""
Dataset Columns:
{df.columns.tolist()}

Previous Question:
{st.session_state.current_question}

Previous Result:
{str(result)[:3000]}

New Question:
{follow_up}

Return ONLY executable Python code.

Store final answer in variable result.
Use dataframe df only.
No markdown.
No explanation.
"""

        try:

            response = llm.invoke(prompt)

            code = (
                response.content
                .replace("```python", "")
                .replace("```", "")
                .strip()
            )

            if "</think>" in code:
                code = code.split("</think>")[-1].strip()

            local_vars = {
                "df": df,
                "pd": pd
            }

            exec(code, {}, local_vars)

            follow_result = local_vars.get("result")

            st.subheader("Follow-up Result")

            if isinstance(
                follow_result,
                (pd.DataFrame, pd.Series)
            ):

                st.dataframe(follow_result)

            else:

                st.success(str(follow_result))

        except Exception as e:

            st.error(
                f"Follow-up Error: {e}"
            )