import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import warnings

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #e0e7ff; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 4px solid #6366f1; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_synthetic_data():
    """Generate correlated synthetic data"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate basic features
    data = {
        'Age': np.random.choice(['18-20', '21-23', '24+'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Year_Level': np.random.choice(['1st', '2nd', '3rd', '4th'], n_samples),
        'Department': np.random.choice(['CE', 'CPE', 'ME', 'Archi'], n_samples),
        'Study_Load': np.random.randint(1, 6, n_samples),
        'Extracurricular': np.random.randint(1, 6, n_samples),
        'Academic_Perf': np.random.randint(1, 6, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate correlated symptoms
    def get_stress(row):
        base_stress = row['Study_Load']
        if row['Academic_Perf'] < 3: base_stress += 1
        return min(5, max(1, base_stress + np.random.randint(-1, 2)))

    def get_sleep(row):
        base_sleep = 6 - row['Stress_Levels']
        return min(5, max(1, base_sleep + np.random.randint(-1, 2)))

    df['Stress_Levels'] = df.apply(get_stress, axis=1)
    df['Sleep_Quality'] = df.apply(get_sleep, axis=1)
    df['Headache_Freq'] = df['Stress_Levels'].apply(lambda x: min(5, max(1, x + np.random.randint(-1, 2))))
    
    return df

def preprocess_data(df):
    """Preprocess data and handle encoding"""
    df = df.copy()
    
    # 1. Rename columns standardizer
    for col in df.columns:
        c_lower = col.lower()
        if "sleep" in c_lower: df.rename(columns={col: 'Sleep_Quality'}, inplace=True)
        elif "headache" in c_lower: df.rename(columns={col: 'Headache_Freq'}, inplace=True)
        elif "academic" in c_lower: df.rename(columns={col: 'Academic_Perf'}, inplace=True)
        elif "study load" in c_lower: df.rename(columns={col: 'Study_Load'}, inplace=True)
        elif "extracurricular" in c_lower: df.rename(columns={col: 'Extracurricular'}, inplace=True)
        elif "stress" in c_lower: df.rename(columns={col: 'Stress_Levels'}, inplace=True)
        elif "age" in c_lower: df.rename(columns={col: 'Age'}, inplace=True)
        elif "sex" in c_lower or "gender" in c_lower: df.rename(columns={col: 'Gender'}, inplace=True)
        elif "department" in c_lower: df.rename(columns={col: 'Department'}, inplace=True)
        elif "year" in c_lower: df.rename(columns={col: 'Year_Level'}, inplace=True)
    
    # 2. Create Target Variable (Risk Score)
    df['Risk_Score'] = (df['Stress_Levels'] + df['Headache_Freq'] + (6 - df['Sleep_Quality'])) / 3
    
    def categorize_risk(score):
        if score >= 3.5: return 'High Risk'
        elif score >= 2.5: return 'Moderate Risk'
        else: return 'Low Risk'
    
    df['Target_Risk'] = df['Risk_Score'].apply(categorize_risk)
    
    # 3. Encoding
    encoders = {}
    
    # Encode Target
    target_le = LabelEncoder()
    target_le.fit(['Low Risk', 'Moderate Risk', 'High Risk'])
    df['Target_Risk_Encoded'] = df['Target_Risk'].apply(lambda x: target_le.transform([x])[0])
    
    # Encode Categorical Features
    categorical_cols = ['Age', 'Gender', 'Year_Level', 'Department']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    return df, target_le, encoders

def train_models(X_train, X_test, y_train, y_test):
    """Train ML models"""
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append({"Model": name, "Accuracy": f"{acc:.4f}", "F1-Score": f"{f1:.4f}"})
    
    return trained_models, pd.DataFrame(results)

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("Configuration")
data_source = st.sidebar.radio("Select Data Source:", ["Upload CSV", "Generate Demo Data"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
    else:
        df = None
else:
    df = generate_synthetic_data()

# ============================================
# MAIN APP
# ============================================
st.title("Mental Health Risk Predictor")
st.markdown("### USTP CEA Student Mental Health Classification System")

if df is None:
    st.warning("Please upload data or select 'Generate Demo Data'.")
else:
    # Process Data
    df_processed, target_le, encoders = preprocess_data(df)
    
    # Features (remove target leakage)
    excluded_cols = ['Risk_Score', 'Target_Risk', 'Target_Risk_Encoded']
    feature_cols = [c for c in df_processed.columns if c not in excluded_cols]
    
    X = df_processed[feature_cols]
    y = df_processed['Target_Risk_Encoded']
    
    # Scale Data
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Features Used", len(feature_cols))
    col3.metric("Test Size", len(X_test))
    
    # Train
    trained_models, results_df = train_models(X_train, X_test, y_train, y_test)
    best_model_name = results_df.sort_values('Accuracy', ascending=False).iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    # ============================================
    # TABS
    # ============================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Visualizations", 
        "Performance", 
        "Feature Importance", 
        "Confusion Matrix", 
        "Predict", 
        "Raw Data"
    ])
    
    # --- TAB 1: VISUALIZATIONS ---
    with tab1:
        st.subheader("Exploratory Data Analysis")
        
        # 1. Target Distribution
        st.markdown("#### 1. Mental Health Risk Distribution")
        risk_counts = df_processed['Target_Risk'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(risk_counts)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            # Custom colors for risk
            colors = {'Low Risk': '#22c55e', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444'}
            pie_colors = [colors.get(x, '#999999') for x in risk_counts.index]
            
            risk_counts.plot.pie(autopct='%1.1f%%', ax=ax, colors=pie_colors, textprops={'color':"white"})
            ax.set_ylabel('')
            st.pyplot(fig)

        st.markdown("---")

        # 2. Categorical Breakdown (Demographics)
        st.markdown("#### 2. Risk Levels by Group")
        cat_options = [c for c in ['Department', 'Gender', 'Year_Level'] if c in df.columns]
        
        if cat_options:
            selected_cat = st.selectbox("Select Category to Analyze:", cat_options)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            
            sns.countplot(data=df_processed, x=selected_cat, hue='Target_Risk', 
                          palette={'Low Risk': '#22c55e', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444'}, ax=ax)
            
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.legend(facecolor='#1e293b', labelcolor='white')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # 3. Correlation Heatmap
        st.markdown("#### 3. Correlation Heatmap (Numeric Factors)")
        numeric_df = df_processed.select_dtypes(include=[np.number]).drop(columns=['Target_Risk_Encoded'], errors='ignore')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f172a')
        
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.tick_params(colors='white')
        st.pyplot(fig)

    # --- TAB 2: MODEL PERFORMANCE ---
    with tab2:
        st.subheader("Model Accuracy")
        st.dataframe(results_df, use_container_width=True)
        st.success(f"Best Model: {best_model_name}")
        
    # --- TAB 3: FEATURE IMPORTANCE ---
    with tab3:
        st.subheader("What affects the risk the most?")
        if best_model_name == "Random Forest":
            importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            importances.plot(kind='barh', ax=ax, color='#6366f1')
            ax.tick_params(colors='white')
            st.pyplot(fig)
        else:
            st.info("Feature importance is best visualized with Random Forest.")
            
    # --- TAB 4: CONFUSION MATRIX ---
    with tab4:
        st.subheader("Confusion Matrix")
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_le.classes_, yticklabels=target_le.classes_, ax=ax)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        
    # --- TAB 5: PREDICTION ---
    with tab5:
        st.subheader("Individual Prediction")
        
        user_input = {}
        cols = st.columns(2)
        
        for i, col in enumerate(feature_cols):
            with cols[i % 2]:
                if col in encoders:
                    original_values = encoders[col].classes_
                    selected_val = st.selectbox(f"Select {col}", original_values)
                    user_input[col] = encoders[col].transform([selected_val])[0]
                else:
                    user_input[col] = st.slider(f"{col}", 1, 5, 3)
        
        if st.button("Predict Risk"):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            
            prediction_idx = best_model.predict(input_scaled)[0]
            prediction_label = target_le.inverse_transform([prediction_idx])[0]
            prob = max(best_model.predict_proba(input_scaled)[0])
            
            if prediction_label == 'High Risk':
                st.error(f"Result: {prediction_label} ({prob:.1%} confidence)")
            elif prediction_label == 'Moderate Risk':
                st.warning(f"Result: {prediction_label} ({prob:.1%} confidence)")
            else:
                st.success(f"Result: {prediction_label} ({prob:.1%} confidence)")

    # --- TAB 6: RAW DATA ---
    with tab6:
        st.dataframe(df.head(20))