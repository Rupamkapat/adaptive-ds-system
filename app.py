import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Adaptive DS System", page_icon="🤖", layout="wide")

st.title("🤖 Adaptive Data Science Model System")
st.markdown("*Automatically detects task type & recommends best models*")
st.markdown("---")

@st.cache_data
def load_data():
    df_titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df_boston = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
    df_air = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
    df_planets = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/planets.csv")
    return df_titanic, df_boston, df_air, df_planets

with st.spinner("Loading datasets..."):
    df_titanic, df_boston, df_air, df_planets = load_data()
st.success("✅ All 4 datasets loaded!")

dataset = st.sidebar.selectbox("Select Dataset:", ["Titanic (Classification)", "Boston Housing (Regression)", "Air Passengers (Time Series)", "Planets (Clustering)"])

if dataset == "Titanic (Classification)":
    st.header("🏔️ Titanic Survival Prediction")
    df = df_titanic.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop('Cabin', axis=1)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Best Model", "Random Forest")
    
    fig, ax = plt.subplots()
    importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance')
    ax.barh(importance['feature'], importance['importance'])
    ax.set_title("Feature Importance")
    st.pyplot(fig)

elif dataset == "Boston Housing (Regression)":
    st.header("🏠 Boston House Price Prediction")
    df = df_boston.copy()
    X = df.drop('medv', axis=1)
    y = df['medv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"${rmse:.1f}K")
    col2.metric("Best Model", "Random Forest")
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

elif dataset == "Air Passengers (Time Series)":
    st.header("✈️ Air Passengers Forecast")
    df = df_air.copy()
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Passengers'])
    ax.set_title("Monthly Air Passengers (1949-1960)")
    st.pyplot(fig)
    
    monthly = df.groupby(df.index.month)['Passengers'].mean()
    fig, ax = plt.subplots()
    ax.bar(range(1, 13), monthly)
    ax.set_title("Seasonal Pattern")
    st.pyplot(fig)

else:
    st.header("🪐 Planet Clustering")
    df = df_planets.dropna(subset=['mass', 'distance', 'year'])
    X = df[['mass', 'distance', 'year']]
    X_scaled = StandardScaler().fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, clusters)
    
    st.metric("Silhouette Score", f"{sil:.3f}")
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(X['mass'], X['distance'], c=clusters, cmap='viridis', alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Mass")
    ax.set_ylabel("Distance")
    ax.set_title("Planet Clusters")
    st.pyplot(fig)

st.markdown("---")
st.caption("🤖 Adaptive Data Science System | Built with Streamlit")
