import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Data KNN", layout="wide")

st.title("Student Dataset – Preprocessing Effect on KNN")
st.write("This app shows how preprocessing changes KNN accuracy.")

df = pd.read_csv("raw_student_dataset_100rows(1).csv")

st.header("Raw Data")
st.dataframe(df.head(10))
st.write(df.isnull().sum())

fig1, ax1 = plt.subplots(figsize=(7,4))
sns.heatmap(df.isnull(), cbar=False, ax=ax1)
st.pyplot(fig1)

df2 = df.copy()

df2['HomeworkCompletion'] = df2['HomeworkCompletion'].astype(str).str.lower().str.strip()
df2['Group'] = df2['Group'].astype(str).str.lower().str.strip()

num_cols = ['StudyHours', 'MathScore', 'ScienceScore', 'Attendance(%)']
for col in num_cols:
    df2[col] = df2[col].apply(lambda x: x if x >= 0 else None)
    df2[col].fillna(df2[col].median(), inplace=True)

df2['HomeworkCompletion'].fillna(df2['HomeworkCompletion'].mode()[0], inplace=True)
df2['Group'].fillna(df2['Group'].mode()[0], inplace=True)

encoder = LabelEncoder()
df2['HomeworkCompletion'] = encoder.fit_transform(df2['HomeworkCompletion'])
df2['Group'] = encoder.fit_transform(df2['Group'])

X = df2[['StudyHours', 'MathScore', 'ScienceScore', 'Attendance(%)', 'Group']]
y = df2['HomeworkCompletion']

st.header("Cleaned Data")
st.dataframe(df2.head(10))
st.write(df2.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train, y_train)
pred1 = knn1.predict(X_test)
acc1 = accuracy_score(y_test, pred1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_s, X_test_s, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_s, y_train)
pred2 = knn2.predict(X_test_s)
acc2 = accuracy_score(y_test, pred2)

st.header("Model Results")
st.metric("Accuracy without preprocessing", round(acc1, 3))
st.metric("Accuracy with preprocessing", round(acc2, 3))

st.write("After preprocessing, the KNN model performs better because all features are on the same scale.")

st.success("Project run completed")
