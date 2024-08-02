#ερώτημα 1ο
import pandas as pd

def load_data(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.tsv'):
        data = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format")
    return data

#ερώτημα 2ο
def validate_data(data):
    if data.shape[1] < 2:
        raise ValueError("The data must have at least one feature and one label column.")
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    return features, labels

#ερώτημα 3ο
import plotly.express as px
from sklearn.decomposition import PCA
import umap

def visualize_data_2d(features, labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("Invalid method")

    reduced_data = reducer.fit_transform(features)
    fig = px.scatter(reduced_data, x=0, y=1, color=labels)
    return fig

def visualize_data_3d(features, labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=3)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=3)
    else:
        raise ValueError("Invalid method")

    reduced_data = reducer.fit_transform(features)
    fig = px.scatter_3d(reduced_data, x=0, y=1, z=2, color=labels)
    return fig

#ερώτημα 4ο
from sklearn.feature_selection import SelectKBest, f_classif

def feature_selection(features, labels, k):
    selector = SelectKBest(score_func=f_classif, k=k)
    selected_features = selector.fit_transform(features, labels)
    return selected_features


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification(features, labels, classifier='knn', param=3):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    if classifier == 'knn':
        model = KNeighborsClassifier(n_neighbors=param)
    else:
        raise ValueError("Unsupported classifier")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr')
    }

    return results


import streamlit as st

st.title('Machine Learning Application')

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    features, labels = validate_data(data)

    st.sidebar.title("Visualization")
    vis_method = st.sidebar.selectbox("Choose a method", ["PCA", "UMAP"])
    vis_type = st.sidebar.radio("Choose visualization type", ["2D", "3D"])

    if vis_type == "2D":
        fig = visualize_data_2d(features, labels, method=vis_method.lower())
    else:
        fig = visualize_data_3d(features, labels, method=vis_method.lower())

    st.plotly_chart(fig)

    st.sidebar.title("Feature Selection")
    k = st.sidebar.slider("Select number of features", 1, features.shape[1], 5)
    selected_features = feature_selection(features, labels, k)

    st.write("Selected Features Shape:", selected_features.shape)

    st.sidebar.title("Classification")
    classifier = st.sidebar.selectbox("Choose a classifier", ["KNN"])
    param = st.sidebar.slider("Set parameter (e.g., k for KNN)", 1, 20, 3)

    results_before = classification(features, labels, classifier=classifier.lower(), param=param)
    results_after = classification(selected_features, labels, classifier=classifier.lower(), param=param)

    st.write("Results before feature selection:", results_before)
    st.write("Results after feature selection:", results_after)

#ερωτημα 5ο
import pandas as pd

def detailed_results(results_before, results_after):
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "ROC AUC"],
        "Before Feature Selection": [results_before['accuracy'], results_before['f1_score'], results_before['roc_auc']],
        "After Feature Selection": [results_after['accuracy'], results_after['f1_score'], results_after['roc_auc']]
    })
    return comparison_df



    results_before = classification(features, labels, classifier=classifier.lower(), param=param)
    results_after = classification(selected_features, labels, classifier=classifier.lower(), param=param)
    comparison_df = detailed_results(results_before, results_after)
    st.write("Comparison of Results:", comparison_df)

#ερωτημα 6ο
st.sidebar.title("Info")
st.sidebar.info("""
    ## Application Information
    This application allows you to load and analyze tabular data, perform feature selection, and apply classification algorithms.

    ### Development Team
    - Member 1: Task A
    - Member 2: Task B
    - Member 3: Task C

    ### Tasks
    - Data Loading and Validation
    - Visualization and EDA
    - Machine Learning: Feature Selection and Classification
    - Results Presentation and Comparison
    - Docker and GitHub Integration
    - Documentation and Reporting
""")

#ενσωματωση στοιχειων
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import umap
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import plotly.express as px

def load_data(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.tsv'):
        data = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format")
    return data

def validate_data(data):
    if data.shape[1] < 2:
        raise ValueError("The data must have at least one feature and one label column.")
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    return features, labels

def visualize_data_2d(features, labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("Invalid method")

    reduced_data = reducer.fit_transform(features)
    fig = px.scatter(reduced_data, x=0, y=1, color=labels)
    return fig

def visualize_data_3d(features, labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=3)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=3)
    else:
        raise ValueError("Invalid method")

    reduced_data = reducer.fit_transform(features)
    fig = px.scatter_3d(reduced_data, x=0, y=1, z=2, color=labels)
    return fig

def feature_selection(features, labels, k):
    selector = SelectKBest(score_func=f_classif, k=k)
    selected_features = selector.fit_transform(features, labels)
    return selected_features

def classification(features, labels, classifier='knn', param=3):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    if classifier == 'knn':
        model = KNeighborsClassifier(n_neighbors=param)
    else:
        raise ValueError("Unsupported classifier")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr')
    }

    return results

def detailed_results(results_before, results_after):
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "ROC AUC"],
        "Before Feature Selection": [results_before['accuracy'], results_before['f1_score'], results_before['roc_auc']],
        "After Feature Selection": [results_after['accuracy'], results_after['f1_score'], results_after['roc_auc']]
    })
    return comparison_df

st.title('Machine Learning Application')

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    features, labels = validate_data(data)

    st.sidebar.title("Visualization")
    vis_method = st.sidebar.selectbox("Choose a method", ["PCA", "UMAP"])
    vis_type = st.sidebar.radio("Choose visualization type", ["2D", "3D"])

    if vis_type == "2D":
        fig = visualize_data_2d(features, labels, method=vis_method.lower())
    else:
        fig = visualize_data_3d(features, labels, method=vis_method.lower())

    st.plotly_chart(fig)

    st.sidebar.title("Feature Selection")
    k = st.sidebar.slider("Select number of features", 1, features.shape[1], 5)
    selected_features = feature_selection(features, labels, k)

    st.write("Selected Features Shape:", selected_features.shape)

    st.sidebar.title("Classification")
    classifier = st.sidebar.selectbox("Choose a classifier", ["KNN"])
    param = st.sidebar.slider("Set parameter (e.g., k for KNN)", 1, 20, 3)

    if features is not None and labels is not None:
        results_before = classification(features, labels, classifier=classifier.lower(), param=param)
        results_after = classification(selected_features, labels, classifier=classifier.lower(), param=param)

        comparison_df = detailed_results(results_before, results_after)
        st.write("Comparison of Results:", comparison_df)

st.sidebar.title("Info")
st.sidebar.info("""
    ## Application Information
    This application allows you to load and analyze tabular data, perform feature selection, and apply classification algorithms.
    
    ### Development Team
    - Member 1: Task A
    - Member 2: Task B
    - Member 3: Task C
    
    ### Tasks
    - Data Loading and Validation
    - Visualization and EDA
    - Machine Learning: Feature Selection and Classification
    - Results Presentation and Comparison
    - Docker and GitHub Integration
    - Documentation and Reporting
""")
