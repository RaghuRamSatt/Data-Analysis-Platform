import os
import multiprocessing

# Try to get physical CPU count, fallback to logical CPU count if that fails
try:
    cpu_count = len(os.sched_getaffinity(0))
except AttributeError:
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = 4  # Fallback to a reasonable default

# Reserve one core for system operations
cpu_count = max(cpu_count - 1, 1)

# Set the environment variable
os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu_count)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_selection import VarianceThreshold
from .config import Config
from functools import lru_cache
import sys
import os 
import multiprocessing

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_data():
    try:
        data = request.json['data']
        params = request.json.get('params', {})
        df = pd.DataFrame(data)
        
        # Validate data
        validation_error = validate_data(data, df)
        if validation_error:
            return validation_error
        
        # Convert target column to numeric if present
        if 'target' in df.columns:
            df['target'] = pd.to_numeric(df['target'], errors='coerce')
            target = df.pop('target').values
            feature_names = df.columns.tolist()
        else:
            feature_names = df.columns.tolist()
            target = None
        
        # Preprocessing
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Feature selection
        selector = VarianceThreshold(threshold=0.1)
        selected_data = selector.fit_transform(scaled_data)
        selected_features = [f for f, s in zip(feature_names, selector.get_support()) if s]
        
        # Dimensionality reduction
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(selected_data)
        pca_explained_variance = pca.explained_variance_ratio_
        
        tsne = TSNE(n_components=3, random_state=42)
        tsne_result = tsne.fit_transform(selected_data)
        
        # Clustering
        n_clusters = params.get('n_clusters', 3)
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(selected_data)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(selected_data)
        
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(selected_data)
        
        # OPTICS clustering
        optics = OPTICS(min_samples=min_samples, max_eps=eps)
        optics_labels = optics.fit_predict(selected_data)
        
        # GMM clustering
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(selected_data)
        
        # Calculate metrics
        metrics = {}
        metrics['kmeans_silhouette'] = silhouette_score(selected_data, kmeans_labels)
        metrics['kmeans_calinski'] = calinski_harabasz_score(selected_data, kmeans_labels)
        
        if len(set(dbscan_labels)) > 1:  # Only calculate if DBSCAN found more than one cluster
            metrics['dbscan_silhouette'] = silhouette_score(selected_data, dbscan_labels)
            metrics['dbscan_calinski'] = calinski_harabasz_score(selected_data, dbscan_labels)
        else:
            metrics['dbscan_silhouette'] = None
            metrics['dbscan_calinski'] = None
            
        metrics['hierarchical_silhouette'] = silhouette_score(selected_data, hierarchical_labels)
        metrics['hierarchical_calinski'] = calinski_harabasz_score(selected_data, hierarchical_labels)
        
        # OPTICS metrics
        if len(set(optics_labels)) > 1:
            metrics['optics_silhouette'] = silhouette_score(selected_data, optics_labels)
            metrics['optics_calinski'] = calinski_harabasz_score(selected_data, optics_labels)
        else:
            metrics['optics_silhouette'] = None
            metrics['optics_calinski'] = None
            
        # GMM metrics
        metrics['gmm_silhouette'] = silhouette_score(selected_data, gmm_labels)
        metrics['gmm_calinski'] = calinski_harabasz_score(selected_data, gmm_labels)
        
        # Calculate elbow curve
        max_clusters = min(10, len(df) - 1)
        elbow_scores = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(selected_data)
            elbow_scores.append(float(kmeans.inertia_))  # Convert numpy.float64 to Python float
        
        # Convert numpy arrays to lists for JSON serialization
        return jsonify({
            'feature_names': feature_names,
            'selected_features': selected_features,
            'pca': pca_result.tolist(),
            'pca_explained_variance': pca_explained_variance.tolist(),
            'tsne': tsne_result.tolist(),
            'kmeans_labels': kmeans_labels.tolist(),
            'dbscan_labels': dbscan_labels.tolist(),
            'hierarchical_labels': hierarchical_labels.tolist(),
            'optics_labels': optics_labels.tolist(),
            'gmm_labels': gmm_labels.tolist(),
            'elbow_scores': elbow_scores,
            'original_data': df.values.tolist(),
            **metrics
        })
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return jsonify({'error': str(e)}), 500

def validate_data(data, df):
    if not data or not isinstance(data, list):
        return {'error': 'Invalid data format'}, 400
    
    if df.empty:
        return {'error': 'Empty dataset'}, 400
    
    if df.shape[1] < 2:
        return {'error': 'Dataset must have at least two columns'}, 400
    
    if len(data) > 10000:
        return {'error': 'Dataset too large. Maximum 10,000 rows allowed.'}, 400
        
    if sys.getsizeof(data) > 50 * 1024 * 1024:
        return {'error': 'Dataset exceeds memory limit of 50MB'}, 400
    
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if not non_numeric_cols.empty:
        return {'error': f'Non-numeric data found in columns: {", ".join(non_numeric_cols)}'}, 400
    
    return None

@lru_cache(maxsize=32)
def perform_dimensionality_reduction(data_hash, data):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)
    pca_explained_variance = pca.explained_variance_ratio_ * 100
    
    tsne = TSNE(n_components=3, random_state=42)
    tsne_result = tsne.fit_transform(data)
    
    return pca_result, pca_explained_variance, tsne_result

def perform_clustering(data, params):
    n_clusters = params.get('n_clusters', 3)
    eps = params.get('eps', 0.5)
    min_samples = params.get('min_samples', 5)
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    
    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(data)
    
    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(data)
    
    return kmeans_labels, dbscan_labels, hierarchical_labels, kmeans.inertia_

def calculate_metrics(data, kmeans_labels, dbscan_labels, hierarchical_labels):
    metrics = {}
    
    metrics['kmeans_silhouette'] = silhouette_score(data, kmeans_labels)
    metrics['kmeans_calinski'] = calinski_harabasz_score(data, kmeans_labels)
    
    if len(set(dbscan_labels)) > 1:
        metrics['dbscan_silhouette'] = silhouette_score(data, dbscan_labels)
        metrics['dbscan_calinski'] = calinski_harabasz_score(data, dbscan_labels)
    else:
        metrics['dbscan_silhouette'] = None
        metrics['dbscan_calinski'] = None
    
    metrics['hierarchical_silhouette'] = silhouette_score(data, hierarchical_labels)
    metrics['hierarchical_calinski'] = calinski_harabasz_score(data, hierarchical_labels)
    
    return metrics

if __name__ == '__main__':
    app.run(debug=True)
