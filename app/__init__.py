from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_selection import VarianceThreshold

import traceback

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/process', methods=['POST'])
def process_data():
    try:
        data = request.json['data']
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid data format'}), 400
        
        params = request.json.get('params', {})

        df = pd.DataFrame(data)
        if df.empty:
            return jsonify({'error': 'Empty dataset'}), 400
        
        if df.shape[1] < 2:
            return jsonify({'error': 'Dataset must have at least two columns'}), 400
        
        feature_names = df.columns.tolist()
        
        # Check for non-numeric data
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if not non_numeric_cols.empty:
            return jsonify({'error': f'Non-numeric data found in columns: {", ".join(non_numeric_cols)}'}), 400
        
        print("Received data for processing") 
        print(f"DataFrame shape: {df.shape}")
        
        print("Scaling data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        print("Performing feature selection...")
        selector = VarianceThreshold(threshold=0.1)  # You can adjust this threshold
        selected_data = selector.fit_transform(scaled_data)
        selected_features = [feature for feature, selected in zip(feature_names, selector.get_support()) if selected]
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Use selected_data for further processing
        print("Performing PCA...")
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(selected_data)
        pca_explained_variance = pca.explained_variance_ratio_ * 100
        
        print("Performing t-SNE...")
        tsne = TSNE(n_components=3, random_state=42)
        tsne_result = tsne.fit_transform(selected_data)
        
        print("Performing K-means clustering...")
        n_clusters = params.get('n_clusters', 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(selected_data)
        kmeans_silhouette = silhouette_score(selected_data, kmeans_labels)
        kmeans_calinski = calinski_harabasz_score(selected_data, kmeans_labels)
        
        print("Performing DBSCAN clustering...")
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(selected_data)
        dbscan_silhouette = silhouette_score(selected_data, dbscan_labels) if len(set(dbscan_labels)) > 1 else None
        dbscan_calinski = calinski_harabasz_score(selected_data, dbscan_labels) if len(set(dbscan_labels)) > 1 else None
        
        print("Performing Hierarchical clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=3)
        hierarchical_labels = hierarchical.fit_predict(selected_data)
        hierarchical_silhouette = silhouette_score(selected_data, hierarchical_labels)
        hierarchical_calinski = calinski_harabasz_score(selected_data, hierarchical_labels)
        
        print("Performing elbow method for K-means...")
        elbow_scores = []
        max_clusters = min(10, len(df) - 1)  # Adjust the maximum number of clusters as needed
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(selected_data)
            elbow_scores.append(kmeans.inertia_)
        
        print("Processing complete. Sending results...")
        return jsonify({
            'feature_names': feature_names,
            'selected_features': selected_features,
            'pca': pca_result.tolist(),
            'pca_explained_variance': pca_explained_variance.tolist(),
            'tsne': tsne_result.tolist(),
            'kmeans_labels': kmeans_labels.tolist(),
            'kmeans_silhouette': kmeans_silhouette,
            'kmeans_calinski': kmeans_calinski,
            'dbscan_labels': dbscan_labels.tolist(),
            'dbscan_silhouette': dbscan_silhouette,
            'dbscan_calinski': dbscan_calinski,
            'hierarchical_labels': hierarchical_labels.tolist(),
            'hierarchical_silhouette': hierarchical_silhouette,
            'hierarchical_calinski': hierarchical_calinski,
            'elbow_scores': elbow_scores,
            'original_data': df.values.tolist()
        })
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)