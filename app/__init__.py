from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/process', methods=['POST'])
def process_data():
    try:
        print("Received data for processing")
        data = request.json['data']
        df = pd.DataFrame(data)
        feature_names = df.columns.tolist()
        print(f"DataFrame shape: {df.shape}")
        
        print("Imputing missing values...")
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(df)
        
        print("Scaling data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_data)
        
        print("Performing PCA...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        pca_explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
        
        
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(scaled_data)
        
        print("Performing K-means clustering...")
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(scaled_data)
        
        print("Performing DBSCAN clustering...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)
        
        print("Processing complete. Sending results...")
        return jsonify({
            'feature_names': feature_names,
            'pca': pca_result.tolist(),
            'pca_explained_variance': pca_explained_variance.tolist(),
            'tsne': tsne_result.tolist(),
            'kmeans_labels': kmeans_labels.tolist(),
            'dbscan_labels': dbscan_labels.tolist(),
            'original_data': df.values.tolist()
        })
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)