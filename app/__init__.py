from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/process', methods=['POST'])
def process_data():
    data = request.json['data']
    df = pd.DataFrame(data)
    
    # Preprocess data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_data)
    
    return jsonify({
        'pca': pca_result.tolist(),
        'tsne': tsne_result.tolist(),
        'kmeans_labels': kmeans_labels.tolist(),
        'dbscan_labels': dbscan_labels.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)