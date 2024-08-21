# Multidimensional Data Analysis and Visualization Platform

## Overview
This project is a web-based platform for multidimensional data analysis and visualization, designed for an unsupervised machine learning class. It allows users to upload CSV data and perform various unsupervised learning techniques, including PCA, t-SNE, K-means clustering, DBSCAN, and Hierarchical clustering. The results are visualized using interactive 2D and 3D plots with user-controlled feature selection.

## Features
- Data upload via CSV file
- Preprocessing with StandardScaler and automatic feature selection
- Dimensionality reduction: PCA and t-SNE
- Clustering: K-means, DBSCAN, and Hierarchical clustering
- Interactive 2D and 3D visualizations using Plotly
- User-controlled feature selection for visualizations
- Original data visualization alongside analysis results
- Cluster evaluation using Silhouette Scores and Calinski-Harabasz Index
- Elbow method for determining optimal number of clusters in K-means
- Interactive parameter tuning for clustering algorithms
- Informative tooltips explaining clustering quality metrics and visualization techniques

## Technologies Used
- Backend: Flask, NumPy, Pandas, Scikit-learn
- Frontend: HTML, JavaScript, Plotly

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/RaghuRamSatt/multidimensional-data-analysis.git
   cd multidimensional-data-analysis
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```
   flask run
   ```

5. Open a web browser and navigate to `http://127.0.0.1:5000/`

## Usage
1. Upload a CSV file using the file input on the web page.
2. Click the "Process Data" button to analyze the data.
3. Use the dropdown menus to select 2D or 3D plotting and choose which features to display on each axis.
4. Click "Update Plots" to refresh the visualizations based on your selections.
5. View the resulting plots for original data, PCA, t-SNE, K-means clustering, DBSCAN, and Hierarchical clustering.
6. Read the explanations below each plot to understand the analysis techniques.
7. Compare the Silhouette Scores to evaluate the quality of different clustering methods.

## Features in Detail
- **Dynamic 2D/3D Plotting**: Switch between 2D and 3D visualizations for all plots, allowing for more comprehensive data exploration.
- **Feature Selection**: Choose which features to display on each axis for original data and clustering plots, enabling focused analysis on specific data attributes.
- **Original Data Visualization**: View your raw data alongside the analysis results to better understand the transformations applied.
- **Interactive Plots**: Zoom, pan, and hover over data points for more detailed information.
- **Multiple Clustering Techniques**: Compare results from K-means, DBSCAN, and Hierarchical clustering algorithms.
- **Dimensionality Reduction**: Visualize high-dimensional data in lower dimensions using PCA and t-SNE.
- **Cluster Evaluation**: Use Silhouette Scores to quantitatively assess the quality of clustering results.
- **Explanations**: Each plot comes with a brief explanation of the technique used, enhancing educational value.

## Clustering Quality Metrics
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. Range is from -1 to 1, where higher values indicate better-defined clusters.
- **Calinski-Harabasz Index**: Also known as the Variance Ratio Criterion, this score is defined as the ratio of the between-clusters dispersion mean and the within-cluster dispersion. Higher values indicate better-defined clusters.
- **Elbow Method**: A method used to determine the optimal number of clusters in K-means clustering. It plots the inertia (within-cluster sum of squares) as a function of the number of clusters. The "elbow" of the curve is considered the optimal number of clusters.

## User Interface
- Interactive parameter tuning for clustering algorithms
- Informative tooltips explaining clustering quality metrics and visualization techniques
- Responsive design for better usability across different devices

## Future Improvements
- Add more unsupervised learning algorithms (e.g., OPTICS, Gaussian Mixture Models)
- Implement automatic feature selection methods
- Add anomaly detection techniques (e.g., Isolation Forest, Local Outlier Factor)
- Enhance the user interface with more advanced controls and visualizations
- Implement data preprocessing options (e.g., handling missing values, feature scaling)
- Add export functionality for analysis results and visualizations
- Implement user authentication and data management features

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is open source and available under the MIT License.

Copyright (c) 2024 RaghuRamSatt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

For more details, see the [LICENSE](LICENSE) file in the project repository.