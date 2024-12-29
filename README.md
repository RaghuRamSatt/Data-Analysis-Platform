# Multidimensional Data Analysis and Visualization Platform

## Overview
This project is a web-based platform for multidimensional data analysis and visualization, designed for an unsupervised machine learning class. It allows users to upload CSV data and perform various unsupervised learning techniques, including PCA, t-SNE, and five different clustering algorithms: K-means, DBSCAN, Hierarchical clustering, OPTICS, and Gaussian Mixture Models (GMM).

## Features
- Data upload via CSV file
- Preprocessing with StandardScaler and automatic feature selection
- Dimensionality reduction: PCA and t-SNE
- Multiple clustering algorithms:
  - K-means clustering
  - DBSCAN clustering
  - Hierarchical clustering
  - OPTICS clustering
  - Gaussian Mixture Models (GMM)
- Interactive 2D and 3D visualizations using Plotly
- User-controlled feature selection for visualizations
- Original data visualization alongside analysis results
- Comprehensive cluster evaluation metrics:
  - Silhouette Scores
  - Calinski-Harabasz Index
  - Elbow method for K-means
- Interactive parameter tuning for all clustering algorithms
- Real-time progress tracking during data processing
- Export functionality for analysis results
- Advanced Statistical Analysis:
  - Cluster Stability Assessment using K-Fold Cross-validation
  - Normality Testing with Shapiro-Wilk test
  - Feature Importance Analysis using Decision Trees
- Density Distribution Visualization with Violin Plots
- Automated CPU Core Management for Performance

## Clustering Algorithms
1. **K-means**: Partitional clustering using centroids
   - Parameters: number of clusters
   - Best for: Spherical clusters of similar size

2. **DBSCAN**: Density-based clustering
   - Parameters: epsilon, minimum samples
   - Best for: Clusters of varying shapes and sizes

3. **Hierarchical**: Agglomerative clustering
   - Parameters: number of clusters
   - Best for: Nested cluster structures

4. **OPTICS**: Ordering points to identify clustering structure
   - Parameters: maximum epsilon, minimum samples
   - Best for: Varying density clusters

5. **GMM**: Gaussian Mixture Models
   - Parameters: number of components
   - Best for: Overlapping, Gaussian-distributed clusters

## Parameter Tuning Guidelines
- **K-means n_clusters**: Start with expected number of clusters (2-10)
- **DBSCAN eps**: Try values between 0.1 and 2.0
- **DBSCAN min_samples**: Usually 5-10 for medium datasets
- **OPTICS max_eps**: Similar to DBSCAN eps, but can be larger
- **GMM components**: Similar to K-means n_clusters

## Metrics Interpretation
- **Silhouette Score** (-1 to 1):
  - > 0.7: Strong structure
  - 0.5-0.7: Reasonable structure
  - < 0.5: Weak or overlapping clusters
- **Calinski-Harabasz Index**:
  - Higher values indicate better-defined clusters
  - Compare between different parameter settings

## System Requirements
- Python 3.8 or higher
- Minimum 4GB RAM recommended
- Multi-core processor recommended for optimal performance
- Modern web browser with JavaScript enabled
- Internet connection (for loading Plotly library)

## Technologies Used
- Backend: Flask, NumPy, Pandas, Scikit-learn
- Frontend: HTML, JavaScript, Plotly
- Data Processing: StandardScaler, PCA, t-SNE
- Clustering Algorithms: K-means, DBSCAN, Hierarchical Clustering

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
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory:
   ```
   SECRET_KEY=your-secret-key-here
   FLASK_DEBUG=True  # Optional, for development
   LOKY_MAX_CPU_COUNT=4  # Optional, for controlling parallel processing
   ```

5. Set the Flask environment variable:
   ```
   # On Windows
   set FLASK_APP=app
   
   # On Unix/MacOS
   export FLASK_APP=app
   ```

6. Run the Flask application:
   ```
   python -m flask run
   ```

7. Open a web browser and navigate to `http://127.0.0.1:5000/`

## Data Requirements
- CSV file format
- Numeric values only
- No missing values
- At least two columns
- First row must contain column headers
- Example datasets provided in `data/` directory

## Usage
1. Upload a CSV file using the file input
2. Set initial clustering parameters:
   - Number of clusters (K-means, GMM)
   - Epsilon and min_samples (DBSCAN)
   - Max epsilon (OPTICS)
3. Click "Process Data" to perform initial analysis
4. Adjust parameters using the interactive controls
5. Switch between 2D and 3D visualizations
6. Compare different clustering results
7. Export results as needed

## Features in Detail

### Visualization Capabilities
- **Dynamic 2D/3D Plotting**: Switch between 2D and 3D visualizations for all plots
- **Feature Selection**: Choose specific features for each axis
- **Interactive Plots**: Zoom, pan, and hover functionality
- **Real-time Updates**: Immediate visual feedback for parameter changes

### Analysis Tools
- **Original Data Visualization**: View raw data alongside analysis results
- **Multiple Clustering Techniques**: Compare different clustering approaches
- **Dimensionality Reduction**: PCA and t-SNE implementations
- **Cluster Evaluation**: Quantitative quality metrics

### User Interface
- Interactive parameter tuning
- Informative tooltips
- Responsive design
- Progress tracking
- Export functionality

## Clustering Quality Metrics
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
- **Elbow Method**: Helps determine optimal K-means cluster count

## Advanced Analysis Features

### Feature Importance Analysis
- Uses Decision Tree Classifier to interpret cluster assignments
- Provides importance scores for each feature
- Separate analysis for K-means and GMM clustering
- Visualized through horizontal bar charts

### Statistical Analysis
- Cluster Stability: K-fold cross-validation with adjusted Rand index
- Normality Testing: Shapiro-Wilk test for distribution analysis
- Density Distribution: Violin plots showing cluster distributions

### Performance Optimization
- Automatic CPU core management
- Parallel processing with worker pool
- Memory-efficient data handling
- Caching for repeated operations

## Performance Notes
The application automatically optimizes for:
- Available CPU cores
- Memory usage
- Parallel processing
- Caching of frequent computations

## API Endpoints
- POST `/api/process` Response includes:
  ```json
  {
    "feature_importance": {
      "kmeans": {
        "features": [...],
        "importance_scores": [...]
      },
      "gmm": {
        "features": [...],
        "importance_scores": [...]
      }
    },
    "cluster_stability": {
      "kmeans": float,
      "gmm": float
    },
    "normality_test": float
  }
  ```

### Feature Importance Interpretation
- K-means importance scores reflect feature relevance based on distance-based clustering
- GMM importance scores reflect feature relevance based on probability distributions
- Differences between K-means and GMM importance can provide insights into data structure
- Higher scores indicate stronger influence on cluster assignments


## Troubleshooting
- If you see CPU count warnings, set LOKY_MAX_CPU_COUNT in your .env file
- For "flask not recognized" errors, try using `python -m flask run`
- If numpy installation fails, try `pip install --upgrade pip` first
- For memory errors with large datasets, reduce the number of parallel processes
- Check browser console for JavaScript-related issues
- Ensure all required packages are installed correctly

## Future Improvements
- Add more clustering algorithms (Mean Shift, Spectral)
- Implement automatic parameter optimization
- Add cluster stability analysis
- Support for categorical data
- Real-time parameter adjustment
- Advanced visualization options
- Batch processing capabilities
- User authentication system
- Results comparison tools
- Export in multiple formats

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

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