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
- Real-time progress tracking during data processing
- Export functionality for analysis results in JSON format
- Interactive tooltips explaining each visualization technique
- Progress bar and status updates during analysis

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
1. Upload a CSV file using the file input on the web page.
2. Click the "Process Data" button to analyze the data.
3. Use the dropdown menus to select 2D or 3D plotting and choose which features to display on each axis.
4. Click "Update Plots" to refresh the visualizations based on your selections.
5. View the resulting plots for original data, PCA, t-SNE, K-means clustering, DBSCAN, and Hierarchical clustering.
6. Read the explanations below each plot to understand the analysis techniques.
7. Compare the Silhouette Scores to evaluate the quality of different clustering methods.
8. Export results using the "Export Results" button if needed.

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

## Performance Notes
The application automatically optimizes CPU usage by:
- Detecting available CPU cores
- Reserving one core for system operations
- Adjusting parallel processing accordingly
- Caching frequent computations

## API Endpoints
- POST `/api/process`
  - Purpose: Process data with clustering parameters
  - Input: JSON with data array and optional parameters
  - Output: Clustering results, metrics, and visualization data
  - Example:
    ```json
    {
      "data": [...],
      "params": {
        "n_clusters": 3,
        "eps": 0.5,
        "min_samples": 5
      }
    }
    ```

## Troubleshooting
- If you see CPU count warnings, set LOKY_MAX_CPU_COUNT in your .env file
- For "flask not recognized" errors, try using `python -m flask run`
- If numpy installation fails, try `pip install --upgrade pip` first
- For memory errors with large datasets, reduce the number of parallel processes
- Check browser console for JavaScript-related issues
- Ensure all required packages are installed correctly

## Future Improvements
- Add more unsupervised learning algorithms (OPTICS, GMM)
- Implement automatic feature selection methods
- Add anomaly detection techniques
- Enhance UI with advanced controls
- Add support for categorical data
- Implement real-time cluster parameter adjustment
- Add more dimensionality reduction techniques
- Implement data export in multiple formats
- Add batch processing capabilities
- Add user authentication and data management

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