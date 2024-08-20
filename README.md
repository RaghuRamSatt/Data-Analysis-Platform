# Multidimensional Data Analysis and Visualization Platform

## Overview
This project is a web-based platform for multidimensional data analysis and visualization. It allows users to upload CSV data and perform various unsupervised learning techniques, including PCA, t-SNE, K-means clustering, and DBSCAN. The results are visualized using interactive plots.

## Features
- Data upload via CSV file
- Preprocessing with StandardScaler
- Dimensionality reduction: PCA and t-SNE
- Clustering: K-means and DBSCAN
- Interactive visualizations using Plotly

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
3. View the resulting plots for PCA, t-SNE, K-means clustering, and DBSCAN.

## Future Improvements
- Add more unsupervised learning algorithms
- Implement feature selection tools
- Enhance the user interface with more controls and options
- Add data export functionality
- Implement user authentication and saved analyses

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is open source and available under the [MIT License](LICENSE).
