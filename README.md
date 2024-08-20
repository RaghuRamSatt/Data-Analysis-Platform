# Multidimensional Data Analysis and Visualization Platform

## Overview
This project is a web-based platform for multidimensional data analysis and visualization. It allows users to upload CSV data and perform various unsupervised learning techniques, including PCA, t-SNE, K-means clustering, and DBSCAN. The results are visualized using interactive plots with user-controlled dimensions and feature selection.

## Features
- Data upload via CSV file
- Preprocessing with StandardScaler
- Dimensionality reduction: PCA and t-SNE
- Clustering: K-means and DBSCAN
- Interactive visualizations using Plotly
- User-controlled 2D and 3D plot options
- Dynamic feature selection for visualizations
- Original data visualization alongside analysis results

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
5. View the resulting plots for original data, PCA, t-SNE, K-means clustering, and DBSCAN.
6. Read the explanations below each plot to understand the analysis techniques.

## Features in Detail
- **Dynamic Plotting**: Switch between 2D and 3D visualizations for all plots.
- **Feature Selection**: Choose which features to display on each axis for original data, K-means, and DBSCAN plots.
- **Original Data Visualization**: View your raw data alongside the analysis results.
- **Interactive Plots**: Zoom, pan, and hover over data points for more information.
- **Explanations**: Each plot comes with a brief explanation of the technique used.

## Future Improvements
- Add more unsupervised learning algorithms
- Implement feature selection tools for analysis, not just visualization
- Enhance the user interface with more controls and options
- Add data export functionality
- Implement user authentication and saved analyses
- Add support for larger datasets with server-side processing
- Implement error handling and data validation for CSV uploads

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