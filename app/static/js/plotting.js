function plotScatter(elementId, data, title, xLabel, yLabel, zLabel, colors) {
    // Ensure data is properly formatted as arrays
    const xData = data.map(p => parseFloat(p[0]));
    const yData = data.map(p => parseFloat(p[1]));
    const zData = zLabel ? data.map(p => parseFloat(p[2])) : null;

    const trace = {
        x: xData,
        y: yData,
        mode: 'markers',
        type: zLabel ? 'scatter3d' : 'scatter',
        marker: colors ? {
            color: colors,
            colorscale: 'Viridis',
            size: 8,
            line: {
                color: 'white',
                width: 0.5
            }
        } : {
            size: 8,
            color: '#1f77b4',
            opacity: 0.7
        }
    };

    if (zLabel) {
        trace.z = zData;
    }

    const layout = {
        title: {
            text: title,
            font: { size: 16 }
        },
        width: 500,  // Set explicit width
        height: 400, // Set explicit height
        xaxis: { title: xLabel },
        yaxis: { title: yLabel },
        hovermode: 'closest',
        showlegend: false,
        margin: { l: 60, r: 60, t: 50, b: 50 }
    };

    if (zLabel) {
        layout.scene = {
            xaxis: { title: xLabel },
            yaxis: { title: yLabel },
            zaxis: { title: zLabel }
        };
        delete layout.xaxis;
        delete layout.yaxis;
    }

    Plotly.newPlot(elementId, [trace], layout);

    // Add plot explanations with tooltips
    let explanationDiv = document.querySelector(`#${elementId} .plot-explanation`);
    if (!explanationDiv) {
        let explanation = '';
        let tooltip = '';
        switch(elementId) {
            case 'pca-plot':
                explanation = 'PCA Results';
                tooltip = 'PCA reduces the dimensionality of the data while preserving as much variance as possible. The explained variance ratios are shown in the plot title.';
                break;
            case 'tsne-plot':
                explanation = 't-SNE Results';
                tooltip = 't-SNE is particularly good at preserving local structures in the data, making it useful for visualization of high-dimensional datasets.';
                break;
            case 'kmeans-plot':
                explanation = 'K-means Clustering';
                tooltip = 'Each color represents a different cluster. Points with the same color are grouped together based on their similarity.';
                break;
            case 'dbscan-plot':
                explanation = 'DBSCAN Clustering';
                tooltip = 'DBSCAN identifies clusters of varying shapes and sizes. Noise points are shown in a different color.';
                break;
            case 'hierarchical-plot':
                explanation = 'Hierarchical Clustering';
                tooltip = 'Shows clusters formed by hierarchical clustering. The dendrogram structure is reflected in the coloring.';
                break;
        }
        
        explanationDiv = document.createElement('div');
        explanationDiv.className = 'plot-explanation tooltip';
        explanationDiv.innerHTML = `
            ${explanation}
            <span class="tooltiptext">${tooltip}</span>
        `;
        document.getElementById(elementId).appendChild(explanationDiv);
    }
}

function plotElbowMethod(elbowScores) {
    const trace = {
        x: Array.from({length: elbowScores.length}, (_, i) => i + 1),
        y: elbowScores,
        mode: 'lines+markers',
        type: 'scatter',
        line: {
            color: '#2c3e50',
            width: 2
        },
        marker: {
            size: 8,
            color: '#e74c3c'
        }
    };
    
    const layout = {
        title: {
            text: 'Elbow Method for K-means',
            font: { size: 16 }
        },
        xaxis: { 
            title: 'Number of Clusters',
            gridcolor: '#e0e0e0'
        },
        yaxis: { 
            title: 'Inertia',
            gridcolor: '#e0e0e0'
        },
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: '#ffffff',
        margin: { l: 60, r: 40, t: 50, b: 50 },
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };
    
    Plotly.newPlot('elbowPlot', [trace], layout, config);
}

function plotResults(result) {
    const plotType = document.getElementById('plot-type').value;
    const xIndex = document.getElementById('x-axis').value;
    const yIndex = document.getElementById('y-axis').value;
    const zIndex = document.getElementById('z-axis').value;

    const xLabel = result.selected_features[xIndex];
    const yLabel = result.selected_features[yIndex];
    const zLabel = plotType === '3d' ? result.selected_features[zIndex] : null;

    plotScatter('original-plot', result.original_data, 'Original Data', xLabel, yLabel, zLabel);
    plotScatter('pca-plot', result.pca, 'PCA Results', 'PC1', 'PC2', plotType === '3d' ? 'PC3' : null);
    plotScatter('tsne-plot', result.tsne, 't-SNE Results', 't-SNE 1', 't-SNE 2', plotType === '3d' ? 't-SNE 3' : null);
    plotScatter('kmeans-plot', result.pca, 'K-means Clustering', 'PC1', 'PC2', plotType === '3d' ? 'PC3' : null, result.kmeans_labels);
    plotScatter('dbscan-plot', result.pca, 'DBSCAN Clustering', 'PC1', 'PC2', plotType === '3d' ? 'PC3' : null, result.dbscan_labels);
    plotScatter('hierarchical-plot', result.pca, 'Hierarchical Clustering', 'PC1', 'PC2', plotType === '3d' ? 'PC3' : null, result.hierarchical_labels);
    plotScatter('optics-plot', result.pca, 'OPTICS Clustering', 'PC1', 'PC2', plotType === '3d' ? 'PC3' : null, result.optics_labels);
    plotScatter('gmm-plot', result.pca, 'GMM Clustering', 'PC1', 'PC2', plotType === '3d' ? 'PC3' : null, result.gmm_labels);

    // Update metrics display
    document.getElementById('kmeans-silhouette').textContent = `K-means Silhouette Score: ${result.kmeans_silhouette.toFixed(3)}`;
    document.getElementById('kmeans-calinski').textContent = `K-means Calinski-Harabasz Score: ${result.kmeans_calinski.toFixed(3)}`;
    document.getElementById('dbscan-silhouette').textContent = `DBSCAN Silhouette Score: ${result.dbscan_silhouette ? result.dbscan_silhouette.toFixed(3) : 'N/A'}`;
    document.getElementById('dbscan-calinski').textContent = `DBSCAN Calinski-Harabasz Score: ${result.dbscan_calinski ? result.dbscan_calinski.toFixed(3) : 'N/A'}`;
    document.getElementById('hierarchical-silhouette').textContent = `Hierarchical Silhouette Score: ${result.hierarchical_silhouette.toFixed(3)}`;
    document.getElementById('hierarchical-calinski').textContent = `Hierarchical Calinski-Harabasz Score: ${result.hierarchical_calinski.toFixed(3)}`;
    document.getElementById('optics-silhouette').textContent = `OPTICS Silhouette Score: ${result.optics_silhouette ? result.optics_silhouette.toFixed(3) : 'N/A'}`;
    document.getElementById('optics-calinski').textContent = `OPTICS Calinski-Harabasz Score: ${result.optics_calinski ? result.optics_calinski.toFixed(3) : 'N/A'}`;
    document.getElementById('gmm-silhouette').textContent = `GMM Silhouette Score: ${result.gmm_silhouette.toFixed(3)}`;
    document.getElementById('gmm-calinski').textContent = `GMM Calinski-Harabasz Score: ${result.gmm_calinski.toFixed(3)}`;

    plotElbowMethod(result.elbow_scores);
}
