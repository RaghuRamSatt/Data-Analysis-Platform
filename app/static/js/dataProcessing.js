let globalResult;
let progressBar;
let progressStatus;

async function readCSV(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            const lines = text.split('\n').filter(line => line.trim() !== '');
            if (lines.length < 2) {
                reject(new Error('CSV file must contain at least two lines (header and data)'));
                return;
            }
            const headers = lines[0].split(',').map(header => header.trim());
            if (headers.length < 2) {
                reject(new Error('CSV file must contain at least two columns'));
                return;
            }
            const data = lines.slice(1).map(line => {
                const values = line.split(',');
                return headers.reduce((obj, header, index) => {
                    const value = values[index] ? parseFloat(values[index]) : null;
                    return { ...obj, [header]: value };
                }, {});
            });
            resolve(data);
        };
        reader.onerror = (e) => reject(new Error('Error reading file'));
        reader.readAsText(file);
    });
}

async function processData() {
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a CSV file');
        return;
    }
    
    console.log('Starting data processing...');
    loadingIndicator.style.display = 'block';
    progressBar = document.querySelector('.progress');
    progressStatus = document.getElementById('progressStatus');
    document.querySelector('.progress-container').style.display = 'block';
    processButton.disabled = true;
    
    try {
        updateProgress(10, 'Reading CSV file...');
        const data = await readCSV(file);
        
        updateProgress(30, 'Processing data...');
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data }),
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Unknown error occurred');
        }
        
        updateProgress(70, 'Generating visualizations...');
        const result = await response.json();
        globalResult = result;
        displaySelectedFeatures(result.feature_names, result.selected_features);
        populateAxisSelectors(result.selected_features);
        plotResults(result);
        updateProgress(100, 'Complete!');
    } catch (error) {
        console.error('Error processing data:', error);
        alert(`An error occurred while processing the data: ${error.message}`);
    } finally {
        setTimeout(() => {
            loadingIndicator.style.display = 'none';
            document.querySelector('.progress-container').style.display = 'none';
            processButton.disabled = false;
        }, 1000);
    }
}

async function updateClustering() {
    const n_clusters = document.getElementById('n_clusters').value;
    const eps = document.getElementById('eps').value;
    const min_samples = document.getElementById('min_samples').value;
    
    const params = {
        n_clusters: parseInt(n_clusters),
        eps: parseFloat(eps),
        min_samples: parseInt(min_samples),
        max_eps: parseFloat(document.getElementById('max_eps').value),
        gmm_components: parseInt(document.getElementById('gmm_components').value)
    };
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                data: globalResult.original_data, 
                params: params 
            }),
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Unknown error occurred');
        }
        
        const result = await response.json();
        globalResult = { ...globalResult, ...result };
        plotResults(globalResult);
    } catch (error) {
        console.error('Error updating clustering:', error);
        alert(`An error occurred while updating clustering: ${error.message}`);
    }
}

function updateProgress(percent, message) {
    if (progressBar && progressStatus) {
        progressBar.style.width = `${percent}%`;
        progressStatus.textContent = `${message} (${percent}%)`;
    }
}

function exportResults() {
    if (!globalResult) {
        alert('No results to export. Please process data first.');
        return;
    }
    
    const exportData = {
        clustering_results: {
            kmeans: globalResult.kmeans_labels,
            dbscan: globalResult.dbscan_labels,
            hierarchical: globalResult.hierarchical_labels,
            optics: globalResult.optics_labels,
            gmm: globalResult.gmm_labels
        },
        metrics: {
            silhouette_scores: {
                kmeans: globalResult.kmeans_silhouette,
                dbscan: globalResult.dbscan_silhouette,
                hierarchical: globalResult.hierarchical_silhouette
            },
            calinski_harabasz_scores: {
                kmeans: globalResult.kmeans_calinski,
                dbscan: globalResult.dbscan_calinski,
                hierarchical: globalResult.hierarchical_calinski
            }
        },
        dimensionality_reduction: {
            pca_explained_variance: globalResult.pca_explained_variance,
            elbow_scores: globalResult.elbow_scores
        },
        optics: globalResult.optics_calinski,
        gmm: globalResult.gmm_calinski
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'clustering_results.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
