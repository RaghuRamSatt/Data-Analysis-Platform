function displaySelectedFeatures(allFeatures, selectedFeatures) {
    const container = document.getElementById('selectedFeatures');
    container.innerHTML = `
        <p>Selected features for analysis: ${selectedFeatures.join(', ')}</p>
        <p>Original features: ${allFeatures.join(', ')}</p>
    `;
}

function populateAxisSelectors(features) {
    const selectors = ['x-axis', 'y-axis', 'z-axis'];
    selectors.forEach((selector, index) => {
        const select = document.getElementById(selector);
        select.innerHTML = '';
        features.forEach((feature, i) => {
            const option = document.createElement('option');
            option.value = i;
            option.text = feature;
            select.appendChild(option);
        });
        select.selectedIndex = index < features.length ? index : 0;
    });
}

function updatePlots() {
    if (globalResult) {
        plotResults(globalResult);
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
            hierarchical: globalResult.hierarchical_labels
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
        features: {
            original: globalResult.feature_names,
            selected: globalResult.selected_features
        }
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

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const processButton = document.getElementById('processButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const plotType = document.getElementById('plot-type');
    const zAxis = document.getElementById('z-axis');
    const exportButton = document.getElementById('exportButton');

    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            fileName.textContent = e.target.files[0].name;
            processButton.disabled = false;
        } else {
            fileName.textContent = '';
            processButton.disabled = true;
        }
    });

    plotType.addEventListener('change', function() {
        const is3D = this.value === '3d';
        zAxis.style.display = is3D ? 'inline' : 'none';
        if (globalResult) {
            plotResults(globalResult);
        }
    });

    // Initialize parameter tooltips
    const tooltips = document.querySelectorAll('.parameter-tooltip');
    tooltips.forEach(tooltip => {
        const infoIcon = tooltip.querySelector('.info-icon');
        if (infoIcon) {
            infoIcon.addEventListener('mouseover', function() {
                this.style.opacity = '0.7';
            });
            
            infoIcon.addEventListener('mouseout', function() {
                this.style.opacity = '1';
            });
        }
    });
});
