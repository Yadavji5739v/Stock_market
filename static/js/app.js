// Stock Prediction App - JavaScript

class StockPredictionApp {
    constructor() {
        this.currentStock = null;
        this.stockChart = null;
        this.featureChart = null;
        this.predictionChart = null;
        this.initializeEventListeners();
        this.updateStatus('Ready', 'warning');
    }

    initializeEventListeners() {
        // Stock search
        document.getElementById('searchBtn').addEventListener('click', () => this.searchStocks());
        document.getElementById('stockSearch').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchStocks();
        });

        // Model training
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModels());
        document.getElementById('testSize').addEventListener('input', (e) => {
            document.querySelector('small').textContent = Math.round(e.target.value * 100) + '%';
        });

        // Predictions
        document.getElementById('predictBtn').addEventListener('click', () => this.makePrediction());

        // Search results
        document.getElementById('searchResults').addEventListener('click', (e) => {
            if (e.target.classList.contains('list-group-item')) {
                this.selectStock(e.target.dataset.symbol);
            }
        });
    }

    async searchStocks() {
        const query = document.getElementById('stockSearch').value.trim();
        if (!query) return;

        this.showLoading('Searching for stocks...');
        
        try {
            const response = await fetch(`/api/search_stocks?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            this.displaySearchResults(data.stocks);
        } catch (error) {
            this.showError('Search failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displaySearchResults(stocks) {
        const resultsContainer = document.getElementById('searchResults');
        resultsContainer.innerHTML = '';

        if (stocks.length === 0) {
            resultsContainer.innerHTML = '<div class="text-muted text-center p-3">No stocks found</div>';
            return;
        }

        stocks.forEach(stock => {
            const item = document.createElement('div');
            item.className = 'list-group-item';
            item.dataset.symbol = stock.symbol;
            item.innerHTML = `
                <strong>${stock.symbol}</strong><br>
                <small class="text-muted">${stock.name}</small>
            `;
            resultsContainer.appendChild(item);
        });
    }

    async selectStock(symbol) {
        this.currentStock = symbol;
        this.showLoading(`Loading data for ${symbol}...`);
        
        try {
            const response = await fetch('/api/load_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, period: '1y' })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.displayStockInfo(data);
            this.showStockChart();
            this.updateStatus(`Loaded ${symbol}`, 'success');
            
            // Show model training section
            document.getElementById('modelTraining').style.display = 'block';
            document.getElementById('stockInfo').style.display = 'block';
            
        } catch (error) {
            this.showError('Failed to load stock data: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayStockInfo(data) {
        document.getElementById('stockSymbol').textContent = data.symbol;
        document.getElementById('latestPrice').textContent = `$${data.latest_price.toFixed(2)}`;
        
        const changeClass = data.price_change >= 0 ? 'price-up' : 'price-down';
        const changeIcon = data.price_change >= 0 ? '↗' : '↘';
        document.getElementById('priceChange').innerHTML = `
            <span class="${changeClass}">
                ${changeIcon} $${Math.abs(data.price_change).toFixed(2)} (${data.price_change_pct.toFixed(2)}%)
            </span>
        `;
        
        document.getElementById('dataPoints').textContent = data.data_points;
        document.getElementById('dateRange').textContent = `${data.date_range.start} to ${data.date_range.end}`;
    }

    async showStockChart() {
        try {
            const response = await fetch('/api/stock_data');
            const data = await response.json();
            
            if (this.stockChart) {
                this.stockChart.destroy();
            }

            const ctx = document.getElementById('stockChart').getContext('2d');
            this.stockChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: [{
                        label: 'Close Price',
                        data: data.prices.close,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#fff'
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { maxTicksLimit: 10 }
                        },
                        y: {
                            grid: { color: 'rgba(0, 0, 0, 0.1)' },
                            ticks: { callback: value => '$' + value.toFixed(2) }
                        }
                    },
                    interaction: { intersect: false }
                }
            });
        } catch (error) {
            console.error('Failed to load chart data:', error);
        }
    }

    async trainModels() {
        if (!this.currentStock) {
            this.showError('Please select a stock first');
            return;
        }

        const testSize = parseFloat(document.getElementById('testSize').value);
        this.showLoading('Training models... This may take a few minutes.');
        
        try {
            const response = await fetch('/api/train_models', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ test_size: testSize, sequence_length: 60 })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.displayModelResults(data);
            this.updateStatus('Models trained successfully', 'success');
            
            // Show predictions section
            document.getElementById('predictions').style.display = 'block';
            document.getElementById('modelPerformance').style.display = 'block';
            
        } catch (error) {
            this.showError('Model training failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayModelResults(data) {
        // Display model metrics
        const metricsContainer = document.getElementById('modelMetrics');
        metricsContainer.innerHTML = '';

        Object.entries(data.evaluation_results).forEach(([modelKey, modelData]) => {
            const metricCard = document.createElement('div');
            metricCard.className = 'metric-card';
            metricCard.innerHTML = `
                <h6>${modelData.name}</h6>
                <div class="row">
                    <div class="col-6">
                        <div class="metric-value">${modelData.metrics.RMSE.toFixed(2)}</div>
                        <div class="metric-label">RMSE</div>
                    </div>
                    <div class="col-6">
                        <div class="metric-value">${modelData.metrics['R²'].toFixed(3)}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                </div>
            `;
            metricsContainer.appendChild(metricCard);
        });

        // Load feature importance
        this.loadFeatureImportance();
    }

    async loadFeatureImportance() {
        try {
            const response = await fetch('/api/feature_importance');
            const data = await response.json();
            
            if (this.featureChart) {
                this.featureChart.destroy();
            }

            const ctx = document.getElementById('featureChart').getContext('2d');
            this.featureChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.feature_importance.map(item => item.feature),
                    datasets: [{
                        label: 'Importance',
                        data: data.feature_importance.map(item => item.importance),
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: '#667eea',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#fff'
                        }
                    },
                    scales: {
                        x: { 
                            grid: { display: false },
                            ticks: { maxRotation: 45 }
                        },
                        y: { 
                            grid: { color: 'rgba(0, 0, 0, 0.1)' },
                            beginAtZero: true
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to load feature importance:', error);
        }
    }

    async makePrediction() {
        const modelName = document.getElementById('modelSelect').value;
        const daysAhead = parseInt(document.getElementById('daysAhead').value);

        this.showLoading('Making predictions...');
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName, days_ahead: daysAhead })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.displayPredictions(data);
            this.updateStatus('Predictions generated', 'success');
            
        } catch (error) {
            this.showError('Prediction failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayPredictions(data) {
        // Show predictions chart section
        document.getElementById('predictionsChart').style.display = 'block';
        
        // Get historical data for context
        const historicalDates = this.stockChart.data.labels;
        const historicalPrices = this.stockChart.data.datasets[0].data;
        
        // Prepare prediction data
        const allDates = [...historicalDates, ...data.predictions.map(p => p.date)];
        const allPrices = [...historicalPrices, ...data.predictions.map(p => p.predicted_price)];
        
        // Create prediction chart
        if (this.predictionChart) {
            this.predictionChart.destroy();
        }

        const ctx = document.getElementById('predictionChart').getContext('2d');
        this.predictionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: allDates,
                datasets: [
                    {
                        label: 'Historical Prices',
                        data: historicalPrices,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: false
                    },
                    {
                        label: 'Predictions',
                        data: [...Array(historicalPrices.length).fill(null), ...data.predictions.map(p => p.predicted_price)],
                        borderColor: '#f093fb',
                        backgroundColor: 'rgba(240, 147, 251, 0.1)',
                        borderWidth: 3,
                        borderDash: [5, 5],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { 
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff'
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { maxTicksLimit: 15 }
                    },
                    y: {
                        grid: { color: 'rgba(0, 0, 0, 0.1)' },
                        ticks: { callback: value => '$' + value.toFixed(2) }
                    }
                },
                interaction: { intersect: false }
            }
        });
    }

    showLoading(message) {
        document.getElementById('loadingMessage').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }

    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) modal.hide();
    }

    showError(message) {
        this.updateStatus(message, 'danger');
        // You could also show a toast notification here
        console.error(message);
    }

    updateStatus(message, type) {
        const indicator = document.getElementById('status-indicator');
        const icon = indicator.querySelector('.fas');
        
        // Update icon color
        icon.className = `fas fa-circle text-${type}`;
        
        // Update text
        indicator.innerHTML = icon.outerHTML + ' ' + message;
        
        // Auto-clear error messages after 5 seconds
        if (type === 'danger') {
            setTimeout(() => {
                this.updateStatus('Ready', 'warning');
            }, 5000);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.stockApp = new StockPredictionApp();
});

// Add some utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
            case 's':
                e.preventDefault();
                document.getElementById('stockSearch').focus();
                break;
            case 't':
                e.preventDefault();
                if (document.getElementById('trainBtn').disabled === false) {
                    document.getElementById('trainBtn').click();
                }
                break;
            case 'p':
                e.preventDefault();
                if (document.getElementById('predictBtn').disabled === false) {
                    document.getElementById('predictBtn').click();
                }
                break;
        }
    }
});
