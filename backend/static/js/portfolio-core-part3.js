/**
 * Portfolio Management Core Functionality - Part 3
 * Handles event listeners and chart creation for the portfolio management UI
 */

// Setup event listeners for the portfolio UI
function setupEventListeners() {
    // Portfolio creation
    document.getElementById('savePortfolioBtn').addEventListener('click', createPortfolio);
    
    // Logout functionality
    document.getElementById('logoutBtn').addEventListener('click', function(e) {
        e.preventDefault();
        localStorage.removeItem('token');
        localStorage.removeItem('username');
        window.location.href = '/login';
    });
}

// Add event listeners for portfolio cards
function addPortfolioCardEventListeners() {
    // Edit portfolio buttons
    document.querySelectorAll('.edit-portfolio-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            const portfolioId = this.getAttribute('data-portfolio-id');
            showEditPortfolioModal(portfolioId);
        });
    });
    
    // Delete portfolio buttons
    document.querySelectorAll('.delete-portfolio-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            const portfolioId = this.getAttribute('data-portfolio-id');
            if (confirm('Are you sure you want to delete this portfolio? This action cannot be undone.')) {
                deletePortfolio(portfolioId);
            }
        });
    });
    
    // Update prices buttons
    document.querySelectorAll('.update-prices-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            const portfolioId = this.getAttribute('data-portfolio-id');
            updatePortfolioPrices(portfolioId);
        });
    });
}

// Add event listeners for positions
function addPositionEventListeners(portfolioId) {
    // Add position button
    document.querySelectorAll(`.add-position-btn[data-portfolio-id="${portfolioId}"]`).forEach(button => {
        button.addEventListener('click', function() {
            const modal = new bootstrap.Modal(document.getElementById(`addPositionModal-${portfolioId}`));
            modal.show();
        });
    });
    
    // View position buttons
    document.querySelectorAll('.view-position-btn').forEach(button => {
        button.addEventListener('click', function() {
            const positionId = this.getAttribute('data-position-id');
            viewPositionDetails(positionId);
        });
    });
    
    // Edit position buttons
    document.querySelectorAll('.edit-position-btn').forEach(button => {
        button.addEventListener('click', function() {
            const positionId = this.getAttribute('data-position-id');
            editPosition(positionId);
        });
    });
    
    // Close position buttons
    document.querySelectorAll('.close-position-btn').forEach(button => {
        button.addEventListener('click', function() {
            const positionId = this.getAttribute('data-position-id');
            showClosePositionModal(positionId);
        });
    });
    
    // Confirm close position button
    if (document.getElementById('confirmClosePositionBtn')) {
        document.getElementById('confirmClosePositionBtn').addEventListener('click', function() {
            const positionId = document.getElementById('closePositionId').value;
            closePosition(positionId);
        });
    }
}

// Add event listeners for trade history
function addTradeHistoryEventListeners(portfolioId) {
    // Filter trades button
    document.querySelectorAll('.filter-trades-btn').forEach(button => {
        button.addEventListener('click', function() {
            const modal = new bootstrap.Modal(document.getElementById('tradeFilterModal'));
            modal.show();
        });
    });
    
    // Export trades button
    document.querySelectorAll(`.export-trades-btn[data-portfolio-id="${portfolioId}"]`).forEach(button => {
        button.addEventListener('click', function() {
            exportTrades(portfolioId);
        });
    });
    
    // Pagination buttons
    document.querySelectorAll(`.trade-page-link[data-portfolio-id="${portfolioId}"]`).forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const page = parseInt(this.getAttribute('data-page'));
            loadTradePage(portfolioId, page);
        });
    });
    
    // Trade row click for details
    document.querySelectorAll('.trade-row').forEach(row => {
        row.addEventListener('click', function() {
            const tradeId = this.getAttribute('data-trade-id');
            showTradeDetails(tradeId);
        });
    });
}

// Add event listeners for performance metrics
function addPerformanceMetricsEventListeners(portfolioId) {
    // Time range buttons
    document.querySelectorAll('.time-range-btn').forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            document.querySelectorAll('.time-range-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Update charts with selected time range
            const timeRange = this.getAttribute('data-range');
            updateChartTimeRange(portfolioId, timeRange);
        });
    });
}

// Create a new portfolio
async function createPortfolio() {
    try {
        const token = localStorage.getItem('token');
        
        // Get form values
        const name = document.getElementById('portfolioName').value;
        const description = document.getElementById('portfolioDescription').value;
        const startingCapital = parseFloat(document.getElementById('startingCapital').value);
        
        // Validate required fields
        if (!name) {
            showAlert('Please enter a portfolio name.', 'warning');
            return;
        }
        
        // Create portfolio data
        const portfolioData = {
            name: name,
            description: description,
            starting_capital: startingCapital
        };
        
        // Submit the portfolio
        const response = await fetch(API.portfolios, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(portfolioData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('createPortfolioModal'));
        modal.hide();
        
        // Reset form
        document.getElementById('createPortfolioForm').reset();
        
        // Refresh portfolios
        await fetchPortfolios();
        
        showAlert('Portfolio created successfully.', 'success');
    } catch (error) {
        console.error('Error creating portfolio:', error);
        showAlert(`Failed to create portfolio: ${error.message}`, 'danger');
    }
}

// Show edit portfolio modal
function showEditPortfolioModal(portfolioId) {
    // Find the portfolio data
    const portfolio = portfolioState.portfolios.find(p => p.id === parseInt(portfolioId));
    
    if (!portfolio) {
        showAlert('Portfolio not found.', 'danger');
        return;
    }
    
    // Create modal if it doesn't exist
    if (!document.getElementById('editPortfolioModal')) {
        const modalHtml = `
            <div class="modal fade" id="editPortfolioModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Edit Portfolio</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <form id="editPortfolioForm">
                                <input type="hidden" id="editPortfolioId">
                                <div class="mb-3">
                                    <label for="editPortfolioName" class="form-label">Portfolio Name</label>
                                    <input type="text" class="form-control" id="editPortfolioName" required>
                                </div>
                                <div class="mb-3">
                                    <label for="editPortfolioDescription" class="form-label">Description</label>
                                    <textarea class="form-control" id="editPortfolioDescription" rows="3"></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="updatePortfolioBtn">Update Portfolio</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        const modalWrapper = document.createElement('div');
        modalWrapper.innerHTML = modalHtml;
        document.body.appendChild(modalWrapper.firstChild);
        
        // Add event listener for update button
        document.getElementById('updatePortfolioBtn').addEventListener('click', updatePortfolio);
    }
    
    // Populate form with portfolio data
    document.getElementById('editPortfolioId').value = portfolio.id;
    document.getElementById('editPortfolioName').value = portfolio.name;
    document.getElementById('editPortfolioDescription').value = portfolio.description || '';
    
    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('editPortfolioModal'));
    modal.show();
}

// Update a portfolio
async function updatePortfolio() {
    try {
        const token = localStorage.getItem('token');
        
        // Get form values
        const portfolioId = document.getElementById('editPortfolioId').value;
        const name = document.getElementById('editPortfolioName').value;
        const description = document.getElementById('editPortfolioDescription').value;
        
        // Validate required fields
        if (!name) {
            showAlert('Please enter a portfolio name.', 'warning');
            return;
        }
        
        // Create portfolio data
        const portfolioData = {
            name: name,
            description: description
        };
        
        // Submit the update
        const response = await fetch(`${API.portfolios}/${portfolioId}`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(portfolioData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('editPortfolioModal'));
        modal.hide();
        
        // Refresh portfolios
        await fetchPortfolios();
        
        showAlert('Portfolio updated successfully.', 'success');
    } catch (error) {
        console.error('Error updating portfolio:', error);
        showAlert(`Failed to update portfolio: ${error.message}`, 'danger');
    }
}

// Delete a portfolio
async function deletePortfolio(portfolioId) {
    try {
        const token = localStorage.getItem('token');
        
        // Submit the delete request
        const response = await fetch(`${API.portfolios}/${portfolioId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }
        
        // Refresh portfolios
        await fetchPortfolios();
        
        showAlert('Portfolio deleted successfully.', 'success');
    } catch (error) {
        console.error('Error deleting portfolio:', error);
        showAlert(`Failed to delete portfolio: ${error.message}`, 'danger');
    }
}

// Update portfolio prices
async function updatePortfolioPrices(portfolioId) {
    try {
        const token = localStorage.getItem('token');
        
        // Show loading indicator
        document.querySelector(`.update-prices-btn[data-portfolio-id="${portfolioId}"]`).innerHTML = '<i class="bi bi-arrow-repeat"></i> Updating...';
        
        // Submit the update request
        const response = await fetch(API.updatePrices(portfolioId), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Refresh positions if on portfolio detail page
        if (portfolioState.currentPortfolio === portfolioId) {
            await fetchPositions(portfolioId);
            renderPositions(portfolioId);
        }
        
        // Refresh portfolio overview
        await fetchPortfolios();
        
        // Restore button text
        document.querySelector(`.update-prices-btn[data-portfolio-id="${portfolioId}"]`).innerHTML = '<i class="bi bi-arrow-repeat"></i> Update Prices';
        
        showAlert('Prices updated successfully.', 'success');
    } catch (error) {
        console.error('Error updating prices:', error);
        showAlert(`Failed to update prices: ${error.message}`, 'danger');
        
        // Restore button text
        document.querySelector(`.update-prices-btn[data-portfolio-id="${portfolioId}"]`).innerHTML = '<i class="bi bi-arrow-repeat"></i> Update Prices';
    }
}

// Show close position modal
async function showClosePositionModal(positionId) {
    try {
        // Get position details
        const position = await getPositionDetails(positionId);
        
        if (!position) {
            showAlert('Failed to fetch position details.', 'danger');
            return;
        }
        
        // Populate modal with position data
        document.getElementById('closePositionId').value = position.id;
        document.getElementById('closePositionSymbol').textContent = position.symbol;
        document.getElementById('closePositionQuantity').textContent = formatNumber(position.quantity);
        document.getElementById('closePositionEntryPrice').textContent = '$' + formatCurrency(position.average_entry_price);
        document.getElementById('closePositionMarketPrice').textContent = '$' + formatCurrency(position.current_price);
        
        const plElement = document.getElementById('closePositionPL');
        plElement.textContent = '$' + formatCurrency(position.unrealized_profit_loss) + 
            ' (' + formatPercent(position.unrealized_profit_loss_percent) + '%)';
        plElement.className = position.unrealized_profit_loss > 0 ? 'text-success' : 
            position.unrealized_profit_loss < 0 ? 'text-danger' : '';
        
        // Set default exit price to current price
        document.getElementById('exitPrice').value = position.current_price;
        
        // Show the modal
        const modal = new bootstrap.Modal(document.getElementById('closePositionModal'));
        modal.show();
    } catch (error) {
        console.error('Error showing close position modal:', error);
        showAlert('Failed to open close position dialog.', 'danger');
    }
}

// Show trade details
function showTradeDetails(tradeId) {
    // Find the trade in portfolioState
    let trade = null;
    
    for (const portfolioId in portfolioState.trades) {
        const trades = portfolioState.trades[portfolioId].trades;
        const found = trades.find(t => t.id === parseInt(tradeId));
        
        if (found) {
            trade = found;
            break;
        }
    }
    
    if (!trade) {
        showAlert('Trade not found.', 'danger');
        return;
    }
    
    // Populate modal with trade data
    document.getElementById('detailSymbol').textContent = trade.symbol;
    document.getElementById('detailType').textContent = trade.status === 'CLOSED' ? 'Exit' : 'Entry';
    document.getElementById('detailDirection').textContent = trade.direction;
    document.getElementById('detailQuantity').textContent = formatNumber(trade.quantity);
    document.getElementById('detailPrice').textContent = '$' + formatCurrency(trade.price);
    document.getElementById('detailTotal').textContent = '$' + formatCurrency(trade.total_cost);
    document.getElementById('detailDateTime').textContent = formatDateTime(trade.timestamp);
    
    const statusElement = document.getElementById('detailStatus');
    statusElement.textContent = trade.status;
    statusElement.className = 'badge ' + getStatusBadgeClass(trade.status);
    
    const plElement = document.getElementById('detailPL');
    if (trade.realized_profit_loss !== null) {
        plElement.textContent = '$' + formatCurrency(trade.realized_profit_loss);
        plElement.className = trade.realized_profit_loss > 0 ? 'text-success' : 
            trade.realized_profit_loss < 0 ? 'text-danger' : '';
    } else {
        plElement.textContent = '-';
        plElement.className = '';
    }
    
    const plPercentElement = document.getElementById('detailPLPercent');
    if (trade.realized_profit_loss_percent !== null) {
        plPercentElement.textContent = formatPercent(trade.realized_profit_loss_percent) + '%';
        plPercentElement.className = trade.realized_profit_loss_percent > 0 ? 'text-success' : 
            trade.realized_profit_loss_percent < 0 ? 'text-danger' : '';
    } else {
        plPercentElement.textContent = '-';
        plPercentElement.className = '';
    }
    
    document.getElementById('detailRMultiple').textContent = trade.r_multiple || '-';
    document.getElementById('detailHoldingPeriod').textContent = trade.holding_period || '-';
    document.getElementById('detailMarketRegime').textContent = trade.market_regime || '-';
    document.getElementById('detailStrategy').textContent = trade.strategy_name || '-';
    document.getElementById('detailTags').textContent = trade.tags || '-';
    document.getElementById('detailNotes').textContent = trade.notes || '-';
    
    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('tradeDetailsModal'));
    modal.show();
}

// Create allocation chart for portfolio overview
function createAllocationChart(portfolioId) {
    // Find the portfolio
    const portfolio = portfolioState.portfolios.find(p => p.id === parseInt(portfolioId));
    
    if (!portfolio) {
        return;
    }
    
    // Create mock data if no positions data is available
    const positions = portfolioState.positions[portfolioId] || [];
    
    let data;
    let labels;
    
    if (positions.length === 0) {
        // If no positions, show cash only
        data = [portfolio.cash_balance];
        labels = ['Cash'];
    } else {
        // Calculate allocation data
        data = positions.map(pos => pos.market_value);
        data.push(portfolio.cash_balance);
        
        labels = positions.map(pos => pos.symbol);
        labels.push('Cash');
    }
    
    // Create colors
    const colors = generateChartColors(data.length);
    
    // Get chart canvas
    const chartCanvas = document.querySelector(`.allocation-chart[data-portfolio-id="${portfolioId}"]`);
    
    if (!chartCanvas) {
        return;
    }
    
    // Check if chart already exists
    if (portfolioState.charts[`allocation-${portfolioId}`]) {
        portfolioState.charts[`allocation-${portfolioId}`].destroy();
    }
    
    // Create new chart
    portfolioState.charts[`allocation-${portfolioId}`] = new Chart(chartCanvas, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 12,
                        font: {
                            size: 10
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${context.label}: $${formatCurrency(value)} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Create performance charts
function createPerformanceCharts(portfolioId, metrics) {
    // Create equity curve chart
    createEquityCurveChart(portfolioId, metrics);
    
    // Create allocation chart
    createAllocationDetailChart(portfolioId);
    
    // Create drawdown chart
    createDrawdownChart(portfolioId, metrics);
    
    // Create regime performance chart
    createRegimePerformanceChart(portfolioId, metrics);
    
    // Create trading activity chart
    createTradingActivityChart(portfolioId, metrics);
}

// Create equity curve chart
function createEquityCurveChart(portfolioId, metrics) {
    const chartCanvas = document.getElementById(`equityCurve-${portfolioId}`);
    
    if (!chartCanvas) {
        return;
    }
    
    // Check if chart already exists
    if (portfolioState.charts[`equityCurve-${portfolioId}`]) {
        portfolioState.charts[`equityCurve-${portfolioId}`].destroy();
    }
    
    // Get equity data from metrics
    const equityData = metrics.equity_curve || [];
    
    // Create labels and data arrays
    const labels = equityData.map(point => point.date);
    const portfolioValues = equityData.map(point => point.value);
    const benchmarkValues = equityData.map(point => point.benchmark_value);
    
    // Create new chart
    portfolioState.charts[`equityCurve-${portfolioId}`] = new Chart(chartCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Portfolio Value',
                    data: portfolioValues,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    fill: true,
                    tension: 0.1
                },
                {
                    label: 'Benchmark (S&P 500)',
                    data: benchmarkValues,
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `${context.dataset.label}: $${formatCurrency(value)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 8
                    }
                },
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + formatCurrency(value);
                        }
                    }
                }
            }
        }
    });
}

// Helper functions
function formatCurrency(value) {
    if (value === null || value === undefined) {
        return '0.00';
    }
    return parseFloat(value).toFixed(2).replace(/\d(?=(\d{3})+\.)/g, '$&,');
}

function formatPercent(value) {
    if (value === null || value === undefined) {
        return '0.0';
    }
    return parseFloat(value).toFixed(1);
}

function formatNumber(value) {
    if (value === null || value === undefined) {
        return '0';
    }
    
    const num = parseFloat(value);
    if (Number.isInteger(num)) {
        return num.toString();
    }
    
    // Format with up to 6 decimal places for small numbers like crypto
    if (Math.abs(num) < 1) {
        return num.toFixed(6).replace(/\.?0+$/, "");
    }
    
    return num.toFixed(2).replace(/\.?0+$/, "");
}

function formatDate(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString();
}

function formatDateTime(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function getStatusBadgeClass(status) {
    switch (status) {
        case 'OPEN':
            return 'bg-success';
        case 'CLOSED':
            return 'bg-secondary';
        case 'PENDING':
            return 'bg-warning';
        case 'CANCELED':
            return 'bg-danger';
        default:
            return 'bg-info';
    }
}

function generateChartColors(count) {
    const baseColors = [
        'rgba(75, 192, 192, 0.8)',  // Teal
        'rgba(54, 162, 235, 0.8)',  // Blue
        'rgba(255, 99, 132, 0.8)',  // Red
        'rgba(255, 206, 86, 0.8)',  // Yellow
        'rgba(153, 102, 255, 0.8)', // Purple
        'rgba(255, 159, 64, 0.8)',  // Orange
        'rgba(105, 240, 174, 0.8)', // Green
        'rgba(199, 83, 180, 0.8)'   // Pink
    ];
    
    // If we have more items than base colors, generate additional colors
    if (count <= baseColors.length) {
        return baseColors.slice(0, count);
    }
    
    const colors = [...baseColors];
    
    for (let i = baseColors.length; i < count; i++) {
        // Generate a new random color
        const r = Math.floor(Math.random() * 255);
        const g = Math.floor(Math.random() * 255);
        const b = Math.floor(Math.random() * 255);
        colors.push(`rgba(${r}, ${g}, ${b}, 0.8)`);
    }
    
    return colors;
}

// Helper function to show alerts
function showAlert(message, type = 'success') {
    const alertContainer = document.getElementById('alertContainer');
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    alertContainer.appendChild(alertDiv);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 500);
    }, 5000);
}
