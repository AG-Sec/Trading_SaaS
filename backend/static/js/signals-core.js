/**
 * signals-core.js - Trading signal scanner and orchestration UI functionality
 * 
 * This file handles the UI interactions for the trading signals feature,
 * including scanning for signals, displaying results, and executing trades.
 * 
 * Combined version of all signal functionality
 */

// Global variables
let currentSignals = [];
let portfolios = [];
let selectedSignal = null;
let signalPerformanceChart = null;
let signalDetailChart = null;
let automationSettings = null;

// DOM elements
const scanSignalsBtn = document.getElementById('scan-signals-btn');
const simulateSignalsBtn = document.getElementById('simulate-signals-btn');
const automationSettingsBtn = document.getElementById('automation-settings-btn');
const portfolioSelect = document.getElementById('portfolio-select');
const timeframeSelect = document.getElementById('timeframe-select');
const assetClassesSelect = document.getElementById('asset-classes-select');
const autoExecuteSwitch = document.getElementById('auto-execute-switch');
const signalResults = document.getElementById('signal-results');
const signalMessage = document.getElementById('signal-message');
const signalCards = document.getElementById('signal-cards');
const loadingContainer = document.getElementById('loading-container');
const recentSignalsTable = document.getElementById('recent-signals-table');

// Modal elements
const automationSettingsModal = new bootstrap.Modal(document.getElementById('automation-settings-modal'));
const signalDetailsModal = new bootstrap.Modal(document.getElementById('signal-details-modal'));
const saveSettingsBtn = document.getElementById('save-settings-btn');
const resetSettingsBtn = document.getElementById('reset-settings-btn');
const executeSignalBtn = document.getElementById('execute-signal-btn');

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    initializePage();
    initializeCharts();
    setupEventListeners();
});

/**
 * Initialize the page by loading user portfolios and automation settings
 */
async function initializePage() {
    try {
        // Load user portfolios
        await loadPortfolios();
        
        // Load automation settings
        await loadAutomationSettings();
        
        // Load market regimes
        await loadMarketRegimes();
        
        // Set default date for regime update
        document.getElementById('regime-update-time').textContent = new Date().toLocaleString();
        
        console.log('Page initialized successfully');
    } catch (error) {
        console.error('Error initializing page:', error);
        showAlert('Error initializing page. Please refresh and try again.', 'danger');
    }
}

/**
 * Load user portfolios from the API
 */
async function loadPortfolios() {
    try {
        const response = await fetch('/api/v1/portfolios', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load portfolios');
        }
        
        const data = await response.json();
        portfolios = data;
        
        // Populate portfolio dropdowns
        populatePortfolioDropdowns(portfolios);
        
        return portfolios;
    } catch (error) {
        console.error('Error loading portfolios:', error);
        return [];
    }
}

/**
 * Populate portfolio selection dropdowns
 */
function populatePortfolioDropdowns(portfolios) {
    // Get portfolio select elements and check if they exist
    const portfolioSelect = document.getElementById('portfolio-select');
    const automationPortfolio = document.getElementById('automation-portfolio');
    
    // Early exit if elements don't exist
    if (!portfolioSelect && !automationPortfolio) {
        console.warn('Portfolio select elements not found');
        return;
    }
    
    // Clear existing options (with null checks)
    if (portfolioSelect) {
        portfolioSelect.innerHTML = '<option value="">Select Portfolio</option>';
    }
    
    if (automationPortfolio) {
        automationPortfolio.innerHTML = '<option value="">Select Portfolio</option>';
    
    // Add portfolio options
    portfolios.forEach(portfolio => {
        if (portfolioSelect) {
            const option1 = document.createElement('option');
            option1.value = portfolio.id;
            option1.textContent = portfolio.name;
            portfolioSelect.appendChild(option1);
        }
        
        if (automationPortfolio) {
            const option2 = document.createElement('option');
            option2.value = portfolio.id;
            option2.textContent = portfolio.name;
            automationPortfolio.appendChild(option2);
        }
    });
}

/**
 * Load automation settings from the API
 */
async function loadAutomationSettings() {
    try {
        const response = await fetch('/api/v1/orchestrator/settings', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load automation settings');
        }
        
        const data = await response.json();
        automationSettings = data.settings;
        
        // Update the form with the loaded settings
        updateAutomationSettingsForm(automationSettings);
        
        return automationSettings;
    } catch (error) {
        console.error('Error loading automation settings:', error);
        return null;
    }
}

/**
 * Update the automation settings form with the loaded settings
 */
function updateAutomationSettingsForm(settings) {
    if (!settings) return;
    
    document.getElementById('automation-enabled').checked = settings.enabled;
    
    const portfolioSelect = document.getElementById('automation-portfolio');
    if (settings.portfolio_id) {
        portfolioSelect.value = settings.portfolio_id;
    }
    
    document.getElementById('automation-frequency').value = settings.run_frequency;
    document.getElementById('max-trades').value = settings.max_trades_per_day;
    document.getElementById('risk-profile').value = settings.risk_profile;
    
    // Handle multiple select for timeframes
    const timeframeSelect = document.getElementById('preferred-timeframes');
    for (let i = 0; i < timeframeSelect.options.length; i++) {
        timeframeSelect.options[i].selected = settings.preferred_timeframes.includes(timeframeSelect.options[i].value);
    }
    
    // Handle assets as comma-separated string
    if (settings.preferred_assets && settings.preferred_assets.length > 0) {
        document.getElementById('preferred-assets').value = settings.preferred_assets.join(', ');
    }
}

/**
 * Load market regimes for different asset classes
 */
async function loadMarketRegimes() {
    try {
        const response = await fetch('/api/v1/market-regimes/summary', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load market regimes');
        }
        
        const data = await response.json();
        
        // Update regime badges
        updateRegimeBadge('stock-regime', data.stock_market || 'Unknown');
        updateRegimeBadge('crypto-regime', data.crypto_market || 'Unknown');
        updateRegimeBadge('forex-regime', data.forex_market || 'Unknown');
        updateRegimeBadge('overall-regime', data.overall || 'Unknown');
        
        return data;
    } catch (error) {
        console.error('Error loading market regimes:', error);
        return null;
    }
}

/**
 * Update a regime badge with appropriate styling
 */
function updateRegimeBadge(elementId, regimeType) {
    const badge = document.getElementById(elementId);
    if (!badge) return;
    
    badge.textContent = regimeType;
    
    // Reset classes
    badge.className = 'regime-badge';
    
    // Add appropriate class based on regime type
    switch (regimeType.toLowerCase()) {
        case 'bullish':
            badge.classList.add('regime-bullish');
            break;
        case 'bearish':
            badge.classList.add('regime-bearish');
            break;
        case 'ranging':
        case 'sideways':
            badge.classList.add('regime-sideways');
            break;
        case 'volatile':
            badge.classList.add('regime-volatile');
            break;
        case 'trending':
            badge.classList.add('regime-trending');
            break;
        default:
            badge.classList.add('regime-unknown');
    }
}
/**
 * signals-core-part2.js - Trading signal scanner and orchestration UI functionality (continued)
 */

/**
 * Initialize charts for signal performance and details
 */
function initializeCharts() {
    // Destroy existing charts if they exist to prevent 'Canvas is already in use' error
    if (window.signalPerformanceChart) {
        window.signalPerformanceChart.destroy();
    }
    
    // Initialize signal performance chart
    const performanceCtx = document.getElementById('signal-performance-chart');
    if (!performanceCtx) {
        console.warn('Performance chart canvas not found');
        return;
    }
    signalPerformanceChart = new Chart(performanceCtx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Bullish', 'Bearish', 'Sideways', 'Trending', 'Volatile'],
            datasets: [{
                label: 'Win Rate by Market Regime',
                data: [65, 45, 55, 70, 50],
                backgroundColor: [
                    'rgba(76, 175, 80, 0.6)',  // Bullish (green)
                    'rgba(244, 67, 54, 0.6)',  // Bearish (red)
                    'rgba(255, 193, 7, 0.6)',  // Sideways (yellow)
                    'rgba(33, 150, 243, 0.6)', // Trending (blue)
                    'rgba(156, 39, 176, 0.6)'  // Volatile (purple)
                ],
                borderColor: [
                    'rgba(76, 175, 80, 1)',
                    'rgba(244, 67, 54, 1)',
                    'rgba(255, 193, 7, 1)',
                    'rgba(33, 150, 243, 1)',
                    'rgba(156, 39, 176, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Win Rate (%)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Signal Performance by Market Regime'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Win Rate: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Set up event listeners for buttons and form elements
 */
function setupEventListeners() {
    // Scan for signals button
    scanSignalsBtn.addEventListener('click', async () => {
        await scanForSignals(false);
    });
    
    // Simulate signals button
    simulateSignalsBtn.addEventListener('click', async () => {
        await scanForSignals(true);
    });
    
    // Automation settings button
    automationSettingsBtn.addEventListener('click', () => {
        automationSettingsModal.show();
    });
    
    // Save settings button
    saveSettingsBtn.addEventListener('click', async () => {
        await saveAutomationSettings();
    });
    
    // Reset settings button
    resetSettingsBtn.addEventListener('click', () => {
        resetAutomationSettingsForm();
    });
    
    // Execute signal button
    executeSignalBtn.addEventListener('click', async () => {
        if (selectedSignal) {
            await executeSignal(selectedSignal);
        }
    });
}

/**
 * Scan for trading signals using the orchestrator API
 */
async function scanForSignals(isSimulation = false) {
    try {
        // Show loading indicator
        loadingContainer.classList.remove('d-none');
        signalResults.classList.add('d-none');
        
        // Get selected values
        const portfolioId = portfolioSelect.value;
        const autoExecute = autoExecuteSwitch.checked && !isSimulation;
        
        // Get selected timeframes
        const selectedTimeframes = Array.from(timeframeSelect.selectedOptions).map(option => option.value);
        
        // Get selected asset classes
        const selectedAssetClasses = Array.from(assetClassesSelect.selectedOptions).map(option => option.value);
        
        // Build request payload
        const payload = {
            portfolio_id: portfolioId ? parseInt(portfolioId) : null,
            auto_execute: autoExecute,
            timeframes: selectedTimeframes,
            asset_classes: selectedAssetClasses
        };
        
        // Determine endpoint based on simulation flag
        const endpoint = isSimulation ? '/api/v1/orchestrator/simulate' : '/api/v1/orchestrator/scan';
        
        // Make API request
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error('Failed to scan for signals');
        }
        
        const data = await response.json();
        
        // Store current signals
        currentSignals = data.signals || [];
        
        // Update UI with results
        updateSignalResults(data, isSimulation);
        
        // Hide loading indicator and show results
        loadingContainer.classList.add('d-none');
        signalResults.classList.remove('d-none');
        
        return data;
    } catch (error) {
        console.error('Error scanning for signals:', error);
        
        // Hide loading indicator and show error
        loadingContainer.classList.add('d-none');
        signalResults.classList.remove('d-none');
        
        signalMessage.className = 'alert alert-danger';
        signalMessage.textContent = `Error: ${error.message}`;
        
        return null;
    }
}

/**
 * Update the UI with signal scan results
 */
function updateSignalResults(data, isSimulation) {
    // Update message
    signalMessage.className = 'alert alert-info';
    signalMessage.textContent = data.message || 'Scan completed';
    
    // Clear existing signal cards
    signalCards.innerHTML = '';
    
    const signals = data.signals || [];
    
    if (signals.length === 0) {
        // No signals found
        signalMessage.className = 'alert alert-warning';
        signalMessage.textContent = 'No trading signals detected at this time.';
        return;
    }
    
    // Add signal cards
    signals.forEach(signal => {
        addSignalCard(signal, isSimulation);
    });
    
    // Update recent signals table
    updateRecentSignalsTable(signals);
}

/**
 * Add a signal card to the UI
 */
function addSignalCard(signal, isSimulation) {
    const card = document.createElement('div');
    card.className = 'col-md-4 mb-3';
    
    // Determine card border color based on signal type
    let borderClass = 'border-primary';
    let badgeClass = 'bg-primary';
    let signalTypeText = 'Unknown';
    
    if (signal.signal_type) {
        const signalType = signal.signal_type.toUpperCase();
        if (signalType.includes('BUY') || signalType.includes('LONG')) {
            borderClass = 'border-success';
            badgeClass = 'bg-success';
            signalTypeText = 'BUY';
        } else if (signalType.includes('SELL') || signalType.includes('SHORT')) {
            borderClass = 'border-danger';
            badgeClass = 'bg-danger';
            signalTypeText = 'SELL';
        }
    }
    
    // Format confidence as percentage
    const confidence = signal.confidence ? `${Math.round(signal.confidence)}%` : 'N/A';
    
    // Format price with appropriate decimal places
    const price = signal.price ? signal.price.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 8
    }) : 'N/A';
    
    // Create simulation metrics section if this is a simulated signal
    let simulationHtml = '';
    if (isSimulation && signal.simulated) {
        // Format win probability as percentage
        const winProb = signal.win_probability ? `${Math.round(signal.win_probability * 100)}%` : 'N/A';
        
        // Format expected return
        const expReturn = signal.expected_return ? `${signal.expected_return > 0 ? '+' : ''}${signal.expected_return.toFixed(2)}%` : 'N/A';
        
        // Format historical stats
        const stats = signal.historical_stats || {};
        
        simulationHtml = `
            <div class="simulation-results mt-3 mb-2">
                <h6 class="border-bottom pb-1">Simulation Results</h6>
                <table class="table table-sm table-bordered">
                    <tr class="${signal.expected_return > 0 ? 'table-success' : 'table-danger'}">
                        <td>Win Probability:</td>
                        <td class="text-end fw-bold">${winProb}</td>
                    </tr>
                    <tr>
                        <td>Expected Return:</td>
                        <td class="text-end fw-bold">${expReturn}</td>
                    </tr>
                    <tr>
                        <td>Avg. Duration:</td>
                        <td class="text-end">${stats.avg_duration_hours || 'N/A'} hours</td>
                    </tr>
                    <tr>
                        <td>Historical Win Rate:</td>
                        <td class="text-end">${stats.win_rate ? `${Math.round(stats.win_rate * 100)}%` : 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>Est. Profit (1% Size):</td>
                        <td class="text-end">$${stats.estimated_profit_usd || '0.00'}</td>
                    </tr>
                </table>
            </div>
        `;
    }
    
    // Create card content
    card.innerHTML = `
        <div class="card h-100 ${borderClass}" data-signal-id="${signal.signal_id || ''}">
            <div class="card-header d-flex justify-content-between align-items-center">
                <span class="fw-bold">${signal.asset || 'Unknown'}</span>
                <span class="badge ${badgeClass}">${signalTypeText}</span>
            </div>
            <div class="card-body">
                <div class="d-flex justify-content-between mb-2">
                    <small class="text-muted">${signal.timeframe || ''} Timeframe</small>
                    <small class="text-muted">${signal.asset_class || ''}</small>
                </div>
                <table class="table table-sm">
                    <tr>
                        <td>Price:</td>
                        <td class="text-end">${price}</td>
                    </tr>
                    <tr>
                        <td>Confidence:</td>
                        <td class="text-end">${confidence}</td>
                    </tr>
                    <tr>
                        <td>Regime:</td>
                        <td class="text-end">${signal.regime || 'Unknown'}</td>
                    </tr>
                </table>
                ${signal.strategy_name ? `<p class="mb-0 small">Strategy: ${signal.strategy_name}</p>` : ''}
                ${simulationHtml}
            </div>
            <div class="card-footer d-flex justify-content-between">
                <small class="text-muted">Generated: ${formatTimestamp(signal.timestamp)}</small>
                <button class="btn btn-sm btn-outline-primary view-signal-btn">View</button>
            </div>
        </div>
    `;
    
    // Add card to container
    signalCards.appendChild(card);
    
    // Add click event for viewing signal details
    const viewButton = card.querySelector('.view-signal-btn');
    viewButton.addEventListener('click', () => {
        openSignalDetailsModal(signal);
    });
}

/**
 * Update the recent signals table
 */
function updateRecentSignalsTable(signals) {
    // Clear table
    recentSignalsTable.innerHTML = '';
    
    if (signals.length === 0) {
        recentSignalsTable.innerHTML = '<tr><td colspan="5" class="text-center">No recent signals</td></tr>';
        return;
    }
    
    // Sort signals by timestamp (newest first)
    const sortedSignals = [...signals].sort((a, b) => {
        const dateA = a.timestamp ? new Date(a.timestamp) : new Date(0);
        const dateB = b.timestamp ? new Date(b.timestamp) : new Date(0);
        return dateB - dateA;
    });
    
    // Only show the most recent 10 signals
    const recentSignals = sortedSignals.slice(0, 10);
    
    // Add rows to table
    recentSignals.forEach(signal => {
        const row = document.createElement('tr');
        
        // Determine status class based on signal status
        let statusClass = 'badge bg-secondary';
        let statusText = 'New';
        
        // Check if signal has a status property directly
        if (signal.status) {
            const status = signal.status.toLowerCase();
            switch (status) {
                case 'executed':
                    statusClass = 'badge bg-success';
                    statusText = 'Executed';
                    break;
                case 'processing':
                    statusClass = 'badge bg-info';
                    statusText = 'Processing';
                    break;
                case 'error':
                    statusClass = 'badge bg-danger';
                    statusText = 'Error';
                    break;
            }
        }
        
        // Create row content
        row.innerHTML = `
            <td>${signal.asset || 'Unknown'}</td>
            <td>${signal.signal_type || 'Unknown'}</td>
            <td>${signal.timeframe || 'N/A'}</td>
            <td>${formatTimestamp(signal.timestamp)}</td>
            <td><span class="${statusClass}">${statusText}</span></td>
        `;
        
        recentSignalsTable.appendChild(row);
    });
}

/**
 * Format timestamp from ISO string to human-readable format
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'Invalid Date';
    
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
/**
 * signals-core-part3.js - Trading signal scanner and orchestration UI functionality (continued)
 */

/**
 * Open the signal details modal with information about a specific signal
 */
function openSignalDetailsModal(signal) {
    if (!signal) return;
    
    // Store selected signal
    selectedSignal = signal;
    
    // Update modal title
    const modalTitle = document.getElementById('signal-details-label');
    modalTitle.textContent = `${signal.asset || 'Unknown'} ${signal.signal_type || 'Signal'} (${signal.timeframe || 'Unknown'})`;
    
    // Populate signal details table
    const signalDetailsTable = document.getElementById('signal-details-table');
    signalDetailsTable.innerHTML = '';
    
    const signalDetails = [
        { label: 'Asset', value: signal.asset || 'Unknown' },
        { label: 'Asset Class', value: signal.asset_class || 'Unknown' },
        { label: 'Signal Type', value: signal.signal_type || 'Unknown' },
        { label: 'Timeframe', value: signal.timeframe || 'Unknown' },
        { label: 'Price', value: formatPrice(signal.price) },
        { label: 'Timestamp', value: formatFullTimestamp(signal.timestamp) },
        { label: 'Confidence', value: signal.confidence ? `${signal.confidence}%` : 'N/A' },
        { label: 'Strategy', value: signal.strategy_name || 'N/A' },
        { label: 'Market Regime', value: signal.regime || 'Unknown' }
    ];
    
    signalDetails.forEach(detail => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="fw-bold">${detail.label}:</td>
            <td>${detail.value}</td>
        `;
        signalDetailsTable.appendChild(row);
    });
    
    // Populate risk details table
    const riskDetailsTable = document.getElementById('risk-details-table');
    riskDetailsTable.innerHTML = '';
    
    const riskDetails = [
        { label: 'Position Size', value: signal.position_size ? `${signal.position_size}%` : 'N/A' },
        { label: 'Stop Loss', value: formatPrice(signal.stop_loss) },
        { label: 'Risk/Reward', value: calculateRiskReward(signal) },
        { label: 'Take Profit', value: formatPrice(signal.take_profit) },
        { label: 'Max Risk', value: signal.risk_percent ? `${signal.risk_percent}%` : '2.0%' },
        { label: 'Entry Quality', value: signal.entry_quality || 'N/A' },
        { label: 'Win Probability', value: calculateWinProbability(signal) }
    ];
    
    riskDetails.forEach(detail => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="fw-bold">${detail.label}:</td>
            <td>${detail.value}</td>
        `;
        riskDetailsTable.appendChild(row);
    });
    
    // Update execute button state based on portfolio selection
    const executeButton = document.getElementById('execute-signal-btn');
    const portfolioId = portfolioSelect.value;
    
    if (portfolioId) {
        executeButton.disabled = false;
        executeButton.textContent = `Execute in Portfolio`;
    } else {
        executeButton.disabled = true;
        executeButton.textContent = 'Select a Portfolio First';
    }
    
    // Create or update signal detail chart
    updateSignalDetailChart(signal);
    
    // Show modal
    signalDetailsModal.show();
}

/**
 * Update the signal detail chart in the modal
 */
function updateSignalDetailChart(signal) {
    // Get chart canvas
    const chartCanvas = document.getElementById('signal-detail-chart');
    
    // Destroy existing chart if it exists
    if (signalDetailChart) {
        signalDetailChart.destroy();
    }
    
    // Create mock price data based on signal
    const currentPrice = signal.price || 100;
    const stopLoss = signal.stop_loss || (currentPrice * 0.95);
    const takeProfit = signal.take_profit || (currentPrice * 1.05);
    const isBuy = (signal.signal_type || '').toUpperCase().includes('BUY') || 
                 (signal.signal_type || '').toUpperCase().includes('LONG');
    
    // Generate historical price data (mock data)
    const days = 14;
    let priceData = [];
    let basePrice = currentPrice * 0.9;
    const volatility = 0.01;
    
    for (let i = 0; i < days; i++) {
        const dailyChange = (Math.random() - 0.5) * volatility * basePrice;
        basePrice += dailyChange;
        priceData.push(basePrice);
    }
    
    // Add entry price as last point
    priceData.push(currentPrice);
    
    // Create labels for the chart (dates)
    const labels = [];
    const today = new Date();
    
    for (let i = days; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }));
    }
    
    // Create chart
    const ctx = chartCanvas.getContext('2d');
    signalDetailChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: signal.asset || 'Price',
                data: priceData,
                borderColor: isBuy ? 'rgba(76, 175, 80, 1)' : 'rgba(244, 67, 54, 1)',
                backgroundColor: 'rgba(0, 0, 0, 0)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                annotation: {
                    annotations: {
                        entryLine: {
                            type: 'line',
                            yMin: currentPrice,
                            yMax: currentPrice,
                            borderColor: 'rgba(33, 150, 243, 1)',
                            borderWidth: 2,
                            label: {
                                display: true,
                                content: 'Entry',
                                position: 'start'
                            }
                        },
                        stopLossLine: {
                            type: 'line',
                            yMin: stopLoss,
                            yMax: stopLoss,
                            borderColor: 'rgba(244, 67, 54, 1)',
                            borderWidth: 2,
                            label: {
                                display: true,
                                content: 'Stop Loss',
                                position: 'start'
                            }
                        },
                        takeProfitLine: {
                            type: 'line',
                            yMin: takeProfit,
                            yMax: takeProfit,
                            borderColor: 'rgba(76, 175, 80, 1)',
                            borderWidth: 2,
                            label: {
                                display: true,
                                content: 'Take Profit',
                                position: 'start'
                            }
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Price'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
}

/**
 * Execute a signal in a portfolio
 */
async function executeSignal(signal) {
    // Get selected portfolio
    const portfolioId = portfolioSelect.value;
    if (!portfolioId) {
        showAlert('Please select a portfolio first', 'warning');
        return;
    }
    
    try {
        // Show loading state
        const executeButton = document.getElementById('execute-signal-btn');
        const originalText = executeButton.textContent;
        executeButton.disabled = true;
        executeButton.textContent = 'Executing...';
        
        // Create a position in the portfolio based on the signal
        const response = await fetch(`/api/v1/portfolios/${portfolioId}/positions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: signal.asset,
                asset_class: signal.asset_class,
                direction: (signal.signal_type || '').toUpperCase().includes('BUY') ? 'LONG' : 'SHORT',
                quantity: calculateQuantity(signal, portfolioId),
                entry_price: signal.price,
                strategy_id: signal.strategy_id,
                stop_loss_price: signal.stop_loss,
                take_profit_price: signal.take_profit,
                max_risk_percent: signal.risk_percent || 2.0,
                notes: `Signal generated by ${signal.strategy_name || 'Trading SaaS'} on ${signal.timeframe} timeframe. Market regime: ${signal.regime || 'Unknown'}`,
                tags: [`timeframe:${signal.timeframe}`, `strategy:${signal.strategy_name || 'unknown'}`, 'auto_signal']
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to execute signal');
        }
        
        const data = await response.json();
        
        // Update signal status in the list
        if (signal.signal_id) {
            orchestrator._signal_status[signal.signal_id] = 'executed';
        }
        
        // Update recent signals table
        updateRecentSignalsTable(currentSignals);
        
        // Show success message
        showAlert(`Successfully created position for ${signal.asset}`, 'success');
        
        // Close modal
        signalDetailsModal.hide();
        
        return data;
    } catch (error) {
        console.error('Error executing signal:', error);
        showAlert(`Error: ${error.message}`, 'danger');
        
        return null;
    } finally {
        // Reset button state
        const executeButton = document.getElementById('execute-signal-btn');
        executeButton.disabled = false;
        executeButton.textContent = 'Execute in Portfolio';
    }
}

/**
 * Calculate quantity of assets to buy based on portfolio value and position size
 */
function calculateQuantity(signal, portfolioId) {
    try {
        // Find the selected portfolio
        const portfolio = portfolios.find(p => p.id.toString() === portfolioId.toString());
        if (!portfolio) return 1;
        
        const portfolioValue = portfolio.current_value || 10000;
        const price = signal.price || 100;
        const positionSizePercent = signal.position_size || 2.0;
        
        // Calculate position value based on portfolio value and position size percentage
        const positionValue = portfolioValue * (positionSizePercent / 100);
        
        // Calculate quantity
        let quantity = positionValue / price;
        
        // Round to appropriate precision based on price
        if (price < 1) {
            // For low-priced assets like some crypto, use more decimal places
            quantity = Math.floor(quantity * 100000) / 100000;
        } else if (price < 10) {
            quantity = Math.floor(quantity * 10000) / 10000;
        } else if (price < 100) {
            quantity = Math.floor(quantity * 1000) / 1000;
        } else if (price < 1000) {
            quantity = Math.floor(quantity * 100) / 100;
        } else {
            quantity = Math.floor(quantity * 10) / 10;
        }
        
        return quantity > 0 ? quantity : 1;
    } catch (error) {
        console.error('Error calculating quantity:', error);
        return 1;
    }
}

/**
 * Save automation settings to the API
 */
async function saveAutomationSettings() {
    try {
        // Get form values
        const enabled = document.getElementById('automation-enabled').checked;
        const portfolioId = document.getElementById('automation-portfolio').value;
        const frequency = document.getElementById('automation-frequency').value;
        const maxTrades = parseInt(document.getElementById('max-trades').value);
        const riskProfile = document.getElementById('risk-profile').value;
        
        // Get selected timeframes
        const timeframesSelect = document.getElementById('preferred-timeframes');
        const timeframes = Array.from(timeframesSelect.selectedOptions).map(option => option.value);
        
        // Parse preferred assets
        const assetsInput = document.getElementById('preferred-assets').value;
        const assets = assetsInput.split(',').map(asset => asset.trim()).filter(asset => asset);
        
        // Build payload
        const payload = {
            enabled,
            portfolio_id: portfolioId ? parseInt(portfolioId) : null,
            run_frequency: frequency,
            max_trades_per_day: maxTrades,
            preferred_timeframes: timeframes,
            preferred_assets: assets,
            risk_profile: riskProfile
        };
        
        // Send to API
        const response = await fetch('/api/v1/orchestrator/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error('Failed to save settings');
        }
        
        const data = await response.json();
        
        // Update stored settings
        automationSettings = data.settings;
        
        // Show success message
        showAlert('Automation settings saved successfully', 'success');
        
        // Close modal
        automationSettingsModal.hide();
        
        return data;
    } catch (error) {
        console.error('Error saving automation settings:', error);
        showAlert(`Error: ${error.message}`, 'danger');
        
        return null;
    }
}

/**
 * Reset automation settings form to defaults
 */
function resetAutomationSettingsForm() {
    document.getElementById('automation-enabled').checked = false;
    document.getElementById('automation-portfolio').value = '';
    document.getElementById('automation-frequency').value = 'daily';
    document.getElementById('max-trades').value = 3;
    document.getElementById('risk-profile').value = 'moderate';
    
    // Reset timeframes
    const timeframesSelect = document.getElementById('preferred-timeframes');
    for (let i = 0; i < timeframesSelect.options.length; i++) {
        timeframesSelect.options[i].selected = ['1h', '4h', '1d'].includes(timeframesSelect.options[i].value);
    }
    
    document.getElementById('preferred-assets').value = '';
    
    showAlert('Settings reset to defaults', 'info');
}

/**
 * Calculate risk/reward ratio from signal data
 */
function calculateRiskReward(signal) {
    if (!signal.price || !signal.stop_loss || !signal.take_profit) return 'N/A';
    
    const entryPrice = signal.price;
    const stopLoss = signal.stop_loss;
    const takeProfit = signal.take_profit;
    
    // Calculate risk and reward
    const risk = Math.abs(entryPrice - stopLoss);
    const reward = Math.abs(takeProfit - entryPrice);
    
    // Calculate ratio
    const ratio = reward / risk;
    
    return ratio.toFixed(1) + ':1';
}

/**
 * Calculate win probability based on signal and regime data
 */
function calculateWinProbability(signal) {
    if (!signal) return 'N/A';
    
    // Base probability
    let probability = 50;
    
    // Adjust based on confidence
    if (signal.confidence) {
        probability = Math.min(Math.max(signal.confidence, 20), 90);
    }
    
    // Adjust based on regime alignment
    if (signal.regime && signal.signal_type) {
        const regime = signal.regime.toLowerCase();
        const signalType = signal.signal_type.toUpperCase();
        
        // Bullish regime favors buy signals
        if (regime === 'bullish' && (signalType.includes('BUY') || signalType.includes('LONG'))) {
            probability += 10;
        }
        // Bearish regime favors sell signals
        else if (regime === 'bearish' && (signalType.includes('SELL') || signalType.includes('SHORT'))) {
            probability += 10;
        }
        // Ranging/sideways market reduces probability
        else if (regime === 'ranging' || regime === 'sideways') {
            probability -= 5;
        }
        // Volatile market reduces probability
        else if (regime === 'volatile') {
            probability -= 10;
        }
        // Regime contradicts signal
        else if ((regime === 'bullish' && (signalType.includes('SELL') || signalType.includes('SHORT'))) ||
                (regime === 'bearish' && (signalType.includes('BUY') || signalType.includes('LONG')))) {
            probability -= 15;
        }
    }
    
    // Cap probability between 10% and 95%
    probability = Math.min(Math.max(probability, 10), 95);
    
    return probability.toFixed(0) + '%';
}

/**
 * Format price with appropriate decimal places
 */
function formatPrice(price) {
    if (!price) return 'N/A';
    
    // Format with appropriate decimal places based on magnitude
    if (price < 0.01) {
        return price.toFixed(8);
    } else if (price < 1) {
        return price.toFixed(5);
    } else if (price < 10) {
        return price.toFixed(4);
    } else if (price < 1000) {
        return price.toFixed(2);
    } else {
        return price.toLocaleString(undefined, { maximumFractionDigits: 2 });
    }
}

/**
 * Format full timestamp from ISO string to human-readable format
 */
function formatFullTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'Invalid Date';
    
    return date.toLocaleString();
}

/**
 * Show a timed alert message
 */
function showAlert(message, type = 'info') {
    // Create alert if it doesn't exist
    let alertContainer = document.querySelector('.alert-container');
    
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.className = 'alert-container';
        document.body.appendChild(alertContainer);
        
        // Add some styling
        alertContainer.style.position = 'fixed';
        alertContainer.style.top = '20px';
        alertContainer.style.right = '20px';
        alertContainer.style.zIndex = '9999';
    }
    
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.role = 'alert';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add alert to container
    alertContainer.appendChild(alert);
    
    // Create Bootstrap alert and auto-dismiss after 5 seconds
    const bsAlert = new bootstrap.Alert(alert);
    setTimeout(() => {
        bsAlert.close();
    }, 5000);
}
