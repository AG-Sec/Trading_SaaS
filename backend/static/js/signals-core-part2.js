/**
 * signals-core-part2.js - Trading signal scanner and orchestration UI functionality (continued)
 */

/**
 * Initialize charts for signal performance and details
 */
function initializeCharts() {
    // Initialize signal performance chart
    const performanceCtx = document.getElementById('signal-performance-chart').getContext('2d');
    signalPerformanceChart = new Chart(performanceCtx, {
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
        
        if (signal.signal_id && signal.signal_id in orchestrator._signal_status) {
            const status = orchestrator._signal_status[signal.signal_id];
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
