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
