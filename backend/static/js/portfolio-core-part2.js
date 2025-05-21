/**
 * Portfolio Management Core Functionality - Part 2
 * Handles component rendering and API interactions for positions, trades, and metrics
 */

// Fetch positions for a portfolio
async function fetchPositions(portfolioId) {
    try {
        const token = localStorage.getItem('token');
        
        const response = await fetch(API.positions(portfolioId), {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        portfolioState.positions[portfolioId] = data;
        
        return data;
    } catch (error) {
        console.error(`Error fetching positions for portfolio ${portfolioId}:`, error);
        showAlert('Failed to fetch positions. Please try again.', 'danger');
        return [];
    }
}

// Fetch trades for a portfolio
async function fetchTrades(portfolioId, limit = 50, offset = 0) {
    try {
        const token = localStorage.getItem('token');
        
        const response = await fetch(`${API.trades(portfolioId)}?limit=${limit}&offset=${offset}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        portfolioState.trades[portfolioId] = {
            trades: data,
            limit: limit,
            offset: offset,
            total_trades: data.length, // This would be replaced with actual pagination data from API
            has_next: data.length === limit,
            has_previous: offset > 0,
            next_page: offset + limit,
            prev_page: Math.max(0, offset - limit)
        };
        
        return data;
    } catch (error) {
        console.error(`Error fetching trades for portfolio ${portfolioId}:`, error);
        showAlert('Failed to fetch trade history. Please try again.', 'danger');
        return [];
    }
}

// Fetch performance metrics for a portfolio
async function fetchPerformanceMetrics(portfolioId) {
    try {
        const token = localStorage.getItem('token');
        
        const response = await fetch(API.metrics(portfolioId), {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        portfolioState.metrics[portfolioId] = data;
        
        return data;
    } catch (error) {
        console.error(`Error fetching performance metrics for portfolio ${portfolioId}:`, error);
        showAlert('Failed to fetch performance metrics. Please try again.', 'danger');
        return null;
    }
}

// Render all portfolio components
function renderPortfolioComponents(portfolioId) {
    // Render positions
    renderPositions(portfolioId);
    
    // Render trade history
    renderTradeHistory(portfolioId);
    
    // Render performance metrics
    renderPerformanceMetrics(portfolioId);
}

// Render positions component
function renderPositions(portfolioId) {
    const positionsContainer = document.getElementById(`positionsContainer-${portfolioId}`);
    const positions = portfolioState.positions[portfolioId] || [];
    
    // Get the positions template
    const templateSource = document.getElementById('positionsTemplate').innerHTML;
    
    // Add direction helper for template
    positions.forEach(position => {
        position.direction_is_long = position.direction === 'LONG';
    });
    
    // Render the template
    positionsContainer.innerHTML = renderTemplate(templateSource, {
        portfolioId: portfolioId,
        positions: positions
    });
    
    // Add event listeners for position buttons
    addPositionEventListeners(portfolioId);
    
    // Render add position modal
    renderAddPositionModal(portfolioId);
}

// Render add position modal
function renderAddPositionModal(portfolioId) {
    const modalContainer = document.getElementById('portfolioDetail-' + portfolioId);
    
    // Get the add position modal template
    const templateSource = document.getElementById('addPositionModalTemplate').innerHTML;
    
    // Render the template
    const modalHtml = renderTemplate(templateSource, { portfolioId: portfolioId });
    
    // Check if modal already exists
    if (!document.getElementById(`addPositionModal-${portfolioId}`)) {
        const modalWrapper = document.createElement('div');
        modalWrapper.innerHTML = modalHtml;
        modalContainer.appendChild(modalWrapper.firstChild);
    }
    
    // Add event listener for the save button
    document.getElementById(`savePositionBtn-${portfolioId}`).addEventListener('click', function() {
        createPosition(portfolioId);
    });
}

// Create a new position
async function createPosition(portfolioId) {
    try {
        const token = localStorage.getItem('token');
        
        // Get form values
        const symbol = document.getElementById(`symbol-${portfolioId}`).value;
        const assetClass = document.getElementById(`assetClass-${portfolioId}`).value;
        const direction = document.getElementById(`direction-${portfolioId}`).value;
        const quantity = parseFloat(document.getElementById(`quantity-${portfolioId}`).value);
        const entryPrice = parseFloat(document.getElementById(`entryPrice-${portfolioId}`).value);
        const strategyId = document.getElementById(`strategyId-${portfolioId}`).value || null;
        const stopLossPrice = document.getElementById(`stopLossPrice-${portfolioId}`).value || null;
        const takeProfitPrice = document.getElementById(`takeProfitPrice-${portfolioId}`).value || null;
        const maxRiskPercent = document.getElementById(`maxRiskPercent-${portfolioId}`).value || null;
        const notes = document.getElementById(`notes-${portfolioId}`).value || null;
        const tagsInput = document.getElementById(`tags-${portfolioId}`).value;
        const tags = tagsInput ? tagsInput.split(',').map(tag => tag.trim()) : null;
        
        // Validate required fields
        if (!symbol || !quantity || !entryPrice) {
            showAlert('Please fill in all required fields.', 'warning');
            return;
        }
        
        // Create position data
        const positionData = {
            symbol: symbol,
            asset_class: assetClass,
            direction: direction,
            quantity: quantity,
            entry_price: entryPrice,
            strategy_id: strategyId,
            stop_loss_price: stopLossPrice ? parseFloat(stopLossPrice) : null,
            take_profit_price: takeProfitPrice ? parseFloat(takeProfitPrice) : null,
            max_risk_percent: maxRiskPercent ? parseFloat(maxRiskPercent) : null,
            notes: notes,
            tags: tags
        };
        
        // Submit the position
        const response = await fetch(API.positions(portfolioId), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(positionData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById(`addPositionModal-${portfolioId}`));
        modal.hide();
        
        // Refresh positions
        await fetchPositions(portfolioId);
        renderPositions(portfolioId);
        
        // Update portfolio overview
        await fetchPortfolios();
        
        showAlert('Position created successfully.', 'success');
    } catch (error) {
        console.error('Error creating position:', error);
        showAlert(`Failed to create position: ${error.message}`, 'danger');
    }
}

// Close a position
async function closePosition(positionId) {
    try {
        const token = localStorage.getItem('token');
        
        // Get form values
        const exitPrice = parseFloat(document.getElementById('exitPrice').value);
        const quantity = document.getElementById('closeQuantity').value ? 
            parseFloat(document.getElementById('closeQuantity').value) : null;
        const notes = document.getElementById('closeNotes').value || null;
        const tagsInput = document.getElementById('closeTags').value;
        const tags = tagsInput ? tagsInput.split(',').map(tag => tag.trim()) : null;
        
        // Validate required fields
        if (!exitPrice) {
            showAlert('Please enter an exit price.', 'warning');
            return;
        }
        
        // Create close data
        const closeData = {
            exit_price: exitPrice,
            quantity: quantity,
            notes: notes,
            tags: tags
        };
        
        // Submit the close request
        const response = await fetch(API.closePosition(positionId), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(closeData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('closePositionModal'));
        modal.hide();
        
        // Get the portfolio ID from the position
        const position = await getPositionDetails(positionId);
        const portfolioId = position.portfolio_id;
        
        // Refresh positions
        await fetchPositions(portfolioId);
        renderPositions(portfolioId);
        
        // Refresh trades
        await fetchTrades(portfolioId);
        renderTradeHistory(portfolioId);
        
        // Update portfolio overview
        await fetchPortfolios();
        
        showAlert('Position closed successfully.', 'success');
    } catch (error) {
        console.error('Error closing position:', error);
        showAlert(`Failed to close position: ${error.message}`, 'danger');
    }
}

// Get position details
async function getPositionDetails(positionId) {
    try {
        const token = localStorage.getItem('token');
        
        const response = await fetch(API.position(positionId), {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error(`Error fetching position details for position ${positionId}:`, error);
        showAlert('Failed to fetch position details. Please try again.', 'danger');
        return null;
    }
}

// Render trade history component
function renderTradeHistory(portfolioId) {
    const tradeHistoryContainer = document.getElementById(`tradeHistoryContainer-${portfolioId}`);
    const tradeData = portfolioState.trades[portfolioId] || { trades: [] };
    
    // Calculate trade metrics
    const tradeMetrics = calculateTradeMetrics(tradeData.trades);
    
    // Get the trade history template
    const templateSource = document.getElementById('tradeHistoryTemplate').innerHTML;
    
    // Add direction helper for template
    tradeData.trades.forEach(trade => {
        trade.direction_is_long = trade.direction === 'LONG';
        trade.trade_type = trade.status === 'CLOSED' ? 'Exit' : 'Entry';
    });
    
    // Render the template
    tradeHistoryContainer.innerHTML = renderTemplate(templateSource, {
        portfolioId: portfolioId,
        trades: tradeData.trades,
        total_trades: tradeData.total_trades,
        has_next: tradeData.has_next,
        has_previous: tradeData.has_previous,
        next_page: tradeData.next_page,
        prev_page: tradeData.prev_page,
        trade_count: tradeMetrics.tradeCount,
        win_rate: tradeMetrics.winRate,
        avg_win: tradeMetrics.avgWin,
        avg_loss: tradeMetrics.avgLoss
    });
    
    // Add event listeners for trade history buttons
    addTradeHistoryEventListeners(portfolioId);
    
    // Render trade filter modal if not already present
    if (!document.getElementById('tradeFilterModal')) {
        const modalContainer = document.body;
        const templateSource = document.getElementById('tradeFilterModalTemplate').innerHTML;
        const modalHtml = renderTemplate(templateSource, {});
        
        const modalWrapper = document.createElement('div');
        modalWrapper.innerHTML = modalHtml;
        modalContainer.appendChild(modalWrapper.firstChild);
        
        // Add event listeners for the filter modal
        document.getElementById('applyTradeFiltersBtn').addEventListener('click', function() {
            // Apply filters (would be implemented to filter trade data)
            const modal = bootstrap.Modal.getInstance(document.getElementById('tradeFilterModal'));
            modal.hide();
            showAlert('Filters applied successfully.', 'success');
        });
        
        document.getElementById('resetTradeFiltersBtn').addEventListener('click', function() {
            // Reset filter form
            document.getElementById('tradeFilterForm').reset();
        });
    }
    
    // Render trade details modal if not already present
    if (!document.getElementById('tradeDetailsModal')) {
        const modalContainer = document.body;
        const templateSource = document.getElementById('tradeDetailsModalTemplate').innerHTML;
        const modalHtml = renderTemplate(templateSource, {});
        
        const modalWrapper = document.createElement('div');
        modalWrapper.innerHTML = modalHtml;
        modalContainer.appendChild(modalWrapper.firstChild);
    }
}

// Calculate trade metrics
function calculateTradeMetrics(trades) {
    const closedTrades = trades.filter(trade => trade.status === 'CLOSED' && trade.realized_profit_loss !== null);
    const tradeCount = closedTrades.length;
    
    if (tradeCount === 0) {
        return {
            tradeCount: 0,
            winRate: 0,
            avgWin: 0,
            avgLoss: 0
        };
    }
    
    const winningTrades = closedTrades.filter(trade => trade.realized_profit_loss > 0);
    const losingTrades = closedTrades.filter(trade => trade.realized_profit_loss <= 0);
    
    const winRate = (winningTrades.length / tradeCount) * 100;
    
    const totalWinAmount = winningTrades.reduce((sum, trade) => sum + trade.realized_profit_loss, 0);
    const totalLossAmount = losingTrades.reduce((sum, trade) => sum + trade.realized_profit_loss, 0);
    
    const avgWin = winningTrades.length > 0 ? totalWinAmount / winningTrades.length : 0;
    const avgLoss = losingTrades.length > 0 ? totalLossAmount / losingTrades.length : 0;
    
    return {
        tradeCount,
        winRate,
        avgWin,
        avgLoss
    };
}

// Render performance metrics component
function renderPerformanceMetrics(portfolioId) {
    const performanceContainer = document.getElementById(`performanceContainer-${portfolioId}`);
    const metrics = portfolioState.metrics[portfolioId] || {
        monthly_returns: [],
        regime_performance: []
    };
    
    // Get the performance metrics template
    const templateSource = document.getElementById('performanceMetricsTemplate').innerHTML;
    
    // Render the template
    performanceContainer.innerHTML = renderTemplate(templateSource, {
        portfolioId: portfolioId,
        metrics: metrics
    });
    
    // Create performance charts
    createPerformanceCharts(portfolioId, metrics);
    
    // Add event listeners for performance metrics buttons
    addPerformanceMetricsEventListeners(portfolioId);
    
    // Render strategy performance modal if not already present
    if (!document.getElementById('strategyPerformanceModal')) {
        const modalContainer = document.body;
        const templateSource = document.getElementById('strategyPerformanceModalTemplate').innerHTML;
        const modalHtml = renderTemplate(templateSource, {});
        
        const modalWrapper = document.createElement('div');
        modalWrapper.innerHTML = modalHtml;
        modalContainer.appendChild(modalWrapper.firstChild);
        
        // Add event listeners for the strategy selector
        document.getElementById('strategySelector').addEventListener('change', function() {
            // Update strategy performance charts (would be implemented)
            updateStrategyPerformanceCharts(this.value);
        });
    }
}

// Simple template rendering function
function renderTemplate(template, data) {
    let result = template;
    
    // Replace {{variable}} with data
    for (const key in data) {
        const regex = new RegExp(`{{${key}}}`, 'g');
        result = result.replace(regex, data[key]);
    }
    
    // Handle conditionals {{#if variable}} content {{/if}}
    const ifRegex = /{{#if ([^}]+)}}([\s\S]*?){{\/if}}/g;
    result = result.replace(ifRegex, (match, condition, content) => {
        const parts = condition.split('.');
        let value = data;
        
        for (const part of parts) {
            value = value?.[part];
        }
        
        return value ? content : '';
    });
    
    // Handle else conditionals {{#if variable}} content {{else}} alternative {{/if}}
    const ifElseRegex = /{{#if ([^}]+)}}([\s\S]*?){{else}}([\s\S]*?){{\/if}}/g;
    result = result.replace(ifElseRegex, (match, condition, content, alternative) => {
        const parts = condition.split('.');
        let value = data;
        
        for (const part of parts) {
            value = value?.[part];
        }
        
        return value ? content : alternative;
    });
    
    // Handle loops {{#each items}} content with {{this}} {{/each}}
    const eachRegex = /{{#each ([^}]+)}}([\s\S]*?){{\/each}}/g;
    result = result.replace(eachRegex, (match, collection, content) => {
        const parts = collection.split('.');
        let items = data;
        
        for (const part of parts) {
            items = items?.[part];
        }
        
        if (!items || !Array.isArray(items)) {
            return '';
        }
        
        return items.map(item => {
            let itemContent = content;
            
            // Replace {{this}} with the current item
            itemContent = itemContent.replace(/{{this}}/g, item);
            
            // Replace {{property}} with item.property
            for (const prop in item) {
                const regex = new RegExp(`{{${prop}}}`, 'g');
                itemContent = itemContent.replace(regex, item[prop]);
            }
            
            return itemContent;
        }).join('');
    });
    
    // Handle helper functions
    result = result.replace(/{{formatCurrency ([^}]+)}}/g, (match, value) => {
        const parts = value.split('.');
        let num = data;
        
        for (const part of parts) {
            num = num?.[part];
        }
        
        return formatCurrency(num);
    });
    
    result = result.replace(/{{formatPercent ([^}]+)}}/g, (match, value) => {
        const parts = value.split('.');
        let num = data;
        
        for (const part of parts) {
            num = num?.[part];
        }
        
        return formatPercent(num);
    });
    
    result = result.replace(/{{formatNumber ([^}]+)}}/g, (match, value) => {
        const parts = value.split('.');
        let num = data;
        
        for (const part of parts) {
            num = num?.[part];
        }
        
        return formatNumber(num);
    });
    
    result = result.replace(/{{formatDate ([^}]+)}}/g, (match, value) => {
        const parts = value.split('.');
        let dateStr = data;
        
        for (const part of parts) {
            dateStr = dateStr?.[part];
        }
        
        return formatDate(dateStr);
    });
    
    result = result.replace(/{{formatDateTime ([^}]+)}}/g, (match, value) => {
        const parts = value.split('.');
        let dateStr = data;
        
        for (const part of parts) {
            dateStr = dateStr?.[part];
        }
        
        return formatDateTime(dateStr);
    });
    
    result = result.replace(/{{profitLossClass ([^}]+)}}/g, (match, value) => {
        const parts = value.split('.');
        let num = data;
        
        for (const part of parts) {
            num = num?.[part];
        }
        
        return num > 0 ? 'positive' : num < 0 ? 'negative' : '';
    });
    
    result = result.replace(/{{statusBadgeClass ([^}]+)}}/g, (match, value) => {
        const parts = value.split('.');
        let status = data;
        
        for (const part of parts) {
            status = status?.[part];
        }
        
        return getStatusBadgeClass(status);
    });
    
    result = result.replace(/{{directionClass ([^}]+)}}/g, (match, value) => {
        const parts = value.split('.');
        let direction = data;
        
        for (const part of parts) {
            direction = direction?.[part];
        }
        
        return direction === 'LONG' ? 'bg-success' : 'bg-danger';
    });
    
    return result;
}
