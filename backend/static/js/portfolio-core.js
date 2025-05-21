/**
 * Portfolio Management Core Functionality
 * Handles API interactions and UI updates for portfolio management
 */

// Global state
const portfolioState = {
    portfolios: [],
    currentPortfolio: null,
    positions: {},
    trades: {},
    metrics: {},
    charts: {}
};

// API endpoints
const API = {
    portfolios: '/api/portfolios',
    positions: (portfolioId) => `/api/portfolios/${portfolioId}/positions`,
    position: (positionId) => `/api/positions/${positionId}`,
    closePosition: (positionId) => `/api/positions/${positionId}/close`,
    trades: (portfolioId) => `/api/portfolios/${portfolioId}/trades`,
    positionTrades: (positionId) => `/api/positions/${positionId}/trades`,
    metrics: (portfolioId) => `/api/portfolios/${portfolioId}/metrics`,
    updatePrices: (portfolioId) => `/api/portfolios/${portfolioId}/update-prices`,
    importSignal: (portfolioId) => `/api/portfolios/${portfolioId}/import-signal`
};

// Initialize the portfolio dashboard
async function initializePortfolioDashboard() {
    try {
        // Show loading spinner
        document.getElementById('overviewLoading').style.display = 'block';
        
        // Load template components
        await loadTemplates();
        
        // Fetch portfolios
        await fetchPortfolios();
        
        // Hide loading spinner
        document.getElementById('overviewLoading').style.display = 'none';
        
        // Setup event listeners for the portfolio UI
        setupEventListeners();
    } catch (error) {
        console.error('Error initializing portfolio dashboard:', error);
        showAlert('Failed to initialize portfolio dashboard. Please try again.', 'danger');
        document.getElementById('overviewLoading').style.display = 'none';
    }
}

// Load template components
async function loadTemplates() {
    const templateContainer = document.getElementById('templateContainer');
    
    try {
        // Load portfolio overview component
        const overviewResponse = await fetch('/templates/portfolio_components/portfolio_overview.html');
        const overviewHtml = await overviewResponse.text();
        templateContainer.innerHTML += overviewHtml;
        
        // Load positions component
        const positionsResponse = await fetch('/templates/portfolio_components/positions.html');
        const positionsHtml = await positionsResponse.text();
        templateContainer.innerHTML += positionsHtml;
        
        // Load trade history component
        const tradeHistoryResponse = await fetch('/templates/portfolio_components/trade_history.html');
        const tradeHistoryHtml = await tradeHistoryResponse.text();
        templateContainer.innerHTML += tradeHistoryHtml;
        
        // Load performance metrics component
        const performanceResponse = await fetch('/templates/portfolio_components/performance_metrics.html');
        const performanceHtml = await performanceResponse.text();
        templateContainer.innerHTML += performanceHtml;
    } catch (error) {
        console.error('Error loading template components:', error);
        showAlert('Failed to load template components. Please refresh the page.', 'danger');
    }
}

// Fetch portfolios from the API
async function fetchPortfolios() {
    try {
        const token = localStorage.getItem('token');
        
        if (!token) {
            window.location.href = '/login';
            return;
        }
        
        const response = await fetch(API.portfolios, {
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
        portfolioState.portfolios = data;
        
        // Render portfolio overview
        renderPortfolioOverview(data);
        
        // Create tabs for each portfolio
        createPortfolioTabs(data);
        
        return data;
    } catch (error) {
        console.error('Error fetching portfolios:', error);
        showAlert('Failed to fetch portfolios. Please try again.', 'danger');
        return [];
    }
}

// Render portfolio overview cards
function renderPortfolioOverview(portfolios) {
    const portfolioCardsContainer = document.getElementById('portfolioCards');
    portfolioCardsContainer.innerHTML = '';
    
    if (portfolios.length === 0) {
        portfolioCardsContainer.innerHTML = `
            <div class="col-12 text-center py-5">
                <i class="bi bi-wallet2 display-1 text-muted"></i>
                <h3 class="mt-3">No Portfolios Found</h3>
                <p class="text-muted">Create your first portfolio to start tracking your investments.</p>
                <button class="btn btn-primary mt-3" id="createFirstPortfolioBtn">
                    <i class="bi bi-plus-circle"></i> Create Portfolio
                </button>
            </div>
        `;
        
        document.getElementById('createFirstPortfolioBtn').addEventListener('click', function() {
            const modal = new bootstrap.Modal(document.getElementById('createPortfolioModal'));
            modal.show();
        });
        
        return;
    }
    
    // Get the portfolio overview template
    const templateSource = document.getElementById('portfolioOverviewTemplate').innerHTML;
    
    // Compile the template using a simple template replacement function
    portfolios.forEach(portfolio => {
        const portfolioCard = document.createElement('div');
        portfolioCard.className = 'col-lg-6 mb-4';
        
        // Add extra properties for template
        portfolio.position_count = portfolio.positions ? portfolio.positions.length : 0;
        
        // Render the template
        portfolioCard.innerHTML = renderTemplate(templateSource, portfolio);
        portfolioCardsContainer.appendChild(portfolioCard);
        
        // Create the allocation chart
        createAllocationChart(portfolio.id);
    });
    
    // Add event listeners for the portfolio cards
    addPortfolioCardEventListeners();
}

// Create tabs for each portfolio
function createPortfolioTabs(portfolios) {
    const portfolioTabs = document.getElementById('portfolioTabs');
    const portfolioTabContent = document.getElementById('portfolioTabContent');
    
    // Remove existing portfolio tabs (keeping the overview tab)
    const existingTabs = portfolioTabs.querySelectorAll('li:not(:first-child):not(.spinner-tab)');
    existingTabs.forEach(tab => tab.remove());
    
    // Remove existing portfolio tab content (keeping the overview tab)
    const existingTabContent = portfolioTabContent.querySelectorAll('.tab-pane:not(#overview)');
    existingTabContent.forEach(content => content.remove());
    
    // Create tabs and content for each portfolio
    portfolios.forEach((portfolio, index) => {
        // Create tab
        const tabItem = document.createElement('li');
        tabItem.className = 'nav-item';
        tabItem.setAttribute('role', 'presentation');
        tabItem.innerHTML = `
            <button class="nav-link" id="portfolio-tab-${portfolio.id}" data-bs-toggle="tab" 
                data-bs-target="#portfolio-${portfolio.id}" type="button" role="tab">
                ${portfolio.name}
            </button>
        `;
        portfolioTabs.insertBefore(tabItem, portfolioTabs.querySelector('.spinner-tab'));
        
        // Create tab content
        const tabContent = document.createElement('div');
        tabContent.className = 'tab-pane fade';
        tabContent.id = `portfolio-${portfolio.id}`;
        tabContent.setAttribute('role', 'tabpanel');
        
        // Add loading spinner
        tabContent.innerHTML = `
            <div class="loading-spinner" id="portfolioLoading-${portfolio.id}">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading portfolio data...</span>
                </div>
            </div>
            
            <div class="portfolio-detail-container" id="portfolioDetail-${portfolio.id}" style="display: none;">
                <!-- Portfolio content will be loaded here -->
                <div class="row mb-4" id="positionsContainer-${portfolio.id}"></div>
                <div class="row mb-4" id="tradeHistoryContainer-${portfolio.id}"></div>
                <div class="row" id="performanceContainer-${portfolio.id}"></div>
            </div>
        `;
        
        portfolioTabContent.appendChild(tabContent);
        
        // Add event listener to load portfolio data when tab is clicked
        tabItem.querySelector('button').addEventListener('click', function() {
            loadPortfolioData(portfolio.id);
        });
    });
}

// Load portfolio data when a tab is clicked
async function loadPortfolioData(portfolioId) {
    try {
        // Show loading spinner
        document.getElementById(`portfolioLoading-${portfolioId}`).style.display = 'block';
        document.getElementById(`portfolioDetail-${portfolioId}`).style.display = 'none';
        
        // Set current portfolio
        portfolioState.currentPortfolio = portfolioId;
        
        // Fetch positions if not already loaded
        if (!portfolioState.positions[portfolioId]) {
            await fetchPositions(portfolioId);
        }
        
        // Fetch trades if not already loaded
        if (!portfolioState.trades[portfolioId]) {
            await fetchTrades(portfolioId);
        }
        
        // Fetch performance metrics if not already loaded
        if (!portfolioState.metrics[portfolioId]) {
            await fetchPerformanceMetrics(portfolioId);
        }
        
        // Render portfolio components
        renderPortfolioComponents(portfolioId);
        
        // Hide loading spinner
        document.getElementById(`portfolioLoading-${portfolioId}`).style.display = 'none';
        document.getElementById(`portfolioDetail-${portfolioId}`).style.display = 'block';
    } catch (error) {
        console.error(`Error loading portfolio data for portfolio ${portfolioId}:`, error);
        showAlert('Failed to load portfolio data. Please try again.', 'danger');
        
        // Hide loading spinner
        document.getElementById(`portfolioLoading-${portfolioId}`).style.display = 'none';
    }
}
