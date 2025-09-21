/**
 * RAGnetic Evaluation & Benchmarking Dashboard
 * Professional evaluation interface following RAGnetic design system
 */

// Global state for evaluation dashboard
let evaluationState = {
    agents: [],
    testSets: [],
    benchmarks: [],
    currentTab: 'testing',
    selectedBenchmark: null
};

// Initialize evaluation dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeEvaluationDashboard();
});

/**
 * Initialize the evaluation dashboard
 */
function initializeEvaluationDashboard() {
    console.log('Initializing evaluation dashboard...');
    
    // Set up tab navigation
    setupEvaluationTabs();
    
    // Load initial data
    loadAgents();
    loadTestSets();
    loadBenchmarksData(); // Load data without UI updates
    
    // Set up periodic refresh
    setInterval(() => {
        if (document.getElementById('evaluation-view').classList.contains('active')) {
            refreshEvaluationData();
        }
    }, 30000); // Refresh every 30 seconds
}

/**
 * Set up evaluation sub-tab navigation
 */
function setupEvaluationTabs() {
    const subTabButtons = document.querySelectorAll('.evaluation-sub-tabs .sub-tab-button');
    const subTabContents = document.querySelectorAll('.sub-tab-content');
    
    subTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetSubTab = button.getAttribute('data-sub-tab');
            
            // Update button states
            subTabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update content visibility
            subTabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${targetSubTab}-tab`) {
                    content.classList.add('active');
                }
            });
            
            evaluationState.currentTab = targetSubTab;
            
            // Load tab-specific data
            switch(targetSubTab) {
                case 'testing':
                    loadTestSets();
                    break;
                case 'benchmarking':
                    loadBenchmarks();
                    break;
            }
        });
    });
}

/**
 * Load agents for dropdowns
 */
async function loadAgents() {
    try {
        const response = await fetch(`${API_BASE_URL}/agents`, {
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            const agents = await response.json();
            evaluationState.agents = agents;
            
            // Populate agent dropdowns
            populateAgentDropdowns(agents);
        } else {
            console.error('Failed to load agents:', response.statusText);
            showToast('Failed to load agents', 'error');
        }
    } catch (error) {
        console.error('Error loading agents:', error);
        showToast('Error loading agents', 'error');
    }
}

/**
 * Populate agent dropdowns in modals
 */
function populateAgentDropdowns(agents) {
    const agentSelects = [
        'test-set-agent',
        'benchmark-agent',
        'agent-filter'
    ];
    
    agentSelects.forEach(selectId => {
        const select = document.getElementById(selectId);
        if (select) {
            // Clear existing options except the first one
            while (select.children.length > 1) {
                select.removeChild(select.lastChild);
            }
            
            // Add agent options
            agents.forEach(agent => {
                const option = document.createElement('option');
                option.value = agent.name;
                option.textContent = agent.display_name || agent.name;
                select.appendChild(option);
            });
        }
    });
}

/**
 * Load test sets
 */
async function loadTestSets() {
    const testSetsList = document.getElementById('test-sets-list');
    if (!testSetsList) return;
    
    try {
        // Show loading state
        testSetsList.innerHTML = `
            <div class="loading-placeholder">
                <div class="loading-spinner"></div>
                <p>Loading test sets...</p>
            </div>
        `;
        
        // Fetch test sets from API
        const response = await fetch(`${API_BASE_URL}/evaluate/test-sets`, {
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            evaluationState.testSets = data.test_sets;
            renderTestSets(data.test_sets);
        } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
    } catch (error) {
        console.error('Error loading test sets:', error);
        testSetsList.innerHTML = `
            <div class="error-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                </svg>
                <h3>Error Loading Test Sets</h3>
                <p>Failed to load test sets. Please try again.</p>
                <button class="btn-secondary" onclick="loadTestSets()">Retry</button>
            </div>
        `;
    }
}

/**
 * Render test sets list
 */
function renderTestSets(testSets) {
    const testSetsList = document.getElementById('test-sets-list');
    if (!testSetsList) return;
    
    if (!testSets || testSets.length === 0) {
        testSetsList.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 12l2 2 4-4"></path>
                    <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"></path>
                    <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"></path>
                    <path d="M12 3c0 1-1 3-3 3s-3-2-3-3 1-3 3-3 3 2 3 3"></path>
                    <path d="M12 21c0-1 1-3 3-3s3 2 3 3-1 3-3 3-3-2-3-3"></path>
                </svg>
                <h3>No Test Sets Found</h3>
                <p>Create your first test set to start evaluating your agents</p>
                <button class="btn-primary" onclick="showCreateTestSetModal()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 5v14M5 12h14"></path>
                    </svg>
                    Create Test Set
                </button>
            </div>
        `;
        return;
    }
    
    testSetsList.innerHTML = testSets.map(testSet => `
        <div class="test-set-item" data-filename="${testSet.filename}">
            <div class="test-set-item-header">
                <h3 class="test-set-name">${testSet.display_name}</h3>
                <div class="test-set-actions">
                    <button class="btn-icon" onclick="viewTestSet('${testSet.filename}')" title="View Details">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                    </button>
                    <button class="btn-icon btn-delete" onclick="deleteTestSet('${testSet.filename}')" title="Delete Test Set">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18"></path>
                            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                            <line x1="10" y1="11" x2="10" y2="17"></line>
                            <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                    </button>
                </div>
            </div>
            <div class="test-set-info">
                <div class="info-item">
                    <span class="info-label">Questions:</span>
                    <span class="info-value">${testSet.num_questions}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Size:</span>
                    <span class="info-value">${formatFileSize(testSet.size_bytes)}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Created:</span>
                    <span class="info-value">${formatDate(testSet.created_at)}</span>
                </div>
            </div>
            <div class="test-set-footer">
                <button class="btn-secondary btn-sm" onclick="useTestSetForBenchmark('${testSet.filename}')">
                    Use for Benchmark
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * Load benchmark data from API (without UI updates)
 * Source: /evaluate/benchmarks (ensures run_id is included)
 */
async function loadBenchmarksData() {
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate/benchmarks`, {
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });

        if (response.ok) {
            const body = await response.json();
            // API returns either { benchmarks: [...] } or raw array
            const list = Array.isArray(body) ? body : (body && body.benchmarks) ? body.benchmarks : [];

            // Ensure every item has a run_id and status; fallback to derive from filename
            const normalized = list.map(item => {
                if (!item.run_id && item.filename) {
                    const nameWithoutExt = item.filename.replace('.csv', '');
                    const parts = nameWithoutExt.split('_');
                    if (parts.length >= 3) {
                        let rid = parts.slice(2).join('_');
                        if (rid.startsWith('agent_')) {
                            rid = rid.substring('agent_'.length);
                        }
                        item.run_id = rid;
                    }
                }
                // Ensure status is set - CSV files exist only when benchmarks are completed
                if (!item.status) {
                    item.status = 'completed';
                }
                return item;
            });

            evaluationState.benchmarks = normalized;
            return normalized;
        } else if (response.status === 404) {
            evaluationState.benchmarks = [];
            return [];
        } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

    } catch (error) {
        console.error('Error loading benchmarks data:', error);
        evaluationState.benchmarks = [];
        throw error;
    }
}

/**
 * Load benchmark runs (with UI updates)
 */
async function loadBenchmarks() {
    const benchmarksList = document.getElementById('benchmarks-list');
    if (!benchmarksList) return;
    
    try {
        // Show loading state
        benchmarksList.innerHTML = `
            <div class="loading-placeholder">
                <div class="loading-spinner"></div>
                <p>Loading benchmark runs...</p>
            </div>
        `;
        
        const benchmarks = await loadBenchmarksData();
        renderBenchmarks(benchmarks);
        
    } catch (error) {
        console.error('Error loading benchmarks:', error);
        benchmarksList.innerHTML = `
            <div class="error-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                </svg>
                <h3>Error Loading Benchmarks</h3>
                <p>Failed to load benchmark runs. Please try again.</p>
                <button class="btn-secondary" onclick="loadBenchmarks()">Retry</button>
            </div>
        `;
    }
}

/**
 * Render benchmark runs
 */
function renderBenchmarks(benchmarks) {
    const benchmarksList = document.getElementById('benchmarks-list');
    
    if (!benchmarks || benchmarks.length === 0) {
        benchmarksList.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg>
                <h3>No Benchmark Runs Found</h3>
                <p>Create a test set first, then run a benchmark to analyze agent performance</p>
                <div class="empty-state-actions">
                    <button class="btn-secondary" onclick="showCreateTestSetModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 5v14M5 12h14"></path>
                        </svg>
                        Create Test Set
                    </button>
                    <button class="btn-primary" onclick="showRunBenchmarkModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M9 12l2 2 4-4"></path>
                            <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"></path>
                            <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"></path>
                            <path d="M12 3c0 1-1 3-3 3s-3-2-3-3 1-3 3-3 3 2 3 3"></path>
                            <path d="M12 21c0-1 1-3 3-3s3 2 3 3-1 3-3 3-3-2-3-3"></path>
                        </svg>
                        Run Benchmark
                    </button>
                </div>
            </div>
        `;
        return;
    }
    
    benchmarksList.innerHTML = benchmarks.map(benchmark => `
        <div class="benchmark-item" onclick="viewBenchmarkResults('${benchmark.run_id}')">
            <div class="benchmark-item-header">
                <div class="benchmark-info">
                    <h4>${benchmark.agent_name}</h4>
                    <p class="benchmark-id">${benchmark.run_id}</p>
                </div>
                <div class="benchmark-actions">
                    <span class="status-badge ${getStatusClass(benchmark.status)}">
                        ${getStatusText(benchmark.status)}
                    </span>
                    ${benchmark.status === 'running' ? `
                        <button class="btn-icon btn-warning" onclick="event.stopPropagation(); cancelBenchmark('${benchmark.run_id}')" title="Cancel Benchmark">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="15" y1="9" x2="9" y2="15"></line>
                                <line x1="9" y1="9" x2="15" y2="15"></line>
                            </svg>
                        </button>
                    ` : ''}
                    <button class="btn-icon btn-delete" onclick="event.stopPropagation(); deleteBenchmark('${benchmark.run_id}')" title="Delete Benchmark">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18"></path>
                            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                            <line x1="10" y1="11" x2="10" y2="17"></line>
                            <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                    </button>
                </div>
            </div>
            <div class="benchmark-metrics">
                ${benchmark.summary_metrics ? `
                    <div class="metric-grid">
                        <div class="metric-item">
                            <span class="metric-label">MRR</span>
                            <span class="metric-value">${benchmark.summary_metrics.avg_mrr?.toFixed(3) || 'N/A'}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">nDCG@10</span>
                            <span class="metric-value">${benchmark.summary_metrics.avg_ndcg_at_10?.toFixed(3) || 'N/A'}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Faithfulness</span>
                            <span class="metric-value">${benchmark.summary_metrics.avg_faithfulness?.toFixed(3) || 'N/A'}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Relevance</span>
                            <span class="metric-value">${benchmark.summary_metrics.avg_relevance?.toFixed(3) || 'N/A'}</span>
                        </div>
                        ${benchmark.summary_metrics.safety_pass_rate !== undefined ? `
                            <div class="metric-item enterprise-metric">
                                <span class="metric-label">Safety Pass Rate</span>
                                <span class="metric-value ${benchmark.summary_metrics.safety_pass_rate >= 0.8 ? 'metric-good' : 'metric-warning'}">${(benchmark.summary_metrics.safety_pass_rate * 100).toFixed(1)}%</span>
                            </div>
                        ` : ''}
                        ${benchmark.summary_metrics.pii_leak_rate !== undefined ? `
                            <div class="metric-item enterprise-metric">
                                <span class="metric-label">PII Leak Rate</span>
                                <span class="metric-value ${benchmark.summary_metrics.pii_leak_rate <= 0.1 ? 'metric-good' : 'metric-danger'}">${(benchmark.summary_metrics.pii_leak_rate * 100).toFixed(1)}%</span>
                            </div>
                        ` : ''}
                        ${benchmark.summary_metrics.toxicity_rate !== undefined ? `
                            <div class="metric-item enterprise-metric">
                                <span class="metric-label">Toxicity Rate</span>
                                <span class="metric-value ${benchmark.summary_metrics.toxicity_rate <= 0.05 ? 'metric-good' : 'metric-warning'}">${(benchmark.summary_metrics.toxicity_rate * 100).toFixed(1)}%</span>
                            </div>
                        ` : ''}
                        ${benchmark.summary_metrics.multi_turn_items !== undefined ? `
                            <div class="metric-item enterprise-metric">
                                <span class="metric-label">Multi-turn Items</span>
                                <span class="metric-value">${benchmark.summary_metrics.multi_turn_items}</span>
                            </div>
                        ` : ''}
                    </div>
                ` : ''}
            </div>
            <div class="benchmark-footer">
                <span class="benchmark-date">${formatDate(benchmark.started_at)}</span>
                <button class="btn-text" onclick="event.stopPropagation(); viewBenchmarkResults('${benchmark.run_id}')">
                    View Details
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 18l6-6-6-6"></path>
                    </svg>
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * Load results analysis
 */
async function loadResultsAnalysis() {
    const resultsCharts = document.getElementById('results-charts');
    if (!resultsCharts) return;
    
    try {
        // Show loading state
        resultsCharts.innerHTML = `
            <div class="loading-placeholder">
                <div class="loading-spinner"></div>
                <p>Loading results analysis...</p>
            </div>
        `;
        
        // Load benchmarks for analysis
        const benchmarks = await loadBenchmarksData();
        renderResultsAnalysis(benchmarks);
        
    } catch (error) {
        console.error('Error loading results analysis:', error);
        resultsCharts.innerHTML = `
            <div class="error-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                </svg>
                <h3>Error Loading Results</h3>
                <p>Failed to load results analysis. Please try again.</p>
                <button class="btn-secondary" onclick="loadResultsAnalysis()">Retry</button>
            </div>
        `;
    }
}

/**
 * Render results analysis charts
 */
function renderResultsAnalysis(benchmarks) {
    const resultsCharts = document.getElementById('results-charts');
    
    if (!benchmarks || benchmarks.length === 0) {
        resultsCharts.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg>
                <h3>No Results to Analyze</h3>
                <p>Create a test set and run benchmarks to see performance analysis</p>
                <div class="empty-state-actions">
                    <button class="btn-secondary" onclick="showCreateTestSetModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 5v14M5 12h14"></path>
                        </svg>
                        Create Test Set
                    </button>
                    <button class="btn-primary" onclick="showRunBenchmarkModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M9 12l2 2 4-4"></path>
                            <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"></path>
                            <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"></path>
                            <path d="M12 3c0 1-1 3-3 3s-3-2-3-3 1-3 3-3 3 2 3 3"></path>
                            <path d="M12 21c0-1 1-3 3-3s3 2 3 3-1 3-3 3-3-2-3-3"></path>
                        </svg>
                        Run Benchmark
                    </button>
                </div>
            </div>
        `;
        return;
    }
    
    // Create performance comparison chart
    const performanceData = benchmarks
        .filter(b => b.summary_metrics)
        .map(b => ({
            agent: b.agent_name,
            mrr: b.summary_metrics.avg_mrr || 0,
            ndcg: b.summary_metrics.avg_ndcg_at_10 || 0,
            faithfulness: b.summary_metrics.avg_faithfulness || 0,
            relevance: b.summary_metrics.avg_relevance || 0
        }));
    
    resultsCharts.innerHTML = `
        <div class="results-analysis">
            <div class="analysis-header">
                <h3>Performance Overview</h3>
                <p>Comparison of agent performance across different metrics</p>
            </div>
            <div class="charts-grid">
                <div class="chart-container">
                    <h4>MRR Comparison</h4>
                    <div class="chart-placeholder">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="20" x2="18" y2="10"></line>
                            <line x1="12" y1="20" x2="12" y2="4"></line>
                            <line x1="6" y1="20" x2="6" y2="14"></line>
                        </svg>
                        <p>Chart visualization coming soon</p>
                    </div>
                </div>
                <div class="chart-container">
                    <h4>Faithfulness vs Relevance</h4>
                    <div class="chart-placeholder">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M8 12l2 2 4-4"></path>
                        </svg>
                        <p>Scatter plot coming soon</p>
                    </div>
                </div>
            </div>
            <div class="performance-table">
                <h4>Detailed Performance Metrics</h4>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Agent</th>
                                <th>MRR</th>
                                <th>nDCG@10</th>
                                <th>Faithfulness</th>
                                <th>Relevance</th>
                                <th>Date</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${performanceData.map(data => `
                                <tr>
                                    <td>${data.agent}</td>
                                    <td>${data.mrr.toFixed(3)}</td>
                                    <td>${data.ndcg.toFixed(3)}</td>
                                    <td>${data.faithfulness.toFixed(3)}</td>
                                    <td>${data.relevance.toFixed(3)}</td>
                                    <td>${formatDate(benchmarks.find(b => b.agent_name === data.agent)?.started_at)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

// Modal Functions

/**
 * Show create test set modal
 */
function showCreateTestSetModal() {
    const modal = document.getElementById('create-test-set-modal');
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

/**
 * Hide create test set modal
 */
function hideCreateTestSetModal() {
    const modal = document.getElementById('create-test-set-modal');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

/**
 * Show run benchmark modal
 */
function showRunBenchmarkModal() {
    const modal = document.getElementById('run-benchmark-modal');
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    
    // Load test sets for dropdown
    loadTestSetsForBenchmark();
}

/**
 * Hide run benchmark modal
 */
function hideRunBenchmarkModal() {
    const modal = document.getElementById('run-benchmark-modal');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

/**
 * Show benchmark results modal
 */
function showBenchmarkResultsModal() {
    const modal = document.getElementById('benchmark-results-modal');
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

/**
 * Hide benchmark results modal
 */
function hideBenchmarkResultsModal() {
    const modal = document.getElementById('benchmark-results-modal');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Action Functions

/**
 * Create test set
 */
async function createTestSet() {
    const form = document.getElementById('create-test-set-form');
    const formData = new FormData(form);
    
    const data = {
        agent_name: formData.get('agent_name'),
        num_questions: parseInt(formData.get('num_questions')),
        output_file: formData.get('output_file')
    };
    
    if (!data.agent_name) {
        showToast('Please select an agent', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate/test-set`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': loggedInUserToken
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            const result = await response.json();
            showToast(`Test set generation started. Job ID: ${result.job_id}`, 'success');
            hideCreateTestSetModal();
            form.reset();
        } else {
            const error = await response.json();
            showToast(`Failed to create test set: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error creating test set:', error);
        showToast('Error creating test set', 'error');
    }
}

/**
 * Run benchmark
 */
async function runBenchmark() {
    const form = document.getElementById('run-benchmark-form');
    const formData = new FormData(form);
    
    const data = {
        agent_name: formData.get('agent_name'),
        test_set_file: formData.get('test_set_file'),
        run_id: formData.get('run_id') || null,
        export_csv: formData.get('export_csv') || null,
        dataset_id: formData.get('dataset_id') || null,
        purpose: formData.get('purpose') || '',
        tags: formData.get('tags') ? formData.get('tags').split(',').map(t => t.trim()).filter(t => t) : []
    };
    
    if (!data.agent_name || !data.test_set_file) {
        showToast('Please select an agent and test set', 'error');
        return;
    }
    
    try {
        // First, load the test set file
        const testSetData = await loadTestSetFile(data.test_set_file);
        
        const response = await fetch(`${API_BASE_URL}/evaluate/benchmark`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': loggedInUserToken
            },
            body: JSON.stringify({
                agent_name: data.agent_name,
                test_set: testSetData,
                dataset_id: data.dataset_id,
                purpose: data.purpose,
                tags: data.tags,
                test_set_file: data.test_set_file  // Include original filename
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            showToast(`Benchmark started. Run ID: ${result.run_id}`, 'success');
            hideRunBenchmarkModal();
            form.reset();
            
            // Refresh benchmarks list
            setTimeout(() => loadBenchmarks(), 1000);
        } else {
            const error = await response.json();
            showToast(`Failed to run benchmark: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error running benchmark:', error);
        showToast('Error running benchmark', 'error');
    }
}

/**
 * Load test set file from API
 */
async function loadTestSetFile(filename) {
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate/test-sets/${filename}`, {
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            return data.data;
        } else {
            throw new Error(`Failed to load test set: ${response.statusText}`);
        }
    } catch (error) {
        console.error('Error loading test set file:', error);
        throw error;
    }
}

/**
 * View test set details
 */
async function viewTestSet(filename) {
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate/test-sets/${filename}`, {
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            showTestSetDetailsModal(data);
        } else {
            showToast('Failed to load test set details', 'error');
        }
    } catch (error) {
        console.error('Error viewing test set:', error);
        showToast('Error loading test set details', 'error');
    }
}

/**
 * Delete test set
 */
async function deleteTestSet(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate/test-sets/${filename}`, {
            method: 'DELETE',
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            showToast('Test set deleted successfully', 'success');
            loadTestSets(); // Refresh the list
        } else {
            const error = await response.json();
            showToast(`Failed to delete test set: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error deleting test set:', error);
        showToast('Error deleting test set', 'error');
    }
}

/**
 * Delete benchmark
 */
async function deleteBenchmark(runId) {
    if (!runId || runId === 'undefined') {
        console.warn('deleteBenchmark called with invalid runId:', runId, evaluationState.benchmarks);
        showToast('Unable to delete: invalid benchmark ID', 'error');
        // Try to refresh benchmarks to recover
        loadBenchmarks().catch(() => {});
        return;
    }
    if (!confirm(`Are you sure you want to delete benchmark "${runId}"? This action cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate/benchmarks/${runId}`, {
            method: 'DELETE',
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            showToast('Benchmark deleted successfully', 'success');
            loadBenchmarks(); // Refresh the list
        } else {
            const error = await response.json();
            showToast(`Failed to delete benchmark: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error deleting benchmark:', error);
        showToast('Error deleting benchmark', 'error');
    }
}

/**
 * Use test set for benchmark
 */
function useTestSetForBenchmark(filename) {
    // Set the test set in the benchmark form
    const testSetSelect = document.getElementById('benchmark-test-set');
    if (testSetSelect) {
        testSetSelect.value = filename;
    }
    
    // Show the run benchmark modal
    showRunBenchmarkModal();
    
    showToast(`Test set "${filename}" selected for benchmark`, 'success');
}

/**
 * Load test sets for benchmark dropdown
 */
async function loadTestSetsForBenchmark() {
    const select = document.getElementById('benchmark-test-set');
    if (!select) return;
    
    // Clear existing options except the first one
    while (select.children.length > 1) {
        select.removeChild(select.lastChild);
    }
    
    try {
        // Load real test sets from API
        const response = await fetch(`${API_BASE_URL}/evaluate/test-sets`, {
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            data.test_sets.forEach(testSet => {
                const option = document.createElement('option');
                option.value = testSet.filename;
                option.textContent = `${testSet.display_name} (${testSet.num_questions} questions)`;
                select.appendChild(option);
            });
        } else {
            console.error('Failed to load test sets for benchmark dropdown');
        }
    } catch (error) {
        console.error('Error loading test sets for benchmark dropdown:', error);
    }
}

/**
 * View benchmark results
 */
function viewBenchmarkResults(runId) {
    evaluationState.selectedBenchmark = runId;
    showBenchmarkResultsModal();
    
    // Load benchmark results
    loadBenchmarkResults(runId);
}

/**
 * Load benchmark results
 */
async function loadBenchmarkResults(runId) {
    const content = document.getElementById('benchmark-results-content');
    const subtitle = document.getElementById('benchmark-results-subtitle');
    
    content.innerHTML = `
        <div class="loading-state">
            <div class="loading-spinner"></div>
            <p>Loading benchmark results...</p>
        </div>
    `;
    
    try {
        // Load detailed benchmark results from the new endpoint
        const response = await fetch(`${API_BASE_URL}/evaluate/benchmarks/${runId}/results`, {
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Benchmark not found');
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const benchmark = await response.json();
        
        subtitle.textContent = `Results for ${benchmark.agent_name} - ${runId}`;
        
        // Render benchmark results
        renderBenchmarkResults(benchmark);
        
    } catch (error) {
        console.error('Error loading benchmark results:', error);
        content.innerHTML = `
            <div class="error-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                </svg>
                <h3>Error Loading Results</h3>
                <p>Failed to load benchmark results. Please try again.</p>
                <button class="btn-secondary" onclick="loadBenchmarkResults('${runId}')">Retry</button>
            </div>
        `;
    }
}

/**
 * Render benchmark results
 */
function renderBenchmarkResults(benchmark) {
    const content = document.getElementById('benchmark-results-content');
    
    const metrics = benchmark.summary_metrics || {};
    const config = benchmark.config_snapshot || {};
    const audit = config.audit || {};
    const dataset = config.dataset_meta || {};
    
    // Generate LLM insights based on metrics (fallback)
    const insights = generateBenchmarkInsights(metrics, benchmark.status);
    
    content.innerHTML = `
        <div class="benchmark-results">
            <!-- Header Section -->
            <div class="results-header">
                <div class="results-info">
                    <h3>${benchmark.agent_name}</h3>
                    <p class="run-id">Run ID: ${benchmark.run_id}</p>
                    <p class="run-date">Started: ${formatDate(benchmark.started_at)}</p>
                    ${benchmark.ended_at ? `<p class="run-date">Ended: ${formatDate(benchmark.ended_at)}</p>` : ''}
                </div>
                <div class="results-status">
                    <span class="status-badge ${getStatusClass(benchmark.status)}">
                        ${getStatusText(benchmark.status)}
                    </span>
                </div>
            </div>
            
            <!-- Executive Summary -->
            <div class="results-summary">
                <h4>Executive Summary</h4>
                <div class="summary-content">
                    <div class="summary-metrics">
                        <div class="summary-metric">
                            <span class="summary-label">Total Items</span>
                            <span class="summary-value">${benchmark.total_items || 0}</span>
                        </div>
                        <div class="summary-metric">
                            <span class="summary-label">Completed</span>
                            <span class="summary-value">${benchmark.completed_items || 0}</span>
                        </div>
                        <div class="summary-metric">
                            <span class="summary-label">Success Rate</span>
                            <span class="summary-value">${((benchmark.completed_items || 0) / (benchmark.total_items || 1) * 100).toFixed(1)}%</span>
                        </div>
                        <div class="summary-metric">
                            <span class="summary-label">Duration</span>
                            <span class="summary-value">${calculateDuration(benchmark.started_at, benchmark.ended_at)}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Performance Metrics -->
            <div class="results-metrics">
                <h4>Performance Metrics</h4>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-header">
                            <h5>Retrieval Quality</h5>
                        </div>
                        <div class="metric-values">
                            <div class="metric-row">
                                <span class="metric-label">MRR</span>
                                <span class="metric-value">${metrics.avg_mrr?.toFixed(3) || 'N/A'}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">nDCG@10</span>
                                <span class="metric-value">${metrics.avg_ndcg_at_10?.toFixed(3) || 'N/A'}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Hit@K</span>
                                <span class="metric-value">${metrics.avg_hit_at_k?.toFixed(3) || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <h5>Response Quality</h5>
                        </div>
                        <div class="metric-values">
                            <div class="metric-row">
                                <span class="metric-label">Faithfulness</span>
                                <span class="metric-value">${metrics.avg_faithfulness?.toFixed(3) || 'N/A'}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Relevance</span>
                                <span class="metric-value">${metrics.avg_relevance?.toFixed(3) || 'N/A'}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Conciseness</span>
                                <span class="metric-value">${metrics.avg_conciseness?.toFixed(3) || 'N/A'}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Coherence</span>
                                <span class="metric-value">${metrics.avg_coherence?.toFixed(3) || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <h5>Performance</h5>
                        </div>
                        <div class="metric-values">
                            <div class="metric-row">
                                <span class="metric-label">Avg Retrieval Time</span>
                                <span class="metric-value">${metrics.avg_retrieval_s?.toFixed(3) || 'N/A'}s</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Avg Generation Time</span>
                                <span class="metric-value">${metrics.avg_generation_s?.toFixed(3) || 'N/A'}s</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Total Cost</span>
                                <span class="metric-value">$${metrics.total_cost_usd?.toFixed(4) || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Enterprise Metrics -->
            ${renderEnterpriseMetrics(metrics)}
            
            <!-- LLM Generated Insights -->
            <div class="results-insights">
                <h4>AI Insights</h4>
                <div class="insights-content" id="insights-content-${benchmark.run_id}">
                    ${insights.map(insight => `
                        <div class="insight-item ${insight.type}">
                            <div class="insight-icon">
                                ${insight.type === 'positive' ? '✓' : insight.type === 'warning' ? '' : 'ℹ'}
                            </div>
                            <div class="insight-text">
                                <strong>${insight.title}</strong>
                                <p>${insight.description}</p>
                            </div>
                        </div>
                    `).join('')}
                    <div class="insights-loading" style="display: none;">
                        <div class="loading-spinner"></div>
                        <p>Generating AI insights...</p>
                    </div>
                </div>
                <button class="btn-text" onclick="generateAIInsights('${benchmark.run_id}')" style="margin-top: 12px;">
                    Generate AI Summary
                </button>
            </div>
            
            <!-- Test Dataset Information -->
            <div class="results-dataset">
                <h4>Test Dataset</h4>
                <div class="dataset-info">
                    <div class="dataset-row">
                        <span class="dataset-label">Dataset ID:</span>
                        <span class="dataset-value">${dataset.dataset_id || 'N/A'}</span>
                    </div>
                    <div class="dataset-row">
                        <span class="dataset-label">Dataset Size:</span>
                        <span class="dataset-value">${dataset.dataset_size || 'N/A'} items</span>
                    </div>
                    <div class="dataset-row">
                        <span class="dataset-label">Checksum:</span>
                        <span class="dataset-value">${dataset.dataset_checksum ? dataset.dataset_checksum.substring(0, 8) + '...' : 'N/A'}</span>
                    </div>
                    <div class="dataset-row">
                        <span class="dataset-label">Test Set File:</span>
                        <span class="dataset-value">${config.test_set_file || 'N/A'}</span>
                    </div>
                </div>
            </div>
            
            <!-- Audit Trail & Metadata -->
            <div class="results-metadata">
                <h4>Audit Trail & Metadata</h4>
                <div class="metadata-grid">
                    <div class="metadata-section">
                        <h5>Run Information</h5>
                        <div class="metadata-row">
                            <span class="metadata-label">Requested By:</span>
                            <span class="metadata-value">${audit.requested_by || 'System'}</span>
                        </div>
                        <div class="metadata-row">
                            <span class="metadata-label">Purpose:</span>
                            <span class="metadata-value">${audit.purpose || 'N/A'}</span>
                        </div>
                        <div class="metadata-row">
                            <span class="metadata-label">Tags:</span>
                            <span class="metadata-value">${audit.tags ? audit.tags.join(', ') : 'N/A'}</span>
                        </div>
                    </div>
                    
                    <div class="metadata-section">
                        <h5>Configuration</h5>
                        <div class="metadata-row">
                            <span class="metadata-label">LLM Model:</span>
                            <span class="metadata-value">${config.llm_model || 'N/A'}</span>
                        </div>
                        <div class="metadata-row">
                            <span class="metadata-label">Embedding Model:</span>
                            <span class="metadata-value">${config.embedding_model || 'N/A'}</span>
                        </div>
                        <div class="metadata-row">
                            <span class="metadata-label">Vector Store:</span>
                            <span class="metadata-value">${config.vector_store || 'N/A'}</span>
                        </div>
                        <div class="metadata-row">
                            <span class="metadata-label">Temperature:</span>
                            <span class="metadata-value">${(config.temperature !== undefined && config.temperature !== null) ? config.temperature : 'N/A'}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            ${benchmark.results && benchmark.results.length > 0 ? `
                <div class="results-details">
                    <h4>Detailed Results</h4>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Question</th>
                                    <th>Ground Truth</th>
                                    <th>Agent Response</th>
                                    <th>Faithfulness</th>
                                    <th>Relevance</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${benchmark.results.slice(0, 10).map(result => `
                                    <tr>
                                        <td>${result.prompt?.substring(0, 100)}${result.prompt?.length > 100 ? '...' : ''}</td>
                                        <td>${result.ground_truth?.substring(0, 100)}${result.ground_truth?.length > 100 ? '...' : ''}</td>
                                        <td>${result.agent_response?.substring(0, 100)}${result.agent_response?.length > 100 ? '...' : ''}</td>
                                        <td>${result.faithfulness?.toFixed(3) || 'N/A'}</td>
                                        <td>${result.relevance?.toFixed(3) || 'N/A'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                        ${benchmark.results.length > 10 ? `<p class="table-note">Showing first 10 results of ${benchmark.results.length} total</p>` : ''}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Generate benchmark insights based on metrics
 */
function generateBenchmarkInsights(metrics, status) {
    const insights = [];
    
    // Performance insights
    if (metrics.avg_mrr && metrics.avg_mrr > 0.7) {
        insights.push({
            type: 'positive',
            title: 'Excellent Retrieval Performance',
            description: `MRR of ${metrics.avg_mrr.toFixed(3)} indicates strong retrieval accuracy. The system is finding relevant documents effectively.`
        });
    } else if (metrics.avg_mrr && metrics.avg_mrr < 0.4) {
        insights.push({
            type: 'warning',
            title: 'Retrieval Performance Needs Improvement',
            description: `MRR of ${metrics.avg_mrr.toFixed(3)} suggests the retrieval system may need tuning or better document indexing.`
        });
    }
    
    // Response quality insights
    if (metrics.avg_faithfulness && metrics.avg_faithfulness > 0.8) {
        insights.push({
            type: 'positive',
            title: 'High Response Faithfulness',
            description: `Faithfulness score of ${metrics.avg_faithfulness.toFixed(3)} shows responses are well-grounded in retrieved content.`
        });
    }
    
    // Enterprise metrics insights
    if (metrics.safety_pass_rate !== undefined) {
        if (metrics.safety_pass_rate > 0.9) {
            insights.push({
                type: 'positive',
                title: 'Strong Safety Performance',
                description: `${(metrics.safety_pass_rate * 100).toFixed(1)}% safety pass rate indicates robust safety controls.`
            });
        } else if (metrics.safety_pass_rate < 0.7) {
            insights.push({
                type: 'warning',
                title: 'Safety Concerns Detected',
                description: `${(metrics.safety_pass_rate * 100).toFixed(1)}% safety pass rate suggests potential safety issues that need attention.`
            });
        }
    }
    
    if (metrics.pii_leak_rate !== undefined && metrics.pii_leak_rate > 0.05) {
        insights.push({
            type: 'warning',
            title: 'PII Leakage Detected',
            description: `${(metrics.pii_leak_rate * 100).toFixed(1)}% PII leak rate detected. Consider implementing stronger PII detection and redaction.`
        });
    }
    
    // Cost insights
    if (metrics.total_cost_usd && metrics.total_cost_usd > 1.0) {
        insights.push({
            type: 'info',
            title: 'High Cost Run',
            description: `Total cost of $${metrics.total_cost_usd.toFixed(4)} is significant. Consider optimizing model usage or reducing test set size.`
        });
    }
    
    // Status-based insights
    if (status === 'failed') {
        insights.push({
            type: 'warning',
            title: 'Benchmark Failed',
            description: 'This benchmark run failed, likely due to quality gate violations or system errors. Check the error details.'
        });
    }
    
    return insights;
}

/**
 * Render enterprise metrics section
 */
function renderEnterpriseMetrics(metrics) {
    const hasEnterpriseMetrics = metrics.safety_pass_rate !== undefined || 
                                metrics.pii_leak_rate !== undefined || 
                                metrics.toxicity_rate !== undefined ||
                                metrics.multi_turn_items !== undefined;
    
    if (!hasEnterpriseMetrics) return '';
    
    return `
        <div class="results-enterprise">
            <h4>Enterprise Metrics</h4>
            <div class="metrics-grid">
                ${metrics.safety_pass_rate !== undefined ? `
                    <div class="metric-card enterprise-metric">
                        <div class="metric-header">
                            <h5>Safety & Security</h5>
                        </div>
                        <div class="metric-values">
                            <div class="metric-row">
                                <span class="metric-label">Safety Pass Rate</span>
                                <span class="metric-value ${metrics.safety_pass_rate >= 0.8 ? 'metric-good' : 'metric-warning'}">${(metrics.safety_pass_rate * 100).toFixed(1)}%</span>
                            </div>
                            ${metrics.pii_leak_rate !== undefined ? `
                                <div class="metric-row">
                                    <span class="metric-label">PII Leak Rate</span>
                                    <span class="metric-value ${metrics.pii_leak_rate <= 0.1 ? 'metric-good' : 'metric-danger'}">${(metrics.pii_leak_rate * 100).toFixed(1)}%</span>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                ` : ''}
                
                ${metrics.toxicity_rate !== undefined ? `
                    <div class="metric-card enterprise-metric">
                        <div class="metric-header">
                            <h5>Content Quality</h5>
                        </div>
                        <div class="metric-values">
                            <div class="metric-row">
                                <span class="metric-label">Toxicity Rate</span>
                                <span class="metric-value ${metrics.toxicity_rate <= 0.05 ? 'metric-good' : 'metric-warning'}">${(metrics.toxicity_rate * 100).toFixed(1)}%</span>
                            </div>
                            ${metrics.multi_turn_items !== undefined ? `
                                <div class="metric-row">
                                    <span class="metric-label">Multi-turn Items</span>
                                    <span class="metric-value">${metrics.multi_turn_items}</span>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

/**
 * Calculate duration between two timestamps
 */
function calculateDuration(startTime, endTime) {
    if (!startTime || !endTime) return 'N/A';
    
    const start = new Date(startTime);
    const end = new Date(endTime);
    const diffMs = end - start;
    
    if (diffMs < 1000) return `${diffMs}ms`;
    if (diffMs < 60000) return `${(diffMs / 1000).toFixed(1)}s`;
    if (diffMs < 3600000) return `${(diffMs / 60000).toFixed(1)}m`;
    return `${(diffMs / 3600000).toFixed(1)}h`;
}

/**
 * Generate AI insights for a benchmark
 */
async function generateAIInsights(runId) {
    const insightsContent = document.getElementById(`insights-content-${runId}`);
    const loadingDiv = insightsContent.querySelector('.insights-loading');
    const existingInsights = insightsContent.querySelectorAll('.insight-item');
    
    // Show loading state
    loadingDiv.style.display = 'block';
    existingInsights.forEach(item => item.style.display = 'none');
    
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate/benchmarks/${runId}/insights`, {
            method: 'POST',
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            const result = await response.json();
            const insights = result.insights || [];
            
            // Hide loading and show new insights
            loadingDiv.style.display = 'none';
            existingInsights.forEach(item => item.remove());
            
            // Add new insights
            insights.forEach(insight => {
                const insightDiv = document.createElement('div');
                insightDiv.className = `insight-item ${insight.type}`;
                insightDiv.innerHTML = `
                    <div class="insight-icon">
                        ${insight.type === 'positive' ? '✓' : insight.type === 'warning' ? '' : 'ℹ'}
                    </div>
                    <div class="insight-text">
                        <strong>${insight.title}</strong>
                        <p>${insight.description}</p>
                    </div>
                `;
                insightsContent.appendChild(insightDiv);
            });
            
            showToast('AI insights generated successfully', 'success');
        } else {
            const error = await response.json();
            showToast(`Failed to generate insights: ${error.detail}`, 'error');
            
            // Show existing insights again
            loadingDiv.style.display = 'none';
            existingInsights.forEach(item => item.style.display = 'flex');
        }
    } catch (error) {
        console.error('Error generating AI insights:', error);
        showToast('Error generating AI insights', 'error');
        
        // Show existing insights again
        loadingDiv.style.display = 'none';
        existingInsights.forEach(item => item.style.display = 'flex');
    }
}

/**
 * Download benchmark results
 */
function downloadBenchmarkResults() {
    if (!evaluationState.selectedBenchmark) {
        showToast('No benchmark selected', 'error');
        return;
    }
    
    // In a real implementation, this would download the CSV file
    showToast('Download functionality coming soon', 'info');
}

// Utility Functions

/**
 * Get status class for styling
 */
function getStatusClass(status) {
    switch (status) {
        case 'completed': return 'status-completed';
        case 'running': return 'status-running';
        case 'failed': return 'status-failed';
        case 'cancelled': return 'status-aborted';
        default: return 'status-unknown';
    }
}

/**
 * Get status text
 */
function getStatusText(status) {
    switch (status) {
        case 'completed': return 'Completed';
        case 'running': return 'Running';
        case 'failed': return 'Failed';
        case 'cancelled': return 'Cancelled';
        default: return 'Unknown';
    }
}

/**
 * Format date for display
 */
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
}

/**
 * Refresh evaluation data
 */
function refreshEvaluationData() {
    switch (evaluationState.currentTab) {
        case 'testing':
            loadTestSets();
            break;
        case 'benchmarking':
            loadBenchmarks();
            break;
    }
    
    // Always refresh benchmark data in background
    loadBenchmarksData().catch(error => {
        console.error('Error refreshing benchmark data:', error);
    });
}

/**
 * Refresh test sets
 */
function refreshTestSets() {
    loadTestSets();
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Show test set details modal
 */
function showTestSetDetailsModal(testSetData) {
    const modal = document.createElement('div');
    modal.className = 'modal show';
    modal.innerHTML = `
        <div class="modal-content large">
            <div class="modal-header">
                <h2>Test Set Details</h2>
                <button class="btn-icon" onclick="closeModal(this)">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
            <div class="modal-body">
                <div class="test-set-details">
                    <div class="details-header">
                        <h3>${testSetData.filename}</h3>
                        <div class="details-meta">
                            <span class="meta-item">
                                <strong>Questions:</strong> ${testSetData.num_questions}
                            </span>
                        </div>
                    </div>
                    <div class="test-set-content">
                        <h4>Questions & Answers</h4>
                        <div class="questions-list">
                            ${testSetData.data.map((item, index) => `
                                <div class="question-item">
                                    <div class="question-header">
                                        <span class="question-number">${index + 1}</span>
                                        <span class="question-type">${item.type || 'Unknown'}</span>
                                    </div>
                                    <div class="question-content">
                                        <div class="question-text">
                                            <strong>Question:</strong> ${item.question}
                                        </div>
                                        <div class="answer-text">
                                            <strong>Answer:</strong> ${item.answer}
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

/**
 * Close modal
 */
function closeModal(button) {
    const modal = button.closest('.modal');
    if (modal) {
        modal.remove();
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="this.parentElement.parentElement.remove()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 5000);
    
    // Show toast
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);
}

/**
 * Refresh benchmarks
 */
function refreshBenchmarks() {
    loadBenchmarks();
}

/**
 * Refresh results
 */
function refreshResults() {
    loadResultsAnalysis();
}

/**
 * Cancel a running benchmark
 */
async function cancelBenchmark(runId) {
    if (!confirm('Are you sure you want to cancel this benchmark run?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate/benchmark/${runId}/cancel`, {
            method: 'POST',
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (response.ok) {
            const result = await response.json();
            showToast(`Benchmark ${runId} cancelled successfully`, 'success');
            // Refresh the benchmarks list to show updated status
            setTimeout(() => loadBenchmarks(), 1000);
        } else {
            const error = await response.json();
            showToast(`Failed to cancel benchmark: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error cancelling benchmark:', error);
        showToast('Error cancelling benchmark', 'error');
    }
}

// Export functions for global access
window.showCreateTestSetModal = showCreateTestSetModal;
window.hideCreateTestSetModal = hideCreateTestSetModal;
window.showRunBenchmarkModal = showRunBenchmarkModal;
window.hideRunBenchmarkModal = hideRunBenchmarkModal;
window.showBenchmarkResultsModal = showBenchmarkResultsModal;
window.hideBenchmarkResultsModal = hideBenchmarkResultsModal;
window.createTestSet = createTestSet;
window.runBenchmark = runBenchmark;
window.viewBenchmarkResults = viewBenchmarkResults;
window.downloadBenchmarkResults = downloadBenchmarkResults;
window.deleteTestSet = deleteTestSet;
window.deleteBenchmark = deleteBenchmark;
window.cancelBenchmark = cancelBenchmark;
window.generateAIInsights = generateAIInsights;
window.refreshTestSets = refreshTestSets;
window.refreshBenchmarks = refreshBenchmarks;
window.refreshResults = refreshResults;
