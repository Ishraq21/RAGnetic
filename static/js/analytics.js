// Analytics Dashboard JavaScript functionality
console.log('Analytics.js loaded successfully');

class AnalyticsDashboard {
    constructor() {
        this.currentTimeRange = '7d';
        this.currentTab = 'overview';
        this.analyticsData = {
            overview: null,
            costs: null,
            usage: null,
            training: null,
            deployments: null
        };
        this.charts = {};
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadAnalyticsData();
        this.initializeCharts();
        this.setupRealTimeUpdates();
    }

    setupEventListeners() {
        // Time range selector
        document.querySelectorAll('.time-range-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.currentTimeRange = e.target.dataset.range;
                this.updateTimeRangeSelection();
                this.loadAnalyticsData();
            });
        });

        // Sub-tab navigation
        document.querySelectorAll('.subtab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabId = e.target.dataset.tab;
                this.switchTab(tabId);
            });
        });

        // Refresh button
        const refreshBtn = document.getElementById('analytics-refresh');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadAnalyticsData();
            });
        }

        // Export buttons
        document.querySelectorAll('.export-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = e.target.dataset.export;
                this.exportData(type);
            });
        });
    }

    updateTimeRangeSelection() {
        document.querySelectorAll('.time-range-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-range="${this.currentTimeRange}"]`).classList.add('active');
    }

    switchTab(tabId) {
        // Update current tab
        this.currentTab = tabId;
        
        // Update tab button states
        document.querySelectorAll('.subtab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        const activeBtn = document.querySelector(`[data-tab="${tabId}"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
        }
        
        // Show/hide tab content
        document.querySelectorAll('.subtab-content').forEach(content => {
            content.classList.remove('active');
        });
        const activeTab = document.getElementById(`${tabId}-tab`);
        if (activeTab) {
            activeTab.classList.add('active');
        }
        
        // Destroy existing charts before creating new ones
        this.destroyAllCharts();
        
        // Initialize charts for the current tab
        this.initializeChartsForTab(tabId);
    }

    async loadAnalyticsData() {
        try {
            this.showLoadingState();
            
            // Load all analytics data in parallel, with error handling for each
            const [overview, costs, usage, training, deployments] = await Promise.allSettled([
                this.fetchOverviewData(),
                this.fetchCostData(),
                this.fetchUsageData(),
                this.fetchTrainingData(),
                this.fetchDeploymentData()
            ]);

            this.analyticsData = {
                overview: overview.status === 'fulfilled' ? overview.value : null,
                costs: costs.status === 'fulfilled' ? costs.value : null,
                usage: usage.status === 'fulfilled' ? usage.value : null,
                training: training.status === 'fulfilled' ? training.value : null,
                deployments: deployments.status === 'fulfilled' ? deployments.value : null
            };

            this.updateOverviewCards();
            this.updateCharts();
            this.updateTables();
            this.hideLoadingState();

        } catch (error) {
            console.error('Error loading analytics data:', error);
            this.showErrorState(error.message);
        }
    }

    async fetchOverviewData() {
        try {
            // Fetch comprehensive analytics overview data
            const [usageSummary, agentRuns, benchmarks, latency] = await Promise.all([
                fetch('/api/v1/analytics/usage-summary?limit=10', {
                    headers: { 'X-API-Key': loggedInUserToken }
                }),
                fetch('/api/v1/analytics/agent-runs?limit=10', {
                    headers: { 'X-API-Key': loggedInUserToken }
                }),
                fetch('/api/v1/analytics/benchmarks?limit=5', {
                    headers: { 'X-API-Key': loggedInUserToken }
                }),
                fetch('/api/v1/analytics/latency', {
                    headers: { 'X-API-Key': loggedInUserToken }
                })
            ]);

            const overviewData = {
                usageSummary: usageSummary.ok ? await usageSummary.json() : [],
                agentRuns: agentRuns.ok ? await agentRuns.json() : [],
                benchmarks: benchmarks.ok ? await benchmarks.json() : [],
                latency: latency.ok ? await latency.json() : null
            };

            return overviewData;
        } catch (error) {
            console.warn('Failed to fetch analytics overview data:', error);
            // Fallback to billing data
        const response = await fetch('/api/v1/billing/credits', {
                headers: { 'X-API-Key': loggedInUserToken }
        });
        if (!response.ok) throw new Error('Failed to fetch overview data');
        return await response.json();
        }
    }

    async fetchCostData() {
        try {
            // Fetch comprehensive cost analytics data
            const [usageSummary, agentRuns] = await Promise.all([
                fetch('/api/v1/analytics/usage-summary?limit=50', {
                    headers: { 'X-API-Key': loggedInUserToken }
                }),
                fetch('/api/v1/analytics/agent-runs?limit=50', {
                    headers: { 'X-API-Key': loggedInUserToken }
                })
            ]);

            const costData = {
                usageSummary: usageSummary.ok ? await usageSummary.json() : [],
                agentRuns: agentRuns.ok ? await agentRuns.json() : []
            };

            return costData;
        } catch (error) {
            console.warn('Failed to fetch analytics cost data:', error);
            // Fallback to billing data
        const response = await fetch(`/api/v1/billing/usage?range=${this.currentTimeRange}`, {
                headers: { 'X-API-Key': loggedInUserToken }
        });
        if (!response.ok) throw new Error('Failed to fetch cost data');
        return await response.json();
        }
    }

    async fetchUsageData() {
        try {
            // Fetch comprehensive usage analytics data
            const [usageSummary, agentSteps, lambdaRuns] = await Promise.all([
                fetch('/api/v1/analytics/usage-summary?limit=100', {
                    headers: { 'X-API-Key': loggedInUserToken }
                }),
                fetch('/api/v1/analytics/agent-steps?limit=100', {
                    headers: { 'X-API-Key': loggedInUserToken }
                }),
                fetch('/api/v1/analytics/lambda-runs?limit=50', {
                    headers: { 'X-API-Key': loggedInUserToken }
                })
            ]);

            const usageData = {
                usageSummary: usageSummary.ok ? await usageSummary.json() : [],
                agentSteps: agentSteps.ok ? await agentSteps.json() : [],
                lambdaRuns: lambdaRuns.ok ? await lambdaRuns.json() : []
            };

            return usageData;
        } catch (error) {
            console.warn('Failed to fetch analytics usage data:', error);
            // Fallback to billing data
        const response = await fetch(`/api/v1/billing/transactions?range=${this.currentTimeRange}`, {
                headers: { 'X-API-Key': loggedInUserToken }
        });
        if (!response.ok) throw new Error('Failed to fetch usage data');
        return await response.json();
        }
    }


    async fetchTrainingData() {
        try {
            // Fetch comprehensive training analytics data
            const [benchmarks, agentRuns] = await Promise.all([
                fetch('/api/v1/analytics/benchmarks?limit=20', {
                    headers: { 'X-API-Key': loggedInUserToken }
                }),
                fetch('/api/v1/analytics/agent-runs?limit=20', {
                    headers: { 'X-API-Key': loggedInUserToken }
                })
            ]);

            const trainingData = {
                benchmarks: benchmarks.ok ? await benchmarks.json() : [],
                agentRuns: agentRuns.ok ? await agentRuns.json() : []
            };

            return trainingData;
        } catch (error) {
            console.warn('Failed to fetch analytics training data:', error);
            // Fallback to training stats
        const response = await fetch('/api/v1/training/stats', {
                headers: { 'X-API-Key': loggedInUserToken }
        });
        if (!response.ok) throw new Error('Failed to fetch training data');
        return await response.json();
        }
    }

    async fetchDeploymentData() {
        try {
            // Fetch comprehensive deployment analytics data
            const [agentRuns, lambdaRuns] = await Promise.all([
                fetch('/api/v1/analytics/agent-runs?limit=50', {
                    headers: { 'X-API-Key': loggedInUserToken }
                }),
                fetch('/api/v1/analytics/lambda-runs?limit=50', {
                    headers: { 'X-API-Key': loggedInUserToken }
                })
            ]);

            const deploymentData = {
                agentRuns: agentRuns.ok ? await agentRuns.json() : [],
                lambdaRuns: lambdaRuns.ok ? await lambdaRuns.json() : []
            };

            return deploymentData;
        } catch (error) {
            console.warn('Failed to fetch analytics deployment data:', error);
            // Fallback to deployments endpoint
        const response = await fetch('/api/v1/deployments/', {
                headers: { 'X-API-Key': loggedInUserToken }
        });
        if (!response.ok) throw new Error('Failed to fetch deployment data');
        return await response.json();
        }
    }

    updateOverviewCards() {
        const data = this.analyticsData.overview;
        if (!data) return;

        // Calculate totals from usage summary
        let totalLLMCost = 0;
        let totalRequests = 0;
        let totalTokens = 0;
        let activeAgents = 0;
        let avgResponseTime = 0;
        let successRate = 0;

        if (data.usageSummary && Array.isArray(data.usageSummary)) {
            data.usageSummary.forEach(usage => {
                totalLLMCost += usage.total_estimated_cost_usd || 0;
                totalRequests += usage.total_requests || 0;
                totalTokens += usage.total_tokens || 0;
                if (usage.total_requests > 0) activeAgents++;
            });
        }


        // Calculate performance metrics
        if (data.latency && Array.isArray(data.latency)) {
            const totalLatency = data.latency.reduce((sum, item) => sum + (item.avg_response_time || 0), 0);
            avgResponseTime = data.latency.length > 0 ? totalLatency / data.latency.length : 0;
        }

        if (data.agentRuns && Array.isArray(data.agentRuns)) {
            const successfulRuns = data.agentRuns.filter(run => run.status === 'completed').length;
            successRate = data.agentRuns.length > 0 ? (successfulRuns / data.agentRuns.length) * 100 : 0;
        }

        const totalSpend = totalLLMCost;

        // Update Total Spend (primary card)
        const totalSpendEl = document.getElementById('total-spend');
        if (totalSpendEl) {
            totalSpendEl.textContent = `$${totalSpend.toFixed(2)}`;
        }

        // Update Total Requests
        const totalRequestsEl = document.getElementById('total-requests');
        if (totalRequestsEl) {
            totalRequestsEl.textContent = totalRequests.toLocaleString();
        }


        // Update Active Agents
        const activeAgentsEl = document.getElementById('active-agents');
        if (activeAgentsEl) {
            activeAgentsEl.textContent = activeAgents.toString();
        }

        // Update Average Response Time
        const avgResponseTimeEl = document.getElementById('avg-response-time');
        if (avgResponseTimeEl) {
            avgResponseTimeEl.textContent = `${avgResponseTime.toFixed(0)}ms`;
        }

        // Update Success Rate
        const successRateEl = document.getElementById('success-rate');
        if (successRateEl) {
            successRateEl.textContent = `${successRate.toFixed(1)}%`;
        }

        // Update trend indicators
        this.updateTrendIndicators();
    }

    updateTrendIndicators() {
        // Mock trend data - in real implementation, this would come from API
        const trends = {
            'spend-trend': { indicator: 'positive', text: '+12% vs last month' },
            'requests-trend': { indicator: 'positive', text: '+8% vs last month' },
            'gpu-trend': { indicator: 'neutral', text: 'Stable usage' },
            'agents-trend': { indicator: 'positive', text: '+3 new this week' },
            'response-trend': { indicator: 'negative', text: '-15% faster' },
            'success-trend': { indicator: 'positive', text: '+2% improvement' }
        };
        
        Object.entries(trends).forEach(([id, trend]) => {
            const trendElement = document.getElementById(id);
            if (trendElement) {
                const indicator = trendElement.querySelector('.trend-indicator');
                const text = trendElement.querySelector('.trend-text');
                
                if (indicator) {
                    indicator.className = `trend-indicator ${trend.indicator}`;
                    indicator.textContent = trend.indicator === 'positive' ? '↗' : 
                                          trend.indicator === 'negative' ? '↘' : '→';
                }
                if (text) {
                    text.textContent = trend.text;
                }
            }
        });
    }

    initializeCharts() {
        this.destroyAllCharts();
        this.initializeChartsForTab(this.currentTab);
    }

    destroyAllCharts() {
        // Destroy all existing charts to prevent canvas reuse errors
        Object.keys(this.charts).forEach(chartKey => {
            if (this.charts[chartKey] && typeof this.charts[chartKey].destroy === 'function') {
                this.charts[chartKey].destroy();
            }
        });
        this.charts = {};
    }

    createChart(chartId, chartKey, config) {
        const ctx = document.getElementById(chartId);
        if (!ctx) return;

        try {
            this.charts[chartKey] = new Chart(ctx, config);
        } catch (error) {
            console.error(`Error creating chart ${chartId}:`, error);
        }
    }

    initializeChartsForTab(tabId) {
        // Initialize Chart.js if available
        if (typeof Chart !== 'undefined') {
            switch(tabId) {
                case 'overview':
                    this.initializeOverviewCharts();
                    break;
                case 'costs':
                    this.initializeCostCharts();
                    break;
                case 'llm':
                    this.initializeLLMCharts();
                    break;
                case 'agent-runs':
                    this.initializeAgentRunCharts();
                    break;
                case 'training':
                    this.initializeTrainingCharts();
                    break;
                case 'gpu':
                    this.initializeGPUCharts();
                    break;
                case 'evaluation':
                    this.initializeEvaluationCharts();
                    break;
                default:
                    this.initializeOverviewCharts();
            }
        } else {
            console.warn('Chart.js not loaded, using fallback visualizations');
            this.initializeFallbackCharts();
        }
    }

    initializeCostChart() {
        const ctx = document.getElementById('cost-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const accentColor = css.getPropertyValue('--accent-color').trim() || '#8b5cf6';
        const accentAlpha = accentColor + '1a'; // Add alpha for background

        this.charts.cost = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.getTimeLabels(),
                datasets: [{
                    label: 'Daily Cost',
                    data: this.generateCostData(),
                    borderColor: accentColor,
                    backgroundColor: accentAlpha,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#a0a0a0',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    }
                }
            }
        });
    }

    initializeUsageChart() {
        const ctx = document.getElementById('usage-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const accentColor = css.getPropertyValue('--accent-color').trim() || '#8b5cf6';
        const successColor = css.getPropertyValue('--success-color').trim() || '#10b981';
        const warningColor = css.getPropertyValue('--warning-color').trim() || '#f59e0b';
        const infoColor = css.getPropertyValue('--info-color').trim() || '#3b82f6';

        this.charts.usage = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['API Calls', 'GPU Usage', 'Training', 'Other'],
                datasets: [{
                    data: this.generateUsageData(),
                    backgroundColor: [
                        accentColor,
                        successColor,
                        warningColor,
                        infoColor
                    ],
                    borderColor: '#000000',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            padding: 20
                        }
                    }
                }
            }
        });
    }

    initializeGPUChart() {
        const ctx = document.getElementById('gpu-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const accentColor = css.getPropertyValue('--accent-color').trim() || '#8b5cf6';
        const accentHover = css.getPropertyValue('--accent-hover').trim() || '#7c3aed';

        this.charts.gpu = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['A100', 'H100', 'RTX4090', 'RTX4080', 'V100'],
                datasets: [{
                    label: 'GPU Hours',
                    data: this.generateGPUData(),
                    backgroundColor: accentColor,
                    borderColor: accentHover,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    }
                }
            }
        });
    }

    initializeTrainingChart() {
        const ctx = document.getElementById('training-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const successColor = css.getPropertyValue('--success-color').trim() || '#10b981';
        const successAlpha = successColor + '1a'; // Add alpha for background

        this.charts.training = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.getTimeLabels(),
                datasets: [{
                    label: 'Training Jobs',
                    data: this.generateTrainingData(),
                    borderColor: successColor,
                    backgroundColor: successAlpha,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    }
                }
            }
        });
    }

    initializeCostDistributionChart() {
        const ctx = document.getElementById('cost-distribution-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const colors = [
            css.getPropertyValue('--accent-color').trim() || '#8b5cf6',
            css.getPropertyValue('--success-color').trim() || '#10b981',
            css.getPropertyValue('--warning-color').trim() || '#f59e0b',
            css.getPropertyValue('--error-color').trim() || '#ef4444'
        ];

        this.charts.costDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['LLM APIs', 'GPU Compute', 'Storage', 'Bandwidth'],
                datasets: [{
                    data: [65, 25, 7, 3],
                    backgroundColor: colors,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }

    initializeResponseTimeChart() {
        const ctx = document.getElementById('response-time-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const accentColor = css.getPropertyValue('--accent-color').trim() || '#8b5cf6';

        this.charts.responseTime = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.getTimeLabels(),
                datasets: [{
                    label: 'Avg Response Time (ms)',
                    data: this.generateResponseTimeData(),
                    borderColor: accentColor,
                    backgroundColor: accentColor + '1a',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    }
                }
            }
        });
    }

    initializeSuccessRateChart() {
        const ctx = document.getElementById('success-rate-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const successColor = css.getPropertyValue('--success-color').trim() || '#10b981';

        this.charts.successRate = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5'],
                datasets: [{
                    label: 'Success Rate (%)',
                    data: [98.5, 97.2, 99.1, 96.8, 98.9],
                    backgroundColor: successColor + '80',
                    borderColor: successColor,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        },
                        min: 90,
                        max: 100
                    }
                }
            }
        });
    }

    initializeGPUEfficiencyChart() {
        const ctx = document.getElementById('gpu-efficiency-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const warningColor = css.getPropertyValue('--warning-color').trim() || '#f59e0b';

        this.charts.gpuEfficiency = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['A100', 'H100', 'RTX4090', 'RTX4080', 'V100'],
                datasets: [{
                    label: 'Cost per Hour ($)',
                    data: [2.50, 3.20, 0.79, 0.65, 1.80],
                    backgroundColor: warningColor + '80',
                    borderColor: warningColor,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#a0a0a0'
                        },
                        grid: {
                            color: '#1f1f1f'
                        }
                    }
                }
            }
        });
    }

    initializeAgentPerformanceChart() {
        const ctx = document.getElementById('agent-performance-chart');
        if (!ctx) return;

        // Get theme colors from CSS variables
        const css = getComputedStyle(document.documentElement);
        const accentColor = css.getPropertyValue('--accent-color').trim() || '#8b5cf6';

        this.charts.agentPerformance = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Speed', 'Reliability', 'Cost Efficiency', 'User Satisfaction'],
                datasets: [{
                    label: 'Agent Performance',
                    data: [85, 78, 92, 88, 90],
                    borderColor: accentColor,
                    backgroundColor: accentColor + '20',
                    pointBackgroundColor: accentColor,
                    pointBorderColor: '#ffffff',
                    pointHoverBackgroundColor: '#ffffff',
                    pointHoverBorderColor: accentColor
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    r: {
                        ticks: {
                            color: '#a0a0a0',
                            backdropColor: 'transparent'
                        },
                        grid: {
                            color: '#1f1f1f'
                        },
                        pointLabels: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }

    initializeFallbackCharts() {
        // Create simple bar charts using CSS and data attributes
        this.createFallbackChart('cost-chart', this.generateCostData());
        this.createFallbackChart('usage-chart', this.generateUsageData());
        this.createFallbackChart('gpu-chart', this.generateGPUData());
        this.createFallbackChart('training-chart', this.generateTrainingData());
    }

    createFallbackChart(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const maxValue = Math.max(...data);
        const chartHtml = data.map((value, index) => {
            const height = (value / maxValue) * 100;
            return `
                <div class="fallback-bar" style="height: ${height}%">
                    <span class="bar-value">${value}</span>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="fallback-chart">
                ${chartHtml}
            </div>
        `;
    }

    updateCharts() {
        if (this.charts.cost) {
            this.charts.cost.data.datasets[0].data = this.generateCostData();
            this.charts.cost.update();
        }
        if (this.charts.usage) {
            this.charts.usage.data.datasets[0].data = this.generateUsageData();
            this.charts.usage.update();
        }
        if (this.charts.gpu) {
            this.charts.gpu.data.datasets[0].data = this.generateGPUData();
            this.charts.gpu.update();
        }
        if (this.charts.training) {
            this.charts.training.data.datasets[0].data = this.generateTrainingData();
            this.charts.training.update();
        }
    }

    updateTables() {
        this.updateCostTable();
        this.updateUsageTable();
        this.updateGPUTable();
        this.updateTrainingTable();
        this.updateDeploymentTable();
    }

    updateCostTable() {
        const tbody = document.getElementById('cost-table-body');
        if (!tbody || !this.analyticsData.costs) return;

        const costData = this.analyticsData.costs;
        const rows = [];

        // Use new analytics data structure if available
        if (costData.usageSummary && Array.isArray(costData.usageSummary)) {
            costData.usageSummary.forEach(usage => {
                const totalCost = usage.total_estimated_cost_usd || 0;
                if (totalCost > 0) {
                    rows.push(`
                        <tr>
                            <td>${usage.agent_name || 'N/A'}</td>
                            <td>$${totalCost.toFixed(2)}</td>
                            <td><span class="status-badge llm">LLM</span></td>
                            <td>${usage.llm_model} (${usage.total_requests} requests)</td>
                        </tr>
                    `);
                }
            });
        } else {
            // Fallback to old data structure
        if (costData.gpu_cost > 0) {
            rows.push(`
                <tr>
                    <td>${new Date().toLocaleDateString()}</td>
                    <td>$${costData.gpu_cost.toFixed(2)}</td>
                    <td><span class="status-badge gpu">GPU</span></td>
                    <td>GPU Usage (${costData.gpu_hours.toFixed(1)} hours)</td>
                </tr>
            `);
        }

        if (costData.api_cost > 0) {
            rows.push(`
                <tr>
                    <td>${new Date().toLocaleDateString()}</td>
                    <td>$${costData.api_cost.toFixed(2)}</td>
                    <td><span class="status-badge api">API</span></td>
                    <td>API Calls (${costData.api_calls} calls)</td>
                </tr>
            `);
            }
        }

        tbody.innerHTML = rows.join('') || '<tr><td colspan="4" class="text-center text-muted">No cost data available</td></tr>';
    }

    updateUsageTable() {
        const tbody = document.getElementById('usage-table-body');
        if (!tbody || !this.analyticsData.usage) return;

        const usageData = this.analyticsData.usage;
        const rows = [];

        // Use new analytics data structure if available
        if (usageData.usageSummary && Array.isArray(usageData.usageSummary)) {
            usageData.usageSummary.forEach(usage => {
                rows.push(`
                    <tr>
                        <td>${usage.agent_name || 'N/A'}</td>
                        <td>${usage.total_requests}</td>
                        <td>${usage.total_tokens}</td>
                        <td>$${usage.total_estimated_cost_usd.toFixed(2)}</td>
                    </tr>
                `);
            });
        } else {
            // Fallback to old data structure
            const oldRows = usageData.map(usage => `
            <tr>
                <td>${new Date(usage.created_at).toLocaleDateString()}</td>
                <td>${usage.request_count}</td>
                <td>$${usage.total_cost.toFixed(2)}</td>
                <td>${usage.deployment_id || 'N/A'}</td>
            </tr>
        `).join('');
            rows.push(oldRows);
        }

        tbody.innerHTML = rows.join('') || '<tr><td colspan="4" class="text-center text-muted">No usage data available</td></tr>';
    }

    updateGPUTable() {
        const tbody = document.getElementById('gpu-table-body');
        if (!tbody || !this.analyticsData.gpu) return;

        // Use the new analytics data structure
        const gpuData = this.analyticsData.gpu;
        
        // If we have the new analytics structure, use it
        if (gpuData.usageSummary && Array.isArray(gpuData.usageSummary)) {
            const rows = gpuData.usageSummary.map(usage => `
                <tr>
                    <td>${usage.gpu_type}</td>
                    <td>${usage.provider}</td>
                    <td>${usage.total_hours.toFixed(1)}h</td>
                    <td>$${usage.total_cost_usd.toFixed(2)}</td>
                    <td>${usage.total_instances}</td>
                    <td>${usage.success_rate.toFixed(1)}%</td>
                </tr>
            `).join('');
            
            tbody.innerHTML = rows || '<tr><td colspan="6" class="text-center text-muted">No GPU usage data available</td></tr>';
        } else {
            // Fallback to old data structure
            const rows = gpuData.map(gpu => `
            <tr>
                <td>${gpu.instance_id}</td>
                <td>${gpu.usage_type}</td>
                <td>${gpu.duration_minutes} min</td>
                <td>$${gpu.cost.toFixed(2)}</td>
                <td>${new Date(gpu.created_at).toLocaleDateString()}</td>
            </tr>
        `).join('');

            tbody.innerHTML = rows || '<tr><td colspan="5" class="text-center text-muted">No GPU usage data available</td></tr>';
        }
    }

    updateTrainingTable() {
        const tbody = document.getElementById('training-table-body');
        if (!tbody || !this.analyticsData.training) return;

        const trainingData = this.analyticsData.training;
        const rows = [];

        // Use new analytics data structure if available
        if (trainingData.benchmarks && Array.isArray(trainingData.benchmarks)) {
            trainingData.benchmarks.forEach(benchmark => {
                rows.push(`
                    <tr>
                        <td>${benchmark.agent_name || 'N/A'} Benchmark</td>
                        <td><span class="status-badge completed">Completed</span></td>
                        <td>$${(benchmark['Total Estimated Cost (USD)'] || 0).toFixed(2)}</td>
                        <td>${benchmark['Total Test Cases Evaluated'] || 0} test cases</td>
                    </tr>
                `);
            });
        } else if (trainingData.agentRuns && Array.isArray(trainingData.agentRuns)) {
            trainingData.agentRuns.forEach(run => {
                rows.push(`
                    <tr>
                        <td>${run.agent_name || 'N/A'}</td>
                        <td><span class="status-badge ${(run.status || 'unknown').toLowerCase()}">${run.status || 'Unknown'}</span></td>
                        <td>${run.total_steps || 0}</td>
                        <td>${run.start_time ? new Date(run.start_time).toLocaleDateString() : 'N/A'}</td>
                    </tr>
                `);
            });
        } else {
            // Fallback to old data structure
        if (trainingData.total_jobs > 0) {
            rows.push(`
                <tr>
                    <td>Training Summary</td>
                    <td><span class="status-badge completed">${trainingData.completed_jobs} completed</span></td>
                    <td>$${trainingData.total_cost_usd.toFixed(2)}</td>
                    <td>${trainingData.total_gpu_hours.toFixed(1)} GPU hours</td>
                </tr>
            `);
            }
        }

        tbody.innerHTML = rows.join('') || '<tr><td colspan="4" class="text-center text-muted">No training data available</td></tr>';
    }

    updateDeploymentTable() {
        const tbody = document.getElementById('deployment-table-body');
        if (!tbody || !this.analyticsData.deployments) return;

        const deploymentData = this.analyticsData.deployments;
        const rows = [];

        // Use new analytics data structure if available
        if (deploymentData.agentRuns && Array.isArray(deploymentData.agentRuns)) {
            deploymentData.agentRuns.forEach(run => {
                rows.push(`
                    <tr>
                        <td>${run.agent_name || 'N/A'}</td>
                        <td><span class="status-badge ${(run.status || 'unknown').toLowerCase()}">${run.status || 'Unknown'}</span></td>
                        <td>${run.total_steps || 0} steps</td>
                        <td>${run.start_time ? new Date(run.start_time).toLocaleDateString() : 'N/A'}</td>
                    </tr>
                `);
            });
        } else if (deploymentData.lambdaRuns && Array.isArray(deploymentData.lambdaRuns)) {
            deploymentData.lambdaRuns.forEach(run => {
                rows.push(`
                    <tr>
                        <td>Lambda Run ${run.run_id || 'N/A'}</td>
                        <td><span class="status-badge ${(run.status || 'unknown').toLowerCase()}">${run.status || 'Unknown'}</span></td>
                        <td>${run.total_outputs || 0} outputs</td>
                        <td>${run.start_time ? new Date(run.start_time).toLocaleDateString() : 'N/A'}</td>
                    </tr>
                `);
            });
        } else {
            // Fallback to old data structure
            const oldRows = deploymentData.map(deployment => `
            <tr>
                <td>${deployment.id}</td>
                <td><span class="status-badge ${deployment.status}">${deployment.status}</span></td>
                <td>${deployment.deployment_type}</td>
                <td>${new Date(deployment.created_at).toLocaleDateString()}</td>
            </tr>
        `).join('');
            rows.push(oldRows);
        }

        tbody.innerHTML = rows.join('') || '<tr><td colspan="4" class="text-center text-muted">No deployment data available</td></tr>';
    }

    // Data generation methods (replace with real data from APIs)
    getTimeLabels() {
        const days = this.currentTimeRange === '7d' ? 7 : this.currentTimeRange === '30d' ? 30 : 90;
        const labels = [];
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        }
        return labels;
    }

    generateCostData() {
        // Mock data - replace with real API data
        const days = this.currentTimeRange === '7d' ? 7 : this.currentTimeRange === '30d' ? 30 : 90;
        return Array.from({ length: days }, () => Math.random() * 50 + 10);
    }

    generateUsageData() {
        // Mock data - replace with real API data
        return [45, 25, 20, 10];
    }

    generateGPUData() {
        // Use real GPU analytics data
        const gpuData = this.analyticsData.gpu;
        
        if (gpuData && gpuData.usageSummary && Array.isArray(gpuData.usageSummary)) {
            // Group by GPU type and sum hours
            const gpuTypeHours = {};
            gpuData.usageSummary.forEach(usage => {
                if (!gpuTypeHours[usage.gpu_type]) {
                    gpuTypeHours[usage.gpu_type] = 0;
                }
                gpuTypeHours[usage.gpu_type] += usage.total_hours;
            });
            
            // Convert to array format for chart
            const labels = Object.keys(gpuTypeHours);
            const data = Object.values(gpuTypeHours);
            
            // Update chart labels if we have real data
            if (this.charts.gpu && labels.length > 0) {
                this.charts.gpu.data.labels = labels;
            }
            
            return data.length > 0 ? data : [0];
        }
        
        // Fallback to mock data
        return [120, 80, 60, 40, 20];
    }

    generateTrainingData() {
        // Mock data - replace with real API data
        const days = this.currentTimeRange === '7d' ? 7 : this.currentTimeRange === '30d' ? 30 : 90;
        return Array.from({ length: days }, () => Math.floor(Math.random() * 5));
    }

    generateResponseTimeData() {
        // Mock data - replace with real API data
        const days = this.currentTimeRange === '7d' ? 7 : this.currentTimeRange === '30d' ? 30 : 90;
        return Array.from({ length: days }, () => Math.random() * 200 + 100);
    }

    setupRealTimeUpdates() {
        // Set up periodic data refresh
        setInterval(() => {
            this.loadAnalyticsData();
        }, 30000); // Refresh every 30 seconds
    }

    showLoadingState() {
        const loadingEl = document.getElementById('analytics-loading');
        if (loadingEl) {
            loadingEl.style.display = 'block';
        }
    }

    hideLoadingState() {
        const loadingEl = document.getElementById('analytics-loading');
        if (loadingEl) {
            loadingEl.style.display = 'none';
        }
    }

    showErrorState(message) {
        const errorEl = document.getElementById('analytics-error');
        if (errorEl) {
            errorEl.textContent = message;
            errorEl.style.display = 'block';
        }
    }

    exportData(type) {
        // TODO: Implement data export functionality
        console.log(`Exporting ${type} data...`);
        this.showToast('Export functionality coming soon!', 'info');
    }

    showToast(message, type = 'info') {
        // Simple toast notification
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            color: var(--text-primary);
            z-index: 1000;
            box-shadow: var(--shadow-lg);
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    // Tab-specific chart initialization methods
    initializeOverviewCharts() {
        // Initialize basic charts for overview
        this.initializeCostChart();
        this.initializeUsageChart();
        this.initializeResponseTimeChart();
        this.initializeSuccessRateChart();
        
        // Update overview cards with real data
        this.updateOverviewCards();
    }

    initializeCostCharts() {
        this.initializeCostChart();
        this.initializeCostDistributionChart();
        this.initializeCostTrendChart();
        this.initializeCostBreakdownChart();
    }

    initializeLLMCharts() {
        this.initializeTokenUsageChart();
        this.initializeModelUsageChart();
        this.initializeLLMCostChart();
        this.initializeLLMPerformanceChart();
    }

    initializeAgentRunCharts() {
        this.initializeAgentPerformanceChart();
        this.initializeRunSuccessChart();
        this.initializeRunDurationChart();
        this.initializeAgentActivityChart();
    }

    initializeTrainingCharts() {
        this.initializeTrainingChart();
        this.initializeTrainingProgressChart();
        this.initializeTrainingCostChart();
        this.initializeTrainingPerformanceChart();
    }

    initializeGPUCharts() {
        this.initializeGPUChart();
        this.initializeGPUEfficiencyChart();
        this.initializeGPUUsageChart();
        this.initializeGPUCostChart();
    }

    initializeEvaluationCharts() {
        this.initializeEvaluationScoreChart();
        this.initializeBenchmarkChart();
        this.initializeEvaluationTrendChart();
        this.initializeEvaluationComparisonChart();
    }

    // New chart methods for detailed analytics
    initializeCostTrendChart() {
        const { labels, data } = this.generateCostTrendData();
        
        this.createChart('cost-trend-chart', 'costTrend', {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Total Cost ($)',
                    data: data,
                    borderColor: 'rgb(99, 102, 241)',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }

    initializeCostBreakdownChart() {
        const data = this.generateCostBreakdownData();
        
        this.createChart('cost-breakdown-chart', 'costBreakdown', {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: [
                        'rgb(99, 102, 241)',
                        'rgb(16, 185, 129)',
                        'rgb(245, 158, 11)',
                        'rgb(239, 68, 68)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    initializeTokenUsageChart() {
        const ctx = document.getElementById('token-usage-chart');
        if (!ctx) return;

        const { labels, data } = this.generateTokenUsageData();
        
        this.charts.tokenUsage = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Tokens Used',
                    data: data,
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: 'rgb(99, 102, 241)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initializeModelUsageChart() {
        const ctx = document.getElementById('model-usage-chart');
        if (!ctx) return;

        const data = this.generateModelUsageData();
        
        this.charts.modelUsage = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: [
                        'rgb(99, 102, 241)',
                        'rgb(16, 185, 129)',
                        'rgb(245, 158, 11)',
                        'rgb(239, 68, 68)',
                        'rgb(139, 92, 246)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    initializeLLMCostChart() {
        const ctx = document.getElementById('llm-cost-chart');
        if (!ctx) return;

        const { labels, data } = this.generateLLMCostData();
        
        this.charts.llmCost = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'LLM Cost ($)',
                    data: data,
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }

    initializeLLMPerformanceChart() {
        const ctx = document.getElementById('llm-performance-chart');
        if (!ctx) return;

        const { labels, data } = this.generateLLMPerformanceData();
        
        this.charts.llmPerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Response Time (ms)',
                    data: data,
                    backgroundColor: 'rgba(245, 158, 11, 0.8)',
                    borderColor: 'rgb(245, 158, 11)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initializeRunSuccessChart() {
        const ctx = document.getElementById('run-success-chart');
        if (!ctx) return;

        const data = this.generateRunSuccessData();
        
        this.charts.runSuccess = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Successful', 'Failed', 'In Progress'],
                datasets: [{
                    data: [data.successful, data.failed, data.inProgress],
                    backgroundColor: [
                        'rgb(16, 185, 129)',
                        'rgb(239, 68, 68)',
                        'rgb(245, 158, 11)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    initializeRunDurationChart() {
        const ctx = document.getElementById('run-duration-chart');
        if (!ctx) return;

        const { labels, data } = this.generateRunDurationData();
        
        this.charts.runDuration = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Average Duration (seconds)',
                    data: data,
                    borderColor: 'rgb(139, 92, 246)',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initializeAgentActivityChart() {
        const ctx = document.getElementById('agent-activity-chart');
        if (!ctx) return;

        const { labels, data } = this.generateAgentActivityData();
        
        this.charts.agentActivity = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Runs',
                    data: data,
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: 'rgb(99, 102, 241)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initializeTrainingProgressChart() {
        const ctx = document.getElementById('training-progress-chart');
        if (!ctx) return;

        const { labels, data } = this.generateTrainingProgressData();
        
        this.charts.trainingProgress = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Progress (%)',
                    data: data,
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    initializeTrainingCostChart() {
        const ctx = document.getElementById('training-cost-chart');
        if (!ctx) return;

        const { labels, data } = this.generateTrainingCostData();
        
        this.charts.trainingCost = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cost ($)',
                    data: data,
                    backgroundColor: 'rgba(245, 158, 11, 0.8)',
                    borderColor: 'rgb(245, 158, 11)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }

    initializeTrainingPerformanceChart() {
        const ctx = document.getElementById('training-performance-chart');
        if (!ctx) return;

        const { labels, data } = this.generateTrainingPerformanceData();
        
        this.charts.trainingPerformance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Accuracy (%)',
                    data: data,
                    borderColor: 'rgb(139, 92, 246)',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    initializeGPUUsageChart() {
        const ctx = document.getElementById('gpu-usage-chart');
        if (!ctx) return;

        const { labels, data } = this.generateGPUUsageData();
        
        this.charts.gpuUsage = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Usage Hours',
                    data: data,
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: 'rgb(99, 102, 241)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initializeGPUCostChart() {
        const ctx = document.getElementById('gpu-cost-chart');
        if (!ctx) return;

        const { labels, data } = this.generateGPUCostData();
        
        this.charts.gpuCost = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'GPU Cost ($)',
                    data: data,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }

    initializeEvaluationScoreChart() {
        const ctx = document.getElementById('evaluation-score-chart');
        if (!ctx) return;

        const { labels, data } = this.generateEvaluationScoreData();
        
        this.charts.evaluationScore = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Score',
                    data: data,
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderColor: 'rgb(16, 185, 129)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    initializeBenchmarkChart() {
        const ctx = document.getElementById('benchmark-chart');
        if (!ctx) return;

        const { labels, data } = this.generateBenchmarkData();
        
        this.charts.benchmark = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Benchmark Score',
                    data: data,
                    borderColor: 'rgb(99, 102, 241)',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    initializeEvaluationTrendChart() {
        const ctx = document.getElementById('evaluation-trend-chart');
        if (!ctx) return;

        const { labels, data } = this.generateEvaluationTrendData();
        
        this.charts.evaluationTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Evaluation Score',
                    data: data,
                    borderColor: 'rgb(139, 92, 246)',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    initializeEvaluationComparisonChart() {
        const ctx = document.getElementById('evaluation-comparison-chart');
        if (!ctx) return;

        const data = this.generateEvaluationComparisonData();
        
        this.charts.evaluationComparison = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Current',
                    data: data.current,
                    borderColor: 'rgb(99, 102, 241)',
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 2
                }, {
                    label: 'Previous',
                    data: data.previous,
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    // Data generation methods for new charts
    generateCostTrendData() {
        const days = 30;
        const data = [];
        const labels = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            const baseCost = 50 + Math.random() * 100;
            data.push(parseFloat(baseCost.toFixed(2)));
        }
        
        return { labels, data };
    }

    generateCostBreakdownData() {
        return {
            labels: ['LLM APIs', 'GPU Compute', 'Training', 'Storage'],
            values: [45, 30, 15, 10]
        };
    }

    generateTokenUsageData() {
        const models = ['GPT-4', 'GPT-3.5', 'Claude-3', 'Gemini Pro'];
        const data = models.map(() => Math.floor(Math.random() * 1000000) + 100000);
        return { labels: models, data };
    }

    generateModelUsageData() {
        return {
            labels: ['GPT-4', 'GPT-3.5', 'Claude-3', 'Gemini Pro', 'Others'],
            values: [35, 25, 20, 15, 5]
        };
    }

    generateLLMCostData() {
        const days = 30;
        const data = [];
        const labels = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            const baseCost = 20 + Math.random() * 40;
            data.push(parseFloat(baseCost.toFixed(2)));
        }
        
        return { labels, data };
    }

    generateLLMPerformanceData() {
        const models = ['GPT-4', 'GPT-3.5', 'Claude-3', 'Gemini Pro'];
        const data = models.map(() => Math.floor(Math.random() * 2000) + 500);
        return { labels: models, data };
    }

    generateRunSuccessData() {
        return {
            successful: 85,
            failed: 10,
            inProgress: 5
        };
    }

    generateRunDurationData() {
        const days = 30;
        const data = [];
        const labels = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            const duration = 30 + Math.random() * 120;
            data.push(Math.round(duration));
        }
        
        return { labels, data };
    }

    generateAgentActivityData() {
        const agents = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5'];
        const data = agents.map(() => Math.floor(Math.random() * 50) + 10);
        return { labels: agents, data };
    }

    generateTrainingProgressData() {
        const days = 30;
        const data = [];
        const labels = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            const progress = Math.min(100, 20 + (days - i) * 2.5 + Math.random() * 10);
            data.push(Math.round(progress));
        }
        
        return { labels, data };
    }

    generateTrainingCostData() {
        const jobs = ['Job 1', 'Job 2', 'Job 3', 'Job 4', 'Job 5'];
        const data = jobs.map(() => Math.floor(Math.random() * 200) + 50);
        return { labels: jobs, data };
    }

    generateTrainingPerformanceData() {
        const days = 30;
        const data = [];
        const labels = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            const accuracy = 70 + Math.random() * 25;
            data.push(Math.round(accuracy));
        }
        
        return { labels, data };
    }

    generateGPUUsageData() {
        const gpus = ['RTX 4090', 'A100', 'V100', 'RTX 3090'];
        const data = gpus.map(() => Math.floor(Math.random() * 100) + 20);
        return { labels: gpus, data };
    }

    generateGPUCostData() {
        const days = 30;
        const data = [];
        const labels = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            const baseCost = 15 + Math.random() * 30;
            data.push(parseFloat(baseCost.toFixed(2)));
        }
        
        return { labels, data };
    }

    generateEvaluationScoreData() {
        const metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'BLEU'];
        const data = metrics.map(() => Math.floor(Math.random() * 30) + 70);
        return { labels: metrics, data };
    }

    generateBenchmarkData() {
        const days = 30;
        const data = [];
        const labels = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            const score = 75 + Math.random() * 20;
            data.push(Math.round(score));
        }
        
        return { labels, data };
    }

    generateEvaluationTrendData() {
        const days = 30;
        const data = [];
        const labels = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            const score = 80 + Math.random() * 15;
            data.push(Math.round(score));
        }
        
        return { labels, data };
    }

    generateEvaluationComparisonData() {
        return {
            labels: ['Accuracy', 'Speed', 'Cost', 'Reliability', 'Scalability'],
            current: [85, 78, 82, 90, 75],
            previous: [80, 75, 85, 88, 70]
        };
    }
}

// Initialize analytics dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if we're on the analytics view
    if (document.getElementById('analytics-view')) {
        window.analyticsDashboard = new AnalyticsDashboard();
    }
});

// Export for use in other modules
window.AnalyticsDashboard = AnalyticsDashboard;
