// Billing Management JavaScript
console.log('Billing.js loaded successfully');

class BillingManager {
    constructor() {
        this.billingSummary = null;
        this.transactions = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Top-up form
        const topUpForm = document.getElementById('top-up-form');
        if (topUpForm) {
            topUpForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.topUp();
            });
        }

        // Spending limit form
        const spendingLimitForm = document.getElementById('spending-limit-form');
        if (spendingLimitForm) {
            spendingLimitForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.updateSpendingLimit();
            });
        }
    }

    async loadSummary() {
        try {
            // Ensure token is available
            const token = loggedInUserToken || localStorage.getItem('ragnetic_user_token');
            
            if (!token) {
                console.error('No authentication token available for billing summary request');
                return;
            }
            
            const response = await fetch(`${API_BASE_URL}/billing/summary`, {
                headers: {
                    'X-API-Key': token,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.billingSummary = data;
            this.renderSummary();
        } catch (error) {
            console.error('Error loading billing summary:', error);
            this.showError('Failed to load billing summary');
        }
    }

    async loadTransactions() {
        try {
            // Ensure token is available
            const token = loggedInUserToken || localStorage.getItem('ragnetic_user_token');
            
            if (!token) {
                console.error('No authentication token available for billing transactions request');
                return;
            }
            
            const response = await fetch(`${API_BASE_URL}/billing/transactions`, {
                headers: {
                    'X-API-Key': token,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.transactions = data;
            this.renderTransactions();
        } catch (error) {
            console.error('Error loading transactions:', error);
            this.showError('Failed to load transactions');
        }
    }

    renderSummary() {
        const container = document.querySelector('.billing-content');
        if (!container) return;

        if (!this.billingSummary) {
            container.innerHTML = `
                <div class="loading-state">
                    <div class="loading-spinner"></div>
                    <p>Loading billing information...</p>
                </div>
            `;
            return;
        }

        const { balance, daily_limit, total_spent, updated_at } = this.billingSummary;
        const lastUpdated = new Date(updated_at).toLocaleDateString();

        container.innerHTML = `
            <div class="billing-sections">
                <div class="billing-section">
                    <h3>Account Balance</h3>
                    <div class="balance-card">
                        <div class="balance-amount">
                            <span class="currency">$</span>
                            <span class="amount">${balance.toFixed(2)}</span>
                        </div>
                        <div class="balance-actions">
                            <button class="btn-primary" onclick="billingManager.showTopUpModal()">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M12 5v14M5 12h14"></path>
                                </svg>
                                Add Credits
                            </button>
                        </div>
                    </div>
                </div>

                <div class="billing-section">
                    <h3>Spending Limits</h3>
                    <div class="limits-card">
                        <div class="limit-item">
                            <span class="limit-label">Daily Limit</span>
                            <span class="limit-value">$${daily_limit?.toFixed(2) || 'No limit'}</span>
                        </div>
                        <div class="limit-item">
                            <span class="limit-label">Total Spent</span>
                            <span class="limit-value">$${total_spent?.toFixed(2) || '0.00'}</span>
                        </div>
                        <div class="limit-actions">
                            <button class="btn-secondary" onclick="billingManager.showSpendingLimitModal()">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                                </svg>
                                Update Limit
                            </button>
                        </div>
                    </div>
                </div>

                <div class="billing-section">
                    <h3>Transaction History</h3>
                    <div class="transactions-container" id="transactions-container">
                        <div class="loading-state">
                            <div class="loading-spinner"></div>
                            <p>Loading transactions...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderTransactions() {
        const container = document.getElementById('transactions-container');
        if (!container) return;

        if (this.transactions.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <line x1="12" y1="1" x2="12" y2="23"></line>
                            <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                        </svg>
                    </div>
                    <p>No transactions yet</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="transactions-list">
                ${this.transactions.map(transaction => this.renderTransactionItem(transaction)).join('')}
            </div>
        `;
    }

    renderTransactionItem(transaction) {
        const date = new Date(transaction.created_at).toLocaleDateString();
        const time = new Date(transaction.created_at).toLocaleTimeString();
        const amount = parseFloat(transaction.amount);
        const isCredit = transaction.transaction_type === 'credit';
        const amountClass = isCredit ? 'credit' : 'debit';
        const amountPrefix = isCredit ? '+' : '-';

        return `
            <div class="transaction-item">
                <div class="transaction-main">
                    <div class="transaction-info">
                        <div class="transaction-description">
                            ${this.escapeHtml(transaction.description || 'Transaction')}
                        </div>
                        <div class="transaction-meta">
                            <span class="transaction-date">${date} at ${time}</span>
                            ${transaction.gpu_instance_id ? `
                                <span class="transaction-gpu">GPU Instance #${transaction.gpu_instance_id}</span>
                            ` : ''}
                        </div>
                    </div>
                    <div class="transaction-amount ${amountClass}">
                        ${amountPrefix}$${Math.abs(amount).toFixed(2)}
                    </div>
                </div>
            </div>
        `;
    }

    showTopUpModal() {
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Add Credits</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                </div>
                <form id="top-up-form">
                    <div class="form-group">
                        <label for="top-up-amount">Amount (USD)</label>
                        <input type="number" id="top-up-amount" name="amount" min="1" max="1000" step="0.01" required>
                        <small>Minimum $1.00, maximum $1,000.00</small>
                    </div>
                    <div class="top-up-options">
                        <h4>Quick Amounts</h4>
                        <div class="quick-amounts">
                            <button type="button" class="btn-secondary btn-sm" onclick="billingManager.setTopUpAmount(10)">$10</button>
                            <button type="button" class="btn-secondary btn-sm" onclick="billingManager.setTopUpAmount(25)">$25</button>
                            <button type="button" class="btn-secondary btn-sm" onclick="billingManager.setTopUpAmount(50)">$50</button>
                            <button type="button" class="btn-secondary btn-sm" onclick="billingManager.setTopUpAmount(100)">$100</button>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                        <button type="submit" class="btn-primary">Add Credits</button>
                    </div>
                </form>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Setup form event listener
        modal.querySelector('#top-up-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.topUp();
        });
    }

    showSpendingLimitModal() {
        const currentLimit = this.billingSummary?.daily_limit || 0;
        
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Update Spending Limit</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                </div>
                <form id="spending-limit-form">
                    <div class="form-group">
                        <label for="daily-limit">Daily Spending Limit (USD)</label>
                        <input type="number" id="daily-limit" name="daily_limit" min="0" max="10000" step="0.01" value="${currentLimit}" required>
                        <small>Set to 0 for no limit. Maximum $10,000.00</small>
                    </div>
                    <div class="limit-options">
                        <h4>Quick Limits</h4>
                        <div class="quick-limits">
                            <button type="button" class="btn-secondary btn-sm" onclick="billingManager.setSpendingLimit(0)">No Limit</button>
                            <button type="button" class="btn-secondary btn-sm" onclick="billingManager.setSpendingLimit(50)">$50</button>
                            <button type="button" class="btn-secondary btn-sm" onclick="billingManager.setSpendingLimit(100)">$100</button>
                            <button type="button" class="btn-secondary btn-sm" onclick="billingManager.setSpendingLimit(500)">$500</button>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                        <button type="submit" class="btn-primary">Update Limit</button>
                    </div>
                </form>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Setup form event listener
        modal.querySelector('#spending-limit-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.updateSpendingLimit();
        });
    }

    setTopUpAmount(amount) {
        const amountInput = document.getElementById('top-up-amount');
        if (amountInput) {
            amountInput.value = amount;
        }
    }

    setSpendingLimit(limit) {
        const limitInput = document.getElementById('daily-limit');
        if (limitInput) {
            limitInput.value = limit;
        }
    }

    async topUp() {
        const form = document.getElementById('top-up-form');
        if (!form) return;

        const formData = new FormData(form);
        const amount = parseFloat(formData.get('amount'));

        if (amount < 1 || amount > 1000) {
            this.showError('Amount must be between $1.00 and $1,000.00');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/billing/top-up`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ amount })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            this.showSuccess(`Successfully added $${amount.toFixed(2)} to your account`);
            document.querySelector('.modal').remove();
            this.loadSummary();
            this.loadTransactions();
        } catch (error) {
            console.error('Error topping up credits:', error);
            this.showError(`Failed to add credits: ${error.message}`);
        }
    }

    async updateSpendingLimit() {
        const form = document.getElementById('spending-limit-form');
        if (!form) return;

        const formData = new FormData(form);
        const dailyLimit = parseFloat(formData.get('daily_limit'));

        if (dailyLimit < 0 || dailyLimit > 10000) {
            this.showError('Daily limit must be between $0.00 and $10,000.00');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/billing/spending-limit`, {
                method: 'PUT',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ daily_limit: dailyLimit })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const limitText = dailyLimit === 0 ? 'removed' : `set to $${dailyLimit.toFixed(2)}`;
            this.showSuccess(`Daily spending limit ${limitText}`);
            document.querySelector('.modal').remove();
            this.loadSummary();
        } catch (error) {
            console.error('Error updating spending limit:', error);
            this.showError(`Failed to update spending limit: ${error.message}`);
        }
    }

    // Utility methods
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    showInfo(message) {
        this.showToast(message, 'info');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        const container = document.getElementById('toast-notification') || document.body;
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }
}

// Initialize billing manager when DOM is loaded
let billingManager;
document.addEventListener('DOMContentLoaded', () => {
    billingManager = new BillingManager();
});
