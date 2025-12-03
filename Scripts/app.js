// SAR Management System - Frontend Application

class SARApplication {
  constructor() {
    this.currentPage = 1;
    this.pageSize = 10;
    this.currentSearch = '';
    this.reports = [];
    this.totalPages = 0;
    
    this.init();
  }

  init() {
    this.bindEvents();
    this.checkSystemHealth();
    this.loadReports();
    
    // Auto-refresh system health every 5 minutes
    setInterval(() => this.checkSystemHealth(), 5 * 60 * 1000);
  }

  bindEvents() {
    // Search functionality
    document.getElementById('searchBtn').addEventListener('click', () => this.performSearch());
    document.getElementById('searchInput').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') this.performSearch();
    });
    document.getElementById('clearSearch').addEventListener('click', () => this.clearSearch());

    // Page size change
    document.getElementById('pageSize').addEventListener('change', (e) => {
      this.pageSize = parseInt(e.target.value);
      this.currentPage = 1;
      this.loadReports();
    });

    // Health check
    document.getElementById('healthCheck').addEventListener('click', () => this.checkSystemHealth());

    // Modal controls
    document.getElementById('modalClose').addEventListener('click', () => this.closeModal());
    document.getElementById('modalOverlay').addEventListener('click', (e) => {
      if (e.target.id === 'modalOverlay') this.closeModal();
    });

    // ESC key to close modal
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.closeModal();
    });
  }

  async checkSystemHealth() {
    try {
      const response = await fetch('/api/health');
      const health = await response.json();
      
      const statusIndicator = document.getElementById('statusIndicator');
      const statusDot = statusIndicator.querySelector('.status-dot');
      const statusText = statusIndicator.querySelector('.status-text');

      if (response.ok && health.status === 'healthy') {
        statusDot.className = 'status-dot healthy';
        statusText.textContent = 'System Healthy';
      } else {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'System Issues';
      }
    } catch (error) {
      const statusIndicator = document.getElementById('statusIndicator');
      const statusDot = statusIndicator.querySelector('.status-dot');
      const statusText = statusIndicator.querySelector('.status-text');
      
      statusDot.className = 'status-dot error';
      statusText.textContent = 'Connection Error';
    }
  }

  showLoading() {
    document.getElementById('loadingSpinner').classList.add('active');
  }

  hideLoading() {
    document.getElementById('loadingSpinner').classList.remove('active');
  }

  async loadReports() {
    this.showLoading();
    
    try {
      const params = new URLSearchParams({
        page: this.currentPage,
        size: this.pageSize,
        ...(this.currentSearch && { search: this.currentSearch })
      });

      const response = await fetch(`/api/sar-reports?${params}`);
      const data = await response.json();

      if (response.ok) {
        this.reports = data.reports;
        this.totalPages = data.totalPages;
        this.renderReports();
        this.renderPagination();
        this.updateResultsInfo(data.total);
      } else {
        this.showError('Failed to load reports: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      this.showError('Connection error: ' + error.message);
    } finally {
      this.hideLoading();
    }
  }

  renderReports() {
    const grid = document.getElementById('reportsGrid');
    
    if (this.reports.length === 0) {
      grid.innerHTML = `
        <div style="text-align: center; padding: 40px; color: #64748b;">
          <p>No reports found.</p>
          ${this.currentSearch ? '<p>Try adjusting your search criteria.</p>' : ''}
        </div>
      `;
      return;
    }

    grid.innerHTML = this.reports.map(report => this.renderReportCard(report)).join('');

    // Add click listeners to report cards
    grid.querySelectorAll('.report-card').forEach((card, index) => {
      card.addEventListener('click', () => this.showReportDetails(this.reports[index]));
    });
  }

  renderReportCard(report) {
    const institutionName = report.financial_institution_name || 'Unknown Institution';
    const suspectName = this.getSuspectDisplayName(report);
    const amount = this.formatCurrency(report.total_dollar_amount);
    const date = this.formatDate(report.suspicious_activity_date);
    const accountNumber = report.account_number || 'N/A';

    return `
      <div class="report-card" data-id="${report.id}">
        <div class="report-header">
          <div class="report-title">${this.escapeHtml(institutionName)}</div>
          <div class="report-subtitle">Report ID: ${this.escapeHtml(report.id || 'Unknown')}</div>
        </div>
        <div class="report-body">
          <div class="report-fields">
            <div class="field-group">
              <div class="field-label">Suspect Name</div>
              <div class="field-value">${this.escapeHtml(suspectName)}</div>
            </div>
            <div class="field-group">
              <div class="field-label">Amount Involved</div>
              <div class="field-value amount">${amount}</div>
            </div>
            <div class="field-group">
              <div class="field-label">Activity Date</div>
              <div class="field-value">${date}</div>
            </div>
            <div class="field-group">
              <div class="field-label">Account Number</div>
              <div class="field-value">${this.escapeHtml(accountNumber)}</div>
            </div>
            <div class="field-group">
              <div class="field-label">Institution Address</div>
              <div class="field-value">${this.formatAddress(report, 'institution')}</div>
            </div>
            <div class="field-group">
              <div class="field-label">Suspect Address</div>
              <div class="field-value">${this.formatAddress(report, 'suspect')}</div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  getSuspectDisplayName(report) {
    if (report.suspect_entity_name) {
      return report.suspect_entity_name;
    }
    
    const firstName = report.suspect_first_name || '';
    const lastName = report.suspect_last_name || '';
    
    if (firstName || lastName) {
      return `${firstName} ${lastName}`.trim();
    }
    
    return 'Unknown';
  }

  formatAddress(report, type) {
    let address = '';
    
    if (type === 'institution') {
      const street = report.financial_institution_address || '';
      const city = report.financial_institution_city || '';
      const state = report.financial_institution_state || '';
      const zip = report.financial_institution_zip || '';
      
      address = [street, city, state, zip].filter(Boolean).join(', ');
    } else if (type === 'suspect') {
      const street = report.suspect_address || '';
      const city = report.suspect_city || '';
      const state = report.suspect_state || '';
      const zip = report.suspect_zip || '';
      
      address = [street, city, state, zip].filter(Boolean).join(', ');
    }
    
    return address || 'N/A';
  }

  formatCurrency(amount) {
    if (!amount && amount !== 0) return 'N/A';
    
    const numAmount = typeof amount === 'string' ? parseFloat(amount) : amount;
    
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(numAmount);
  }

  formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch {
      return dateStr;
    }
  }

  renderPagination() {
    const pagination = document.getElementById('pagination');
    
    if (this.totalPages <= 1) {
      pagination.innerHTML = '';
      return;
    }

    let paginationHTML = '';
    
    // Previous button
    paginationHTML += `
      <button ${this.currentPage === 1 ? 'disabled' : ''} data-page="${this.currentPage - 1}">
        Previous
      </button>
    `;
    
    // Page numbers
    const startPage = Math.max(1, this.currentPage - 2);
    const endPage = Math.min(this.totalPages, this.currentPage + 2);
    
    if (startPage > 1) {
      paginationHTML += `<button data-page="1">1</button>`;
      if (startPage > 2) {
        paginationHTML += `<span style="padding: 8px;">...</span>`;
      }
    }
    
    for (let i = startPage; i <= endPage; i++) {
      paginationHTML += `
        <button class="${i === this.currentPage ? 'active' : ''}" data-page="${i}">
          ${i}
        </button>
      `;
    }
    
    if (endPage < this.totalPages) {
      if (endPage < this.totalPages - 1) {
        paginationHTML += `<span style="padding: 8px;">...</span>`;
      }
      paginationHTML += `<button data-page="${this.totalPages}">${this.totalPages}</button>`;
    }
    
    // Next button
    paginationHTML += `
      <button ${this.currentPage === this.totalPages ? 'disabled' : ''} data-page="${this.currentPage + 1}">
        Next
      </button>
    `;
    
    pagination.innerHTML = paginationHTML;
    
    // Add click listeners
    pagination.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const page = parseInt(e.target.dataset.page);
        if (page && page !== this.currentPage) {
          this.currentPage = page;
          this.loadReports();
        }
      });
    });
  }

  updateResultsInfo(total) {
    const start = (this.currentPage - 1) * this.pageSize + 1;
    const end = Math.min(this.currentPage * this.pageSize, total);
    
    document.getElementById('resultsInfo').textContent = 
      `Showing ${start}-${end} of ${total} reports`;
  }

  performSearch() {
    this.currentSearch = document.getElementById('searchInput').value.trim();
    this.currentPage = 1;
    this.loadReports();
  }

  clearSearch() {
    document.getElementById('searchInput').value = '';
    this.currentSearch = '';
    this.currentPage = 1;
    this.loadReports();
  }

  async showReportDetails(report) {
    try {
      // Fetch full report details
      const response = await fetch(`/api/sar-reports/${report.id}`);
      const fullReport = await response.json();

      if (response.ok) {
        this.renderReportModal(fullReport);
        document.getElementById('modalOverlay').classList.add('active');
      } else {
        this.showError('Failed to load report details: ' + (fullReport.error || 'Unknown error'));
      }
    } catch (error) {
      this.showError('Failed to load report details: ' + error.message);
    }
  }

  renderReportModal(report) {
    const modalBody = document.getElementById('modalBody');
    
    modalBody.innerHTML = `
      <div class="modal-fields">
        ${this.renderModalSection('Financial Institution Information', [
          { label: 'Name (Field 2)', value: report.financial_institution_name },
          { label: 'EIN (Field 3)', value: report.financial_institution_ein },
          { label: 'Address (Field 4)', value: report.financial_institution_address },
          { label: 'City (Field 6)', value: report.financial_institution_city },
          { label: 'State (Field 7)', value: report.financial_institution_state },
          { label: 'Zip Code (Field 8)', value: report.financial_institution_zip }
        ])}
        
        ${this.renderModalSection('Branch Information', [
          { label: 'Branch Address (Field 9)', value: report.branch_address },
          { label: 'Branch City (Field 10)', value: report.branch_city },
          { label: 'Branch State (Field 11)', value: report.branch_state },
          { label: 'Branch Zip Code (Field 12)', value: report.branch_zip }
        ])}
        
        ${this.renderModalSection('Account Information', [
          { label: 'Account Number(s) (Field 14)', value: report.account_number }
        ])}
        
        ${this.renderModalSection('Suspect Information', [
          { label: 'Last Name/Entity Name (Field 15)', value: report.suspect_last_name || report.suspect_entity_name },
          { label: 'First Name (Field 16)', value: report.suspect_first_name },
          { label: 'Address (Field 18)', value: report.suspect_address },
          { label: 'City (Field 20)', value: report.suspect_city },
          { label: 'State (Field 21)', value: report.suspect_state },
          { label: 'Zip Code (Field 22)', value: report.suspect_zip },
          { label: 'Phone Number (Field 24)', value: report.suspect_phone }
        ])}
        
        ${this.renderModalSection('Activity Information', [
          { label: 'Date/Range of Activity (Field 33)', value: this.formatActivityDateRange(report) },
          { label: 'Total Dollar Amount (Field 34)', value: this.formatCurrency(report.total_dollar_amount), class: 'amount' }
        ])}
        
        ${report.activity_description ? this.renderModalSection('Activity Description', [
          { label: 'Description', value: report.activity_description, fullWidth: true }
        ]) : ''}
      </div>
    `;
  }

  renderModalSection(title, fields) {
    const fieldsHTML = fields.map(field => {
      const value = field.value || 'N/A';
      const fieldClass = field.class || '';
      const width = field.fullWidth ? 'grid-column: 1 / -1;' : '';
      
      return `
        <div class="field-group" style="${width}">
          <div class="field-label">${field.label}</div>
          <div class="field-value ${fieldClass} ${value === 'N/A' ? 'empty' : ''}">${this.escapeHtml(value)}</div>
        </div>
      `;
    }).join('');

    return `
      <div class="modal-section">
        <div class="modal-section-title">${title}</div>
        <div class="modal-field-group">
          ${fieldsHTML}
        </div>
      </div>
    `;
  }

  formatActivityDateRange(report) {
    const startDate = report.suspicious_activity_date_start || report.suspicious_activity_date;
    const endDate = report.suspicious_activity_date_end;
    
    if (!startDate && !endDate) return 'N/A';
    
    const formatDate = (date) => {
      if (!date) return null;
      try {
        return new Date(date).toLocaleDateString('en-US');
      } catch {
        return date;
      }
    };
    
    const start = formatDate(startDate);
    const end = formatDate(endDate);
    
    if (start && end && start !== end) {
      return `${start} to ${end}`;
    }
    
    return start || end || 'N/A';
  }

  closeModal() {
    document.getElementById('modalOverlay').classList.remove('active');
  }

  showError(message) {
    // Simple error display - in production, use a proper notification system
    alert(message);
  }

  escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new SARApplication();
});
