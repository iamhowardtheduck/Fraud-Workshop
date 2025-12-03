const express = require('express');
const { Client } = require('@elastic/elasticsearch');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Elasticsearch configuration - Workshop Environment Defaults
const elasticsearchConfig = {
  node: process.env.ELASTICSEARCH_URL || 'http://kubernetes-vm:30920',
  auth: {
    username: process.env.ELASTICSEARCH_USERNAME || 'fraud',
    password: process.env.ELASTICSEARCH_PASSWORD || 'hunter'
  },
  tls: {
    rejectUnauthorized: false // Set to true in production with proper certificates
  }
};

const esClient = new Client(elasticsearchConfig);

// Middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"]
    }
  }
}));

app.use(compression());
app.use(morgan('combined'));
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Serve static files
app.use(express.static('public'));

// Set view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Routes
app.get('/', (req, res) => {
  res.render('index');
});

// API endpoint to get SAR data from Elasticsearch
app.get('/api/sar-reports', async (req, res) => {
  try {
    const { page = 1, size = 10, search } = req.query;
    const from = (page - 1) * size;

    let query = { match_all: {} };
    
    if (search) {
      query = {
        multi_match: {
          query: search,
          fields: [
            'financial_institution_name',
            'suspect_name',
            'suspect_entity_name',
            'account_number',
            'address'
          ]
        }
      };
    }

    const response = await esClient.search({
      index: process.env.ELASTICSEARCH_INDEX || 'sar-reports',
      body: {
        query: query,
        from: from,
        size: parseInt(size),
        sort: [
          { '@timestamp': { order: 'desc' } },
          { 'report_date': { order: 'desc' } }
        ]
      }
    });

    const reports = response.body.hits.hits.map(hit => ({
      id: hit._id,
      ...hit._source
    }));

    res.json({
      reports,
      total: response.body.hits.total.value,
      page: parseInt(page),
      totalPages: Math.ceil(response.body.hits.total.value / size)
    });

  } catch (error) {
    console.error('Error fetching SAR reports:', error);
    res.status(500).json({ 
      error: 'Failed to fetch SAR reports',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// API endpoint to get a specific SAR report
app.get('/api/sar-reports/:id', async (req, res) => {
  try {
    const response = await esClient.get({
      index: process.env.ELASTICSEARCH_INDEX || 'sar-reports',
      id: req.params.id
    });

    res.json({
      id: response.body._id,
      ...response.body._source
    });

  } catch (error) {
    console.error('Error fetching SAR report:', error);
    if (error.meta && error.meta.statusCode === 404) {
      res.status(404).json({ error: 'SAR report not found' });
    } else {
      res.status(500).json({ 
        error: 'Failed to fetch SAR report',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  }
});

// Health check endpoint
app.get('/api/health', async (req, res) => {
  try {
    const health = await esClient.cluster.health();
    res.json({
      status: 'healthy',
      elasticsearch: {
        cluster_status: health.body.status,
        number_of_nodes: health.body.number_of_nodes
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: 'Cannot connect to Elasticsearch',
      timestamp: new Date().toISOString()
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Internal Server Error',
    details: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Not Found' });
});

// Start server
app.listen(PORT, () => {
  console.log(`SAR Web System running on port ${PORT}`);
  console.log(`Access the application at: http://localhost:${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = app;
