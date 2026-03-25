const express = require('express')
const cors = require('cors')
const rateLimit = require('express-rate-limit')

const app = express()
const PORT = 3004

// Security: API Key (environment variable)
const API_KEY = process.env.RAG_API_KEY || 'default-secret-key-change-me'

// Security: Rate limiting
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // 100 requests per minute
  message: 'Too many requests from this IP, please try again later.'
})

// Middleware
app.use(cors())
app.use(limiter)
app.use(express.json({ limit: '1mb' }))

// Security: API Key validation middleware
function validateApiKey(req, res, next) {
  const providedKey = req.headers['x-api-key']

  if (!providedKey || providedKey !== API_KEY) {
    return res.status(401).json({ error: 'Unauthorized: Invalid or missing API key' })
  }

  next()
}

// Apply API key validation to all RAG endpoints
app.use('/api/rag', validateApiKey)

// ============ RAG API ENDPOINTS ============

/**
 * POST /api/rag/search - Vector similarity search
 * Request: { query: string, top_k: number }
 * Response: [ { id, title, snippet, score } ]
 */
app.post('/api/rag/search', async (req, res) => {
  try {
    const { query, top_k = 5 } = req.body

    if (!query || query.trim().length === 0) {
      return res.status(400).json({ error: 'Missing or empty query' })
    }

    console.log(`[RAG API] Search query: "${query}", top_k: ${top_k}`)

    // TODO: Implement actual Vector DB search
    // For now, return mock response
    const mockResults = [
      {
        id: 'paper_123',
        title: 'Sample Paper 1',
        snippet: 'This is a sample snippet from the paper...',
        score: 0.95
      },
      {
        id: 'paper_456',
        title: 'Sample Paper 2',
        snippet: 'Another sample snippet from the paper...',
        score: 0.87
      }
    ]

    res.json({
      query,
      top_k,
      results: mockResults,
      total: mockResults.length
    })

  } catch (err) {
    console.error('[RAG API] Search error:', err)
    res.status(500).json({ error: 'Internal server error' })
  }
})

/**
 * GET /api/rag/browse - Browse papers (paginated)
 * Request: { page: number, limit: number }
 * Response: { papers: [], total: number, page: number }
 */
app.get('/api/rag/browse', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 20

    console.log(`[RAG API] Browse: page=${page}, limit=${limit}`)

    // TODO: Implement actual Vector DB browse
    // For now, return mock response
    const mockPapers = [
      {
        id: 'paper_123',
        title: 'Sample Paper 1',
        authors: 'Author One, Author Two',
        year: '2023',
        venue: 'Journal of Example',
        doi: '10.1234/example.2023.001'
      },
      {
        id: 'paper_456',
        title: 'Sample Paper 2',
        authors: 'Author Three, Author Four',
        year: '2024',
        venue: 'Conference on Example',
        doi: '10.5678/conf.2024.002'
      }
    ]

    const total = mockPapers.length
    const papers = mockPapers.slice((page - 1) * limit, page * limit)

    res.json({
      page,
      limit,
      papers,
      total,
      hasMore: page * limit < total
    })

  } catch (err) {
    console.error('[RAG API] Browse error:', err)
    res.status(500).json({ error: 'Internal server error' })
  }
})

/**
 * GET /api/rag/stats - Collection statistics
 * Response: { collection_count: number, paper_count: number }
 */
app.get('/api/rag/stats', async (req, res) => {
  try {
    console.log('[RAG API] Stats request')

    // TODO: Implement actual Vector DB stats
    // For now, return mock response
    const mockStats = {
      collection_count: 1017, // From actual vector DB
      paper_count: 84, // From actual database
      last_updated: new Date().toISOString()
    }

    res.json(mockStats)

  } catch (err) {
    console.error('[RAG API] Stats error:', err)
    res.status(500).json({ error: 'Internal server error' })
  }
})

// ============ HEALTH CHECK ============

app.get('/api/rag/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'rag-api-wrapper',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  })
})

// ============ ERROR HANDLING ============

app.use((err, req, res, next) => {
  console.error('[RAG API] Error:', err)
  res.status(500).json({ error: 'Internal server error' })
})

// ============ START SERVER ============

app.listen(PORT, '127.0.0.1', () => {
  console.log(`[RAG API] Server running on http://127.0.0.1:${PORT}`)
  console.log(`[RAG API] Health check: http://127.0.0.1:${PORT}/api/rag/health`)
  console.log('[RAG API] Security: API key validation ENABLED')
  console.log('[RAG API] Security: Rate limiting ENABLED (100 req/min)')
})
