require('dotenv').config()
const express = require('express')
const cors = require('cors')
const rateLimit = require('express-rate-limit')
const { execFileSync } = require('child_process')
const path = require('path')
const validation = require('./validation')

const app = express()
const rawEnv = process.env
const envConfig = validation.validateEnvConfig(rawEnv)
const PORT = Number(envConfig.PORT || 3004)
const DATA_PROVIDER = path.join(__dirname, 'rag_public_data.py')
const STATS_TTL_MS = Number(envConfig.RAG_STATS_TTL_MS || 60_000)

app.set('trust proxy', 1)

// Security: API Key (environment variable)
const API_KEY = envConfig.RAG_API_KEY || 'default-secret-key-change-me'

// Security: Rate limiting
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 100,
  message: 'Too many requests from this IP, please try again later.'
})

app.use(cors())
app.use(limiter)
app.use(express.json({ limit: '1mb' }))

function isLoopbackRequest(req) {
  const socketIp = req.socket?.remoteAddress || ''
  const forwardedIp = req.ip || ''
  const loopbacks = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1'])
  return loopbacks.has(socketIp) || loopbacks.has(forwardedIp)
}

function validateApiKey(req, res, next) {
  if (isLoopbackRequest(req)) {
    return next()
  }

  const providedKey = req.headers['x-api-key']

  if (!providedKey || providedKey !== API_KEY) {
    return res.status(401).json({ error: 'Unauthorized: Invalid or missing API key' })
  }

  next()
}

app.use('/api/rag', validateApiKey)

const statsCache = {
  data: null,
  expiresAt: 0,
}

function runProvider(command, args = []) {
  const output = execFileSync('python3', [DATA_PROVIDER, command, ...args], {
    cwd: __dirname,
    encoding: 'utf8',
    maxBuffer: 20 * 1024 * 1024,
    env: process.env,
  })

  const trimmed = String(output || '').trim()
  if (!trimmed) {
    throw new Error('Empty response from data provider')
  }

  try {
    return JSON.parse(trimmed)
  } catch (error) {
    throw new Error(`Invalid JSON from data provider: ${trimmed.slice(0, 400)}`)
  }
}

function getStatsCached({ force = false } = {}) {
  const now = Date.now()
  if (!force && statsCache.data && statsCache.expiresAt > now) {
    return statsCache.data
  }

  const data = runProvider('stats')
  statsCache.data = data
  statsCache.expiresAt = now + STATS_TTL_MS
  return data
}

app.post('/api/rag/search', (req, res) => {
  try {
    const validated = validation.validateSearchBody(req.body)
    const { query, top_k } = validated

    const data = runProvider('search', ['--query', query, '--top-k', String(top_k)])
    // Normalize provider response to ensure type safety
    const results = Array.isArray(data.results) ? data.results : []
    const rawTotal = data.total
    const total = (rawTotal != null && !isNaN(Number(rawTotal))) ? Number(rawTotal) : results.length
    return res.json({
      query: query,
      top_k: Number(top_k),
      results,
      total,
    })
  } catch (err) {
    console.error('[RAG API] Search error:', err)
    if (err.type === 'validation_error') {
      return res.status(400).json({ error: 'Validation error', details: err.errors, message: err.message })
    }
    return res.status(500).json({ error: 'Internal server error', message: String(err.message || err) })
  }
})

app.get('/api/rag/browse', (req, res) => {
  try {
    const validated = validation.validateBrowseQuery(req.query)
    const { page, limit } = validated
    const data = runProvider('browse', ['--page', String(page), '--limit', String(limit)])
    // Normalize: ensure papers is an array
    const normalized = {
      ...data,
      papers: Array.isArray(data.papers) ? data.papers : []
    }
    return res.json(normalized)
  } catch (err) {
    console.error('[RAG API] Browse error:', err)
    if (err.type === 'validation_error') {
      return res.status(400).json({ error: 'Validation error', details: err.errors, message: err.message })
    }
    return res.status(500).json({ error: 'Internal server error', message: String(err.message || err) })
  }
})

app.get('/api/rag/stats', (req, res) => {
  try {
    const validated = validation.validateStatsQuery(req.query)
    const { force } = validated
    const data = getStatsCached({ force })
    return res.json(data)
  } catch (err) {
    console.error('[RAG API] Stats error:', err)
    if (err.type === 'validation_error') {
      return res.status(400).json({ error: 'Validation error', details: err.errors, message: err.message })
    }
    return res.status(500).json({ error: 'Internal server error', message: String(err.message || err) })
  }
})

app.post('/api/rag/answer', (req, res) => {
  try {
    const validated = validation.validateAnswerBody(req.body)
    const { query, top_k, history } = validated

    const args = ['--query', query, '--top-k', String(top_k)]
    if (history && history.length > 0) {
      args.push('--history', JSON.stringify(history))
    }

    const data = runProvider('answer', args)
    // Normalize response to ensure frontend safety
    const normalized = {
      ...data,
      answer: typeof data.answer === 'string' ? data.answer : '',
      sources: Array.isArray(data.sources) ? data.sources : []
    }
    return res.json(normalized)
  } catch (err) {
    console.error('[RAG API] Answer error:', err)
    if (err.type === 'validation_error') {
      return res.status(400).json({ error: 'Validation error', details: err.errors, message: err.message })
    }
    return res.status(500).json({ error: 'Internal server error', message: String(err.message || err) })
  }
})

app.get('/api/rag/health', (req, res) => {
  try {
    const stats = getStatsCached()
    return res.json({
      status: 'healthy',
      service: 'rag-api-wrapper',
      version: stats.version || '2.0.0',
      timestamp: new Date().toISOString(),
      mode: stats.mode || 'public_read_only',
      llm_model: stats.llm_model || null,
      llm_ready: Boolean(stats.llm_ready),
      corpus: {
        indexed_papers: stats.indexed_papers || 0,
        summary_count: stats.summary_count || 0,
        collection_count: stats.collection_count || 0,
      },
      cache_ttl_ms: STATS_TTL_MS,
    })
  } catch (err) {
    console.error('[RAG API] Health error:', err)
    return res.json({
      status: 'degraded',
      service: 'rag-api-wrapper',
      timestamp: new Date().toISOString(),
      error: String(err.message || err),
    })
  }
})

app.use((err, req, res, next) => {
  console.error('[RAG API] Error:', err)
  res.status(500).json({ error: 'Internal server error' })
})

app.listen(PORT, '127.0.0.1', () => {
  console.log(`[RAG API] Server running on http://127.0.0.1:${PORT}`)
  console.log(`[RAG API] Health check: http://127.0.0.1:${PORT}/api/rag/health`)
  console.log('[RAG API] Security: API key validation ENABLED')
  console.log('[RAG API] Security: Rate limiting ENABLED (100 req/min)')
})
