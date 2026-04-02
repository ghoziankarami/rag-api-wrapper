const { z } = require('zod')

// Query validation schema
const querySchema = z.object({
  query: z.string().min(1).max(1000).describe('Search query text'),
  top_k: z.preprocess(
    (val) => (val === undefined ? 5 : Number(val)),
    z.number().int().min(1).max(100).default(5)
  )
})

// Search request body validation
const searchRequestBodySchema = z.object({
  query: z.string().min(1).max(1000),
  top_k: z.preprocess(
    (val) => (val === undefined ? 5 : Number(val)),
    z.number().int().min(1).max(100).default(5)
  )
})

// Answer request body validation
const answerRequestBodySchema = z.object({
  query: z.string().min(1).max(1000),
  top_k: z.preprocess(
    (val) => (val === undefined ? 5 : Number(val)),
    z.number().int().min(1).max(100).default(5)
  ),
  history: z.array(z.object({
    role: z.enum(['user', 'assistant']),
    content: z.string().max(2000),
  })).max(10).optional().default([]),
})

// Browse query params validation
const browseQuerySchema = z.object({
  page: z.preprocess(
    (val) => (val === undefined ? 1 : Number(val)),
    z.number().int().positive().default(1)
  ),
  limit: z.preprocess(
    (val) => (val === undefined ? 20 : Number(val)),
    z.number().int().min(1).max(100).default(20)
  )
})

// Stats query params validation (for refresh)
const statsQuerySchema = z.object({
  refresh: z.preprocess(
    (val) => (val === undefined ? false : String(val) === '1'),
    z.boolean().default(false)
  )
})

// Environment config validation
const envConfigSchema = z.object({
  PORT: z.preprocess(
    (val) => (val === undefined ? 3004 : Number(val)),
    z.number().int().min(1).max(65535).default(3004)
  ),
  RAG_API_KEY: z.string().min(1).default('default-secret-key-change-me'),
  RAG_STATS_TTL_MS: z.preprocess(
    (val) => (val === undefined ? 60000 : Number(val)),
    z.number().int().positive().default(60000)
  )
})

function validateSearchBody(body) {
  const result = searchRequestBodySchema.safeParse(body)
  if (!result.success) {
    throw {
      type: 'validation_error',
      errors: result.error.errors,
      message: 'Invalid search request body'
    }
  }
  return result.data
}

function validateAnswerBody(body) {
  const result = answerRequestBodySchema.safeParse(body)
  if (!result.success) {
    throw {
      type: 'validation_error',
      errors: result.error.errors,
      message: 'Invalid answer request body'
    }
  }
  return result.data
}

function validateBrowseQuery(query) {
  const result = browseQuerySchema.safeParse(query)
  if (!result.success) {
    throw {
      type: 'validation_error',
      errors: result.error.errors,
      message: 'Invalid browse query parameters'
    }
  }
  return result.data
}

function validateStatsQuery(query) {
  const result = statsQuerySchema.safeParse(query)
  if (!result.success) {
    throw {
      type: 'validation_error',
      errors: result.error.errors,
      message: 'Invalid stats query parameters'
    }
  }
  return result.data
}

function validateEnvConfig(env) {
  const result = envConfigSchema.safeParse(env)
  if (!result.success) {
    console.error('[ZOD Validation] Environment config validation failed:')
    result.error.errors.forEach((err) => {
      console.error(`  - ${err.path.join('.')}: ${err.message}`)
    })
    // Return defaults if validation fails, but log errors
    return envConfigSchema.parse({})
  }
  return result.data
}

function validateQueryString(queryStr) {
  const result = querySchema.safeParse({ query: queryStr })
  if (!result.success) {
    throw {
      type: 'validation_error',
      errors: result.error.errors,
      message: 'Invalid query parameter'
    }
  }
  return result.data
}

module.exports = {
  validateSearchBody,
  validateAnswerBody,
  validateBrowseQuery,
  validateStatsQuery,
  validateEnvConfig,
  validateQueryString,
  schemas: {
    searchRequestBody: searchRequestBodySchema,
    answerRequestBody: answerRequestBodySchema,
    browseQuery: browseQuerySchema,
    statsQuery: statsQuerySchema,
    envConfig: envConfigSchema,
    query: querySchema
  }
}
