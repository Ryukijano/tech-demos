import { useEffect, useState, type FormEvent } from 'react'

type Hit = { id: string; title: string; text: string; score: number }
type SearchResponse = {
  query: string
  mode: string
  model: string
  dim: number
  hits: Hit[]
}
type Health = { ok: boolean; mode: string; model: string; dim: number; corpus_size: number }

const API = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:43122'

export default function App() {
  const [query, setQuery] = useState('What causes the northern lights?')
  const [apiUrl, setApiUrl] = useState(API)
  const [health, setHealth] = useState<Health | null>(null)
  const [result, setResult] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${apiUrl}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [apiUrl])

  async function onSearch(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setResult((await res.json()) as SearchResponse)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const mode = result?.mode ?? health?.mode

  return (
    <div className="app">
      <h1 className="brand">EmbeddingGemma 2</h1>
      <p className="sub">
        Text → 768-d similarity search over a tiny local corpus. Uses{' '}
        <code>SentenceTransformer("google/embeddinggemma-2")</code> with vision/audio dropped
        when available.
      </p>

      {mode === 'mock' && (
        <div className="banner mock" role="status">
          <strong>MOCK</strong> — hash embedder (model download unavailable). Rankings are demo-only.
        </div>
      )}
      {mode === 'live' && (
        <div className="banner live" role="status">
          <strong>LIVE</strong> — {result?.model ?? health?.model} · dim {result?.dim ?? health?.dim}
        </div>
      )}

      <form className="panel" onSubmit={onSearch}>
        <label htmlFor="api">API base</label>
        <input id="api" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} />
        <label htmlFor="q" style={{ marginTop: '0.85rem' }}>
          Query
        </label>
        <textarea id="q" value={query} onChange={(e) => setQuery(e.target.value)} />
        <div className="row">
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? 'Embedding…' : 'Search'}
          </button>
          <span className="hint">
            corpus {health?.corpus_size ?? '—'} docs · cosine similarity
          </span>
        </div>
        {error && <p className="err">{error}</p>}
      </form>

      <section className="panel">
        {result?.hits?.length ? (
          result.hits.map((h, i) => (
            <article className="hit" key={h.id} style={{ animationDelay: `${i * 0.05}s` }}>
              <div className="score">{h.score.toFixed(3)}</div>
              <div>
                <h3>{h.title}</h3>
                <p>{h.text}</p>
              </div>
            </article>
          ))
        ) : (
          <p className="hint">Run a search to rank matches.</p>
        )}
      </section>
    </div>
  )
}
