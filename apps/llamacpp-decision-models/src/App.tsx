import { useState, type FormEvent } from 'react'
import { callSystemOne } from './api'
import type { Answer, SystemOneRequest, SystemOneResponse } from './types'

const DEFAULT_STATE =
  'Customer message: I was charged twice for my order last week and nobody has replied.'

const DEFAULT_QUESTIONS: SystemOneRequest['questions'] = {
  route: {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: {
      billing: 'payments, charges, refunds, invoices',
      shipping: 'delivery, tracking, lost or late parcels',
      technical: 'bugs, errors, login problems',
    },
  },
  angry: {
    type: 'noul',
    instructions: 'Is the customer angry?',
  },
  urgency: {
    type: 'score',
    instructions: 'How urgent is this?',
    criteria: ['can wait', 'this week', 'today', 'right now'],
  },
}

function ProbBars({
  probabilities,
  legend,
}: {
  probabilities: Record<string, number>
  legend?: Record<string, string>
}) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1])
  return (
    <div className="bars">
      {entries.map(([key, p]) => (
        <div className="bar-row" key={key}>
          <span>{legend?.[key] ?? key}</span>
          <div className="track">
            <div className="fill" style={{ width: `${Math.max(2, p * 100)}%` }} />
          </div>
          <span className="pct">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

function AnswerCard({ name, answer }: { name: string; answer: Answer }) {
  switch (answer.type) {
    case 'choice':
      return (
        <article className="answer">
          <h3>{name}</h3>
          <p className="meta">
            top: <strong>{answer.choice}</strong>
            {answer.confidence != null ? ` · confidence ${answer.confidence.toFixed(3)}` : ''}
          </p>
          <ProbBars probabilities={answer.probabilities} />
        </article>
      )
    case 'score':
      return (
        <article className="answer">
          <h3>{name}</h3>
          <p className="meta">
            expected level: <strong>{answer.score.toFixed(3)}</strong>
            {answer.confidence != null ? ` · confidence ${answer.confidence.toFixed(3)}` : ''}
          </p>
          <ProbBars probabilities={answer.probabilities} legend={answer.legend} />
        </article>
      )
    case 'noul':
      return (
        <article className="answer">
          <h3>{name}</h3>
          <p className="meta">
            P(yes): <strong>{(answer.noul * 100).toFixed(1)}%</strong>
          </p>
          <ProbBars probabilities={{ yes: answer.noul, no: 1 - answer.noul }} />
        </article>
      )
    default: {
      const _exhaustive: never = answer
      return _exhaustive
    }
  }
}

export default function App() {
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:8080')
  const [model, setModel] = useState('ggml-org/Julia-1-GGUF:Q8_0')
  const [state, setState] = useState(DEFAULT_STATE)
  const [forceMock, setForceMock] = useState(true)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SystemOneResponse | null>(null)
  const [mode, setMode] = useState<'live' | 'mock' | null>(null)
  const [reason, setReason] = useState<string | undefined>()
  const [error, setError] = useState<string | null>(null)

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    const req: SystemOneRequest = {
      model,
      state,
      questions: DEFAULT_QUESTIONS,
    }
    try {
      const { data, mode: m, reason: r } = await callSystemOne(req, {
        baseUrl,
        forceMock,
      })
      setResult(data)
      setMode(m)
      setReason(r)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <h1 className="brand">Decision Models</h1>
      <p className="tagline">
        Score typed options against a state via llama.cpp <code>POST /v1/systemone</code>.
        Prefer Julia-1 (144M) for local demos. Mock mode needs no GPU.
      </p>

      {mode === 'mock' && (
        <div className="banner mock" role="status">
          <strong>MOCK</strong>
          <span>{reason ?? 'Offline heuristic probabilities — not a real decision model.'}</span>
        </div>
      )}
      {mode === 'live' && (
        <div className="banner live" role="status">
          <strong>LIVE</strong>
          <span>Response from {result?.model ?? baseUrl}</span>
        </div>
      )}

      <form className="panel" onSubmit={onSubmit}>
        <div className="row">
          <label>
            Base URL
            <input
              type="url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://127.0.0.1:8080"
            />
          </label>
          <label>
            Model id
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="ggml-org/Julia-1-GGUF:Q8_0"
            />
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={forceMock}
              onChange={(e) => setForceMock(e.target.checked)}
            />
            Force MOCK mode
          </label>
        </div>

        <label style={{ marginTop: '1rem' }}>
          State
          <textarea value={state} onChange={(e) => setState(e.target.value)} />
        </label>

        <div className="row" style={{ marginTop: '1rem' }}>
          <button className="btn" type="submit" disabled={loading}>
            {loading ? 'Scoring…' : 'Score questions'}
          </button>
          <button
            className="btn btn-ghost"
            type="button"
            onClick={() => setState(DEFAULT_STATE)}
          >
            Reset sample
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </form>

      <section className="panel">
        <p className="meta" style={{ marginTop: 0 }}>
          Fixed demo questions: choice (route) · noul (angry) · score (urgency)
        </p>
        {result ? (
          <div className="answers">
            {Object.entries(result.answers).map(([name, answer]) => (
              <AnswerCard key={name} name={name} answer={answer} />
            ))}
          </div>
        ) : (
          <p className="meta">Run a request to see option probabilities.</p>
        )}
      </section>

      <p className="footer">
        Upstream: ggml-org Decision Models blog · Julia-1 / Laya / Kev-4B / lev / OpenJev
      </p>
    </div>
  )
}
