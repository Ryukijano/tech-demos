import type { Question, SystemOneRequest, SystemOneResponse, Answer } from './types'

function softmax(scores: number[]): number[] {
  const max = Math.max(...scores)
  const exps = scores.map((s) => Math.exp(s - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

function hashSeed(text: string): number {
  let h = 2166136261
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function seededUnit(seed: number, salt: number): number {
  const x = Math.sin(seed * 0.0001 + salt * 12.9898) * 43758.5453
  return x - Math.floor(x)
}

function scoreText(state: string, label: string, desc: string): number {
  const hay = state.toLowerCase()
  const needles = `${label} ${desc}`.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)
  let score = 0.15
  for (const n of needles) {
    if (n.length < 3) continue
    if (hay.includes(n)) score += 1.2
  }
  // Mild lexical overlap bonus for common support routing words
  const hints: Record<string, string[]> = {
    billing: ['charge', 'charged', 'refund', 'invoice', 'payment', 'bill'],
    shipping: ['delivery', 'tracking', 'parcel', 'ship', 'late', 'lost'],
    technical: ['bug', 'error', 'login', 'crash', 'password'],
  }
  for (const [key, words] of Object.entries(hints)) {
    if (label.toLowerCase().includes(key) || desc.toLowerCase().includes(key)) {
      for (const w of words) {
        if (hay.includes(w)) score += 1.5
      }
    }
  }
  return score
}

function mockAnswer(_name: string, q: Question, state: string, seed: number): Answer {
  switch (q.type) {
    case 'choice': {
      const keys = Object.keys(q.criteria)
      const raw = keys.map((k, i) => scoreText(state, k, q.criteria[k]) + seededUnit(seed, i) * 0.3)
      const probs = softmax(raw)
      const probabilities: Record<string, number> = {}
      keys.forEach((k, i) => {
        probabilities[k] = Number(probs[i].toFixed(4))
      })
      const topIdx = probs.indexOf(Math.max(...probs))
      const choice = keys[topIdx]
      const sorted = [...probs].sort((a, b) => b - a)
      const confidence = Number((sorted[0] - (sorted[1] ?? 0)).toFixed(4))
      return { type: 'choice', choice, probabilities, confidence }
    }
    case 'score': {
      const levels = q.criteria
      const angryBoost = /angry|urgent|nobody|twice|asap|immediately/i.test(state) ? 1.4 : 0
      const raw = levels.map((_, i) => i * 0.55 + angryBoost * (i / Math.max(levels.length - 1, 1)) + seededUnit(seed, i + 7) * 0.4)
      const probs = softmax(raw)
      const probabilities: Record<string, number> = {}
      const legend: Record<string, string> = {}
      let score = 0
      levels.forEach((label, i) => {
        probabilities[String(i)] = Number(probs[i].toFixed(4))
        legend[String(i)] = label
        score += i * probs[i]
      })
      return {
        type: 'score',
        score: Number(score.toFixed(4)),
        legend,
        probabilities,
        confidence: Number((Math.max(...probs) - 1 / levels.length).toFixed(4)),
      }
    }
    case 'noul': {
      const cues = /angry|furious|frustrated|nobody|twice|broken|hate|upset/i.test(state)
      const base = cues ? 0.78 : 0.32
      const noul = Number(Math.min(0.97, Math.max(0.05, base + (seededUnit(seed, 99) - 0.5) * 0.12)).toFixed(4))
      return { type: 'noul', noul }
    }
    default: {
      const _exhaustive: never = q
      return _exhaustive
    }
  }
}

/** Deterministic offline System One response — clearly labeled MOCK in the UI. */
export function mockSystemOne(req: SystemOneRequest): SystemOneResponse {
  const seed = hashSeed(req.state + JSON.stringify(req.questions))
  const answers: Record<string, Answer> = {}
  for (const [name, q] of Object.entries(req.questions)) {
    answers[name] = mockAnswer(name, q, req.state, seed)
  }
  return {
    model: req.model ?? 'mock/Julia-1',
    answers,
    usage: { input_tokens: Math.max(8, Math.round(req.state.length / 4)), output_tokens: 0 },
    _mock: true,
  }
}
