import type { SystemOneRequest, SystemOneResponse } from './types'
import { mockSystemOne } from './mock'

export type CallOptions = {
  baseUrl: string
  forceMock?: boolean
  timeoutMs?: number
}

export async function callSystemOne(
  req: SystemOneRequest,
  opts: CallOptions,
): Promise<{ data: SystemOneResponse; mode: 'live' | 'mock'; reason?: string }> {
  if (opts.forceMock) {
    return { data: mockSystemOne(req), mode: 'mock', reason: 'Mock mode enabled' }
  }

  const url = `${opts.baseUrl.replace(/\/$/, '')}/v1/systemone`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? 4000)

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: controller.signal,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`)
    }
    const data = (await res.json()) as SystemOneResponse
    return { data, mode: 'live' }
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err)
    return {
      data: mockSystemOne(req),
      mode: 'mock',
      reason: `Server unreachable (${reason}). Showing MOCK.`,
    }
  } finally {
    clearTimeout(timer)
  }
}
