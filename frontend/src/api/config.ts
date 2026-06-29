import { api } from './client'
import { Config } from '../types'

export async function fetchConfig(): Promise<Config> {
  const { data } = await api.get<Config>('/config')
  return data
}

export async function applyConfig(payload: Record<string, unknown>): Promise<{ reindexed: boolean }> {
  const { data } = await api.post('/config', payload, { timeout: 1800_000 })
  return data
}

export async function fetchHealth() {
  const { data } = await api.get('/health', { timeout: 5_000 })
  return data as { status: string; vectors?: number }
}
