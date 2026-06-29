import { api } from './client'
import { NaraNotice } from '../types'

export async function naraSearch(q: string, days: number): Promise<NaraNotice[]> {
  const { data } = await api.get('/nara/search', { params: { q, days }, timeout: 60_000 })
  return data.notices as NaraNotice[]
}

export async function naraIngest(item: Record<string, unknown>): Promise<{ 파일명: string }> {
  const { data } = await api.post('/nara/ingest', { item }, { timeout: 600_000 })
  return data
}
