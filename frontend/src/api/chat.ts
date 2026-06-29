import { api } from './client'
import { ChatResponse, InfographicResponse, CompanyInfo } from '../types'

export interface ChatPayload {
  question: string
  history: [string, string][]
  agent?: string
  filename?: string
  company_info?: CompanyInfo
}

export async function sendChat(payload: ChatPayload): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/chat', payload)
  return data
}

export async function sendInfographic(payload: { question: string; filename?: string }): Promise<InfographicResponse> {
  const { data } = await api.post<InfographicResponse>('/infographic', payload)
  return data
}

export async function fetchAgents() {
  const { data } = await api.get('/agents')
  return data.agents as import('../types').Agent[]
}
