import { create } from 'zustand'
import { Turn, Document, Agent } from '../types'

interface AppState {
  // auth
  token: string | null
  user: { id: string; email: string; hasCompanyInfo: boolean } | null
  setAuth: (token: string, user: AppState['user']) => void
  clearAuth: () => void
  setHasCompanyInfo: (v: boolean) => void

  // chat sessions
  sessions: Record<string, Turn[]>
  currentSessionId: string
  newSession: () => void
  switchSession: (id: string) => void
  addTurn: (turn: Turn) => void
  clearCurrentSession: () => void

  // active document
  activeFilename: string | null
  setActiveFilename: (fn: string | null) => void

  // pending eligibility flow
  pendingAgent: string | null
  pendingPrompt: string
  setPending: (agent: string | null, prompt?: string) => void

  // sidebar
  sidebarOpen: boolean
  toggleSidebar: () => void

  // documents cache
  documents: Document[]
  setDocuments: (docs: Document[]) => void

  // agents cache
  agents: Agent[]
  setAgents: (agents: Agent[]) => void
}

function generateId() {
  return Math.random().toString(36).slice(2, 10)
}

const initSessionId = generateId()

export const useStore = create<AppState>((set, get) => ({
  // auth (hydrate from localStorage)
  token: localStorage.getItem('token'),
  user: (() => { try { return JSON.parse(localStorage.getItem('user') || 'null') } catch { return null } })(),
  setAuth: (token, user) => {
    localStorage.setItem('token', token)
    localStorage.setItem('user', JSON.stringify(user))
    set({ token, user })
  },
  clearAuth: () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    set({ token: null, user: null })
  },
  setHasCompanyInfo: (v) => {
    const u = get().user
    if (!u) return
    const updated = { ...u, hasCompanyInfo: v }
    localStorage.setItem('user', JSON.stringify(updated))
    set({ user: updated })
  },

  // sessions
  sessions: { [initSessionId]: [] },
  currentSessionId: initSessionId,
  newSession: () => {
    const id = generateId()
    set((s) => ({ sessions: { ...s.sessions, [id]: [] }, currentSessionId: id, activeFilename: null }))
  },
  switchSession: (id) => set({ currentSessionId: id }),
  addTurn: (turn) => {
    set((s) => {
      const id = s.currentSessionId
      return { sessions: { ...s.sessions, [id]: [...(s.sessions[id] || []), turn] } }
    })
  },
  clearCurrentSession: () => {
    set((s) => ({ sessions: { ...s.sessions, [s.currentSessionId]: [] } }))
  },

  activeFilename: null,
  setActiveFilename: (fn) => set({ activeFilename: fn }),

  pendingAgent: null,
  pendingPrompt: '',
  setPending: (agent, prompt = '') => set({ pendingAgent: agent, pendingPrompt: prompt }),

  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  documents: [],
  setDocuments: (docs) => set({ documents: docs }),

  agents: [],
  setAgents: (agents) => set({ agents }),
}))

// computed helper — 현재 세션의 turns
export const useCurrentTurns = () =>
  useStore((s) => s.sessions[s.currentSessionId] || [])
