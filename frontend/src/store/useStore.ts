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
  addTurn: (sessionId: string, turn: Turn) => void
  clearCurrentSession: () => void

  // 백그라운드에서 응답이 도착했지만 아직 안 본 세션 — 사이드바에 살짝 표시만 하고
  // 화면을 강제로 옮기지는 않는다.
  sessionUnread: Record<string, boolean>

  // per-session async state — keyed by sessionId so a response landing after
  // the user switched sessions doesn't show up as the wrong session's loading state
  sessionLoading: Record<string, boolean>
  sessionPendingQuestion: Record<string, string | null>
  setSessionLoading: (sessionId: string, loading: boolean) => void
  setSessionPendingQuestion: (sessionId: string, question: string | null) => void

  // active document
  activeFilename: string | null
  setActiveFilename: (fn: string | null) => void

  // main content area — nav-driven full-view switch
  mainView: 'chat' | 'nara' | 'recommend'
  setMainView: (v: AppState['mainView']) => void

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
  switchSession: (id) => {
    set((s) => ({ currentSessionId: id, sessionUnread: { ...s.sessionUnread, [id]: false } }))
  },
  addTurn: (sessionId, turn) => {
    set((s) => {
      // 그 세션의 채팅 화면을 "지금 실제로 보고 있는" 경우에만 안 읽음 표시가 필요 없다.
      // currentSessionId가 같아도 나라장터/AI추천으로 이동해 있으면 안 읽음으로 표시한다.
      const viewingThisChat = sessionId === s.currentSessionId && s.mainView === 'chat'
      return {
        sessions: { ...s.sessions, [sessionId]: [...(s.sessions[sessionId] || []), turn] },
        sessionUnread: viewingThisChat
          ? s.sessionUnread
          : { ...s.sessionUnread, [sessionId]: true },
      }
    })
  },
  clearCurrentSession: () => {
    set((s) => ({ sessions: { ...s.sessions, [s.currentSessionId]: [] } }))
  },

  sessionUnread: {},

  sessionLoading: {},
  sessionPendingQuestion: {},
  setSessionLoading: (sessionId, loading) => {
    set((s) => ({ sessionLoading: { ...s.sessionLoading, [sessionId]: loading } }))
  },
  setSessionPendingQuestion: (sessionId, question) => {
    set((s) => ({ sessionPendingQuestion: { ...s.sessionPendingQuestion, [sessionId]: question } }))
  },

  activeFilename: null,
  setActiveFilename: (fn) => set({ activeFilename: fn }),

  mainView: 'chat',
  setMainView: (v) => {
    set((s) => ({
      mainView: v,
      // 채팅 화면으로 돌아오면 지금 보고 있게 될 세션의 안 읽음 표시를 지운다.
      sessionUnread: v === 'chat' ? { ...s.sessionUnread, [s.currentSessionId]: false } : s.sessionUnread,
    }))
  },

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

// computed helper — 현재 세션의 loading / pendingQuestion (세션별로 분리됨)
export const useCurrentLoading = () =>
  useStore((s) => s.sessionLoading[s.currentSessionId] ?? false)

export const useCurrentPendingQuestion = () =>
  useStore((s) => s.sessionPendingQuestion[s.currentSessionId] ?? null)
