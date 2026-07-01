import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Paperclip } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { useStore, useCurrentTurns } from '../store/useStore'
import { sendChat, sendInfographic, fetchAgents } from '../api/chat'
import { fetchDocuments, uploadFile } from '../api/documents'
import { getCompany } from '../api/company'
import { Agent, Turn, ChatResponse, CompanyInfo } from '../types'
import Sidebar from '../components/Sidebar'
import MessageBubble from '../components/MessageBubble'
import ActionBar from '../components/ActionBar'

function useContextFilename() {
  const activeFilename = useStore((s) => s.activeFilename)
  const turns = useCurrentTurns()

  if (activeFilename) return activeFilename

  for (let i = turns.length - 1; i >= 0; i--) {
    const sources = (turns[i].data as ChatResponse).sources || []
    const names = new Set(sources.map((s) => s.파일명).filter(Boolean))
    if (names.size === 1) return [...names][0]!
    if (names.size > 1) return null
  }
  return null
}

type ModalType = 'login' | 'no-company' | null

function CompanyInfoModal({ type, onClose }: { type: ModalType; onClose: () => void }) {
  const navigate = useNavigate()
  if (!type) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={onClose}>
      <div
        className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-sm mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="w-9 h-9 rounded-xl bg-amber-50 flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="font-semibold text-gray-800">
            {type === 'login' ? '로그인이 필요합니다' : '회사 정보가 필요합니다'}
          </h3>
        </div>
        <p className="text-sm text-gray-500 mb-5 leading-relaxed">
          {type === 'login'
            ? '이 기능을 사용하려면 로그인 후 회사 정보를 입력해주세요.'
            : '적격성 판정 및 리스크 분석에는 회사 정보가 필요합니다.\n먼저 회사 정보를 입력해주세요.'}
        </p>
        <div className="flex gap-2">
          <button className="btn-secondary flex-1 text-sm" onClick={onClose}>닫기</button>
          <button
            className="btn-primary flex-1 text-sm"
            onClick={() => {
              onClose()
              navigate(type === 'login' ? '/login' : '/company-setup')
            }}
          >
            {type === 'login' ? '로그인하기' : '회사 정보 입력'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Chat() {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [modal, setModal] = useState<ModalType>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const sidebarOpen = useStore((s) => s.sidebarOpen)
  const toggleSidebar = useStore((s) => s.toggleSidebar)
  const addTurn = useStore((s) => s.addTurn)
  const user = useStore((s) => s.user)
  const documents = useStore((s) => s.documents)
  const setDocuments = useStore((s) => s.setDocuments)
  const agents = useStore((s) => s.agents)
  const setAgents = useStore((s) => s.setAgents)
  const activeFilename = useStore((s) => s.activeFilename)
  const setActiveFilename = useStore((s) => s.setActiveFilename)

  const turns = useCurrentTurns()
  const contextFilename = useContextFilename()

  const { data: docsData } = useQuery({
    queryKey: ['documents-init'],
    queryFn: fetchDocuments,
    staleTime: 120_000,
  })
  useEffect(() => { if (docsData) setDocuments(docsData) }, [docsData, setDocuments])

  const { data: agentsData } = useQuery({
    queryKey: ['agents'],
    queryFn: fetchAgents,
    staleTime: 300_000,
  })
  useEffect(() => { if (agentsData) setAgents(agentsData) }, [agentsData, setAgents])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, loading])

  function buildHistory(): [string, string][] {
    return turns.map((t) => [t.question, (t.data as ChatResponse).answer || ''])
  }

  async function doSend(
    question: string,
    agent?: string,
    filename?: string,
    companyInfo?: CompanyInfo,
  ) {
    const fn = filename ?? activeFilename ?? undefined
    const history = buildHistory()
    const payload = {
      question,
      history,
      ...(agent ? { agent } : {}),
      ...(fn ? { filename: fn } : {}),
      ...(companyInfo ? { company_info: companyInfo } : {}),
    }
    const data = await sendChat(payload)
    addTurn({ kind: 'chat', question, data })
    return data
  }

  async function handleSend() {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    setLoading(true)
    try {
      if (q.includes('인포그래픽')) {
        const fn = contextFilename ?? undefined
        const data = await sendInfographic({ question: q, ...(fn ? { filename: fn } : {}) })
        addTurn({ kind: 'infographic', question: q, data })
      } else if (q.includes('평가구조') || q.includes('평가배점 구조') || q.includes('다이어그램')) {
        await doSend(q, 'evaluation', contextFilename ?? undefined)
      } else {
        await doSend(q)
      }
    } catch {
      addTurn({ kind: 'chat', question: q, data: { answer: '오류가 발생했습니다. 백엔드를 확인해주세요.', refused: true } })
    } finally {
      setLoading(false)
    }
  }

  async function handleQuickSend(question: string, agent: string) {
    if (loading) return
    setLoading(true)
    try {
      await doSend(question, agent, contextFilename ?? undefined)
    } catch {
      addTurn({ kind: 'chat', question, data: { answer: '오류가 발생했습니다. 백엔드를 확인해주세요.', refused: true } })
    } finally {
      setLoading(false)
    }
  }

  const handleAgentAction = useCallback(async (agent: Agent) => {
    if (loading) return

    const question = agent.prompt || agent.label
    const fn = contextFilename ?? undefined

    // needs_company_info 에이전트: 회사 정보 자동 주입
    if (agent.needs_company_info) {
      if (!user) {
        setModal('login')
        return
      }
      const companyInfo = await getCompany()
      if (!companyInfo) {
        setModal('no-company')
        return
      }
      setLoading(true)
      try {
        if (agent.endpoint === '/infographic') {
          const data = await sendInfographic({ question, ...(fn ? { filename: fn } : {}) })
          addTurn({ kind: 'infographic', question, data })
        } else {
          await doSend(question, agent.id, fn, companyInfo)
        }
      } catch {
        addTurn({ kind: 'chat', question, data: { answer: '에이전트 실행 중 오류가 발생했습니다.', refused: true } })
      } finally {
        setLoading(false)
      }
      return
    }

    // 일반 에이전트
    setLoading(true)
    try {
      if (agent.endpoint === '/infographic') {
        const data = await sendInfographic({ question, ...(fn ? { filename: fn } : {}) })
        addTurn({ kind: 'infographic', question, data })
      } else {
        await doSend(question, agent.id, fn)
      }
    } catch {
      addTurn({ kind: 'chat', question, data: { answer: '에이전트 실행 중 오류가 발생했습니다.', refused: true } })
    } finally {
      setLoading(false)
    }
  }, [loading, contextFilename, user])

  // 📎 파일 업로드
  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    setUploading(true)
    try {
      const result = await uploadFile(file)
      const docs = await fetchDocuments()
      setDocuments(docs)
      setActiveFilename(result.파일명)
    } catch {
      addTurn({
        kind: 'chat',
        question: `[파일 업로드: ${file.name}]`,
        data: { answer: '파일 업로드에 실패했습니다. 백엔드를 확인해주세요.', refused: true },
      })
    } finally {
      setUploading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function autoResize(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  function activeDocLabel() {
    if (!activeFilename) return null
    const d = documents.find((d) => d.파일명 === activeFilename)
    return d ? `${d.사업명 || activeFilename} · ${d.기관 || '미상'}` : activeFilename
  }

  const showActionBar = !!contextFilename && agents.length > 0

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      <CompanyInfoModal type={modal} onClose={() => setModal(null)} />

      {sidebarOpen && <Sidebar />}

      <div className="flex flex-col flex-1 min-w-0">
        {/* 헤더 */}
        <header className="flex items-center gap-3 px-4 py-3 border-b border-gray-100 flex-shrink-0">
          <button
            onClick={toggleSidebar}
            className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors"
            title={sidebarOpen ? '사이드바 닫기' : '사이드바 열기'}
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          {/* 현재 문서 표시 — flex-1로 남는 공간 차지 */}
          <div className="flex-1 min-w-0">
            {activeFilename ? (
              <div className="flex items-center gap-2 min-w-0">
                <svg className="w-4 h-4 text-primary-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="text-sm font-medium text-gray-800 truncate">{activeDocLabel()}</span>
                <span className="text-xs bg-primary-50 text-primary-700 px-2 py-0.5 rounded-full flex-shrink-0">분석 중</span>
              </div>
            ) : (
              <span className="text-sm text-gray-400">전체 문서 대상으로 검색 중</span>
            )}
          </div>

          {/* 로그인/로그아웃 — 오른쪽 고정 */}
          {user ? (
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-xs text-gray-400 hidden sm:block truncate max-w-[140px]">{user.email}</span>
              <button
                onClick={() => useStore.getState().clearAuth()}
                className="text-xs text-gray-400 hover:text-red-500 border border-gray-200 hover:border-red-200 px-2.5 py-1.5 rounded-lg transition-colors"
              >
                로그아웃
              </button>
            </div>
          ) : (
            <Link
              to="/login"
              className="flex-shrink-0 flex items-center gap-1.5 text-xs font-medium text-white bg-primary-600 hover:bg-primary-700 px-3 py-1.5 rounded-lg transition-colors"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
              </svg>
              로그인
            </Link>
          )}
        </header>

        {/* 채팅 영역 */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {turns.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="w-16 h-16 rounded-2xl bg-primary-50 flex items-center justify-center mb-4">
                  <svg className="w-9 h-9 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-2">안녕하세요, 입찰메이트입니다</h2>
                <p className="text-sm text-gray-500 max-w-sm leading-relaxed">
                  RFP 공고에 대해 무엇이든 물어보세요.<br />
                  <Paperclip className="inline w-3.5 h-3.5 mb-0.5" /> 버튼으로 문서를 업로드하거나 사이드바 나라장터에서<br />
                  공고를 불러온 후 에이전트 버튼으로 분석을 시작하세요.
                </p>
                <div className="grid grid-cols-2 gap-3 mt-8 w-full max-w-sm">
                  {[
                    { q: 'RFP 핵심 요약해줘', icon: '📄', agent: 'summary' },
                    { q: '입찰 참가 자격 확인해줘', icon: '✅', agent: 'eligibility' },
                    { q: '리스크 요소 알려줘', icon: '⚠️', agent: 'risk' },
                    { q: '평가 배점 구조 보여줘', icon: '📊', agent: 'evaluation' },
                  ].map((item) => (
                    <button
                      key={item.q}
                      onClick={() => handleQuickSend(item.q, item.agent)}
                      disabled={loading}
                      className="text-left p-3 bg-gray-50 hover:bg-primary-50 border border-gray-100 hover:border-primary-200 rounded-xl text-sm text-gray-600 hover:text-primary-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      <span className="block mb-1">{item.icon}</span>
                      {item.q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {turns.map((turn, i) => {
              const isLast = i === turns.length - 1
              return (
                <MessageBubble
                  key={i}
                  turn={turn}
                  actionBar={isLast && showActionBar && !loading
                    ? <ActionBar agents={agents} onAction={handleAgentAction} disabled={loading} />
                    : undefined
                  }
                />
              )
            })}

            {loading && (
              <div className="flex justify-start">
                <div className="flex gap-2.5">
                  <div className="flex-shrink-0 w-7 h-7 rounded-full bg-primary-100 flex items-center justify-center">
                    <svg className="w-4 h-4 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                    <div className="flex gap-1 items-center">
                      <div className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </div>

        {/* 하단 입력 영역 */}
        <div className="flex-shrink-0 px-4 py-4 border-t border-gray-100 bg-white">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2 bg-gray-50 border border-gray-200 rounded-2xl px-3 py-3 focus-within:border-primary-400 focus-within:ring-2 focus-within:ring-primary-600/10 transition-all">
              {/* 📎 파일 업로드 버튼 */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading || loading}
                title="HWP · HWPX · PDF 업로드"
                className="flex-shrink-0 w-8 h-8 rounded-lg hover:bg-gray-200 disabled:opacity-40 text-gray-400 hover:text-gray-600 flex items-center justify-center transition-colors"
              >
                {uploading ? (
                  <svg className="w-4 h-4 animate-spin text-primary-500" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                ) : (
                  <Paperclip size={16} />
                )}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".hwp,.hwpx,.pdf"
                className="hidden"
                onChange={handleFileChange}
              />

              <textarea
                ref={textareaRef}
                rows={1}
                className="flex-1 bg-transparent resize-none outline-none text-sm text-gray-800 placeholder-gray-400 max-h-40 py-0.5"
                placeholder="RFP 요약, 공고 검색, 적격성 판정 등 무엇이든 물어보세요"
                value={input}
                onChange={autoResize}
                onKeyDown={handleKeyDown}
                disabled={loading}
              />

              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="flex-shrink-0 w-8 h-8 rounded-xl bg-primary-600 hover:bg-primary-700 disabled:bg-gray-200 text-white flex items-center justify-center transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                </svg>
              </button>
            </div>
            <p className="text-center text-xs text-gray-300 mt-2 flex items-center justify-center gap-1">
              Enter로 전송 · Shift+Enter로 줄바꿈 ·
              <Paperclip className="inline w-3 h-3" />
              로 HWP·PDF 업로드
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
