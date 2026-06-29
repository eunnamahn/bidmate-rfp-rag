import { useState, useEffect } from 'react'
import { useStore } from '../store/useStore'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '../api/config'
import { fetchDocuments, fileViewUrl, fileDownloadUrl } from '../api/documents'
import NaraSearch from './NaraSearch'
import RecommendCards from './RecommendCards'
import Settings from './Settings'

type Tab = 'docs' | 'nara' | 'recommend' | 'settings'

export default function Sidebar() {
  const [tab, setTab] = useState<Tab>('docs')
  const [expandDocs, setExpandDocs] = useState(true)

  const sessions = useStore((s) => s.sessions)
  const currentSessionId = useStore((s) => s.currentSessionId)
  const newSession = useStore((s) => s.newSession)
  const switchSession = useStore((s) => s.switchSession)
  const clearCurrentSession = useStore((s) => s.clearCurrentSession)
  const documents = useStore((s) => s.documents)
  const setDocuments = useStore((s) => s.setDocuments)
  const activeFilename = useStore((s) => s.activeFilename)
  const setActiveFilename = useStore((s) => s.setActiveFilename)

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
    retry: false,
  })

  const { data: docsData } = useQuery({
    queryKey: ['documents'],
    queryFn: fetchDocuments,
    staleTime: 120_000,
  })
  useEffect(() => { if (docsData) setDocuments(docsData) }, [docsData, setDocuments])

  const sessionList = Object.keys(sessions)

  function docLabel(fn: string) {
    const d = documents.find((d) => d.파일명 === fn)
    if (!d) return fn
    return `${d.사업명 || fn} · ${d.기관 || '미상'}`
  }

  const TAB_ITEMS: { id: Tab; icon: React.ReactNode; label: string }[] = [
    {
      id: 'docs',
      label: '문서',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
    {
      id: 'nara',
      label: '나라장터',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      ),
    },
    {
      id: 'recommend',
      label: 'AI 추천',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
        </svg>
      ),
    },
    {
      id: 'settings',
      label: '설정',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      ),
    },
  ]

  return (
    <aside className="flex flex-col w-64 bg-gray-50 border-r border-gray-100 h-full flex-shrink-0">
      {/* 헤더 */}
      <div className="px-4 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2.5 mb-3">
          <div className="w-7 h-7 rounded-lg bg-primary-600 flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div className="min-w-0">
            <p className="font-bold text-gray-900 text-sm">입찰메이트</p>
            <p className="text-xs text-gray-400 truncate">B2G 입찰지원 AI</p>
          </div>
        </div>

        {/* 백엔드 상태 */}
        <div className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded-md ${
          health?.status === 'ok' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full ${health?.status === 'ok' ? 'bg-green-500' : 'bg-red-500'}`} />
          {health?.status === 'ok'
            ? `연결됨 · 벡터 ${(health.vectors || 0).toLocaleString()}개`
            : health ? '백엔드 준비 중' : '백엔드 확인 중...'}
        </div>

        <button
          onClick={newSession}
          className="mt-3 w-full flex items-center gap-2 justify-center py-2 text-xs font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          새 채팅
        </button>
      </div>

      {/* 대화 히스토리 */}
      <div className="px-3 py-3 border-b border-gray-100">
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wide px-1 mb-2">대화</p>
        <div className="space-y-0.5 max-h-32 overflow-y-auto">
          {sessionList.map((sid, i) => {
            const turns = sessions[sid]
            const label = turns[0]?.question
              ? turns[0].question.slice(0, 28) + (turns[0].question.length > 28 ? '...' : '')
              : `새 대화 ${i + 1}`
            return (
              <button
                key={sid}
                onClick={() => switchSession(sid)}
                className={`w-full text-left px-2 py-1.5 rounded-md text-xs transition-colors truncate ${
                  sid === currentSessionId
                    ? 'bg-primary-50 text-primary-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {label}
              </button>
            )
          })}
        </div>
        {sessions[currentSessionId]?.length > 0 && (
          <button
            onClick={clearCurrentSession}
            className="mt-2 w-full text-xs text-gray-400 hover:text-red-500 py-1 transition-colors"
          >
            대화 초기화
          </button>
        )}
      </div>

      {/* 탭 네비게이션 */}
      <div className="flex border-b border-gray-100">
        {TAB_ITEMS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 flex flex-col items-center py-2 gap-0.5 text-[10px] transition-colors ${
              tab === t.id
                ? 'text-primary-600 border-b-2 border-primary-600'
                : 'text-gray-400 hover:text-gray-600'
            }`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* 탭 콘텐츠 */}
      <div className="flex-1 overflow-y-auto px-3 py-3">
        {tab === 'docs' && (
          <div className="space-y-3">
            <div>
              <button
                className="flex items-center justify-between w-full text-xs font-medium text-gray-600 mb-2"
                onClick={() => setExpandDocs((v) => !v)}
              >
                <span>현재 문서</span>
                <svg className={`w-3.5 h-3.5 transition-transform ${expandDocs ? 'rotate-0' : '-rotate-90'}`}
                  fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {expandDocs && (
                <div className="space-y-1">
                  <button
                    onClick={() => setActiveFilename(null)}
                    className={`w-full text-left px-2 py-1.5 rounded-md text-xs transition-colors ${
                      !activeFilename ? 'bg-primary-50 text-primary-700 font-medium' : 'text-gray-500 hover:bg-gray-100'
                    }`}
                  >
                    전체 문서 대상
                  </button>
                  {documents.map((d) => (
                    <div key={d.파일명}>
                      <button
                        onClick={() => setActiveFilename(d.파일명)}
                        className={`w-full text-left px-2 py-1.5 rounded-md text-xs transition-colors truncate ${
                          activeFilename === d.파일명
                            ? 'bg-primary-50 text-primary-700 font-medium'
                            : 'text-gray-500 hover:bg-gray-100'
                        }`}
                      >
                        {docLabel(d.파일명)}
                      </button>
                      {activeFilename === d.파일명 && (
                        <div className="flex gap-2 px-2 pb-1">
                          <a href={fileViewUrl(d.파일명)} target="_blank" rel="noreferrer"
                            className="text-[10px] text-primary-600 hover:underline">보기</a>
                          <a href={fileDownloadUrl(d.파일명)} target="_blank" rel="noreferrer"
                            className="text-[10px] text-primary-600 hover:underline">다운로드</a>
                        </div>
                      )}
                    </div>
                  ))}
                  {documents.length === 0 && (
                    <p className="text-xs text-gray-400 px-2 py-1">인덱싱된 문서가 없습니다.</p>
                  )}
                </div>
              )}
            </div>

          </div>
        )}

        {tab === 'nara' && <NaraSearch />}
        {tab === 'recommend' && <RecommendCards />}
        {tab === 'settings' && <Settings />}
      </div>

    </aside>
  )
}
