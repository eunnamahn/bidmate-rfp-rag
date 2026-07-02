import { useStore } from '../store/useStore'

export default function ActiveDocumentBadge() {
  const activeFilename = useStore((s) => s.activeFilename)
  const setActiveFilename = useStore((s) => s.setActiveFilename)
  const documents = useStore((s) => s.documents)

  if (!activeFilename) {
    return <span className="text-sm text-gray-400">전체 문서 대상으로 검색 중</span>
  }

  const doc = documents.find((d) => d.파일명 === activeFilename)
  const label = doc ? `${doc.사업명 || activeFilename} · ${doc.기관 || '미상'}` : activeFilename

  return (
    <div className="flex items-center gap-2 min-w-0">
      <svg className="w-4 h-4 text-primary-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <span className="text-sm font-medium text-gray-800 truncate">{label}</span>
      <span className="text-xs bg-primary-50 text-primary-700 px-2 py-0.5 rounded-full flex-shrink-0">분석 중</span>
      <button
        onClick={() => setActiveFilename(null)}
        title="문서 해제"
        className="flex-shrink-0 w-4 h-4 rounded-full hover:bg-gray-200 text-gray-400 hover:text-gray-600 flex items-center justify-center transition-colors"
      >
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )
}
