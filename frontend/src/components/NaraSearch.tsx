import { useState } from 'react'
import { naraSearch, naraIngest } from '../api/nara'
import { fetchDocuments } from '../api/documents'
import { NaraNotice } from '../types'
import { useStore } from '../store/useStore'

export default function NaraSearch() {
  const setDocuments = useStore((s) => s.setDocuments)
  const setActiveFilename = useStore((s) => s.setActiveFilename)
  const [keyword, setKeyword] = useState('')
  const [days, setDays] = useState(3)
  const [results, setResults] = useState<NaraNotice[]>([])
  const [searching, setSearching] = useState(false)
  const [ingesting, setIngesting] = useState<number | null>(null)
  const [error, setError] = useState('')

  async function handleSearch() {
    if (!keyword.trim()) return
    setError('')
    setSearching(true)
    try {
      const notices = await naraSearch(keyword.trim(), days)
      setResults(notices)
    } catch {
      setError('검색에 실패했습니다.')
    } finally {
      setSearching(false)
    }
  }

  async function handleIngest(notice: NaraNotice, idx: number) {
    setIngesting(idx)
    try {
      const data = await naraIngest(notice.item)
      const docs = await fetchDocuments()
      setDocuments(docs)
      setActiveFilename(data.파일명)
    } catch {
      setError('인덱싱에 실패했습니다.')
    } finally {
      setIngesting(null)
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          className="input-field flex-1 text-xs"
          placeholder="공고명 키워드"
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        />
        <input
          type="number" min={1} max={30}
          className="input-field w-14 text-xs text-center"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          title="최근 N일"
        />
      </div>
      <button
        className="btn-primary w-full py-1.5 text-xs"
        onClick={handleSearch}
        disabled={searching || !keyword.trim()}
      >
        {searching ? '검색 중...' : '검색'}
      </button>

      {error && <p className="text-xs text-red-500">{error}</p>}

      {results.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-gray-400">결과 {results.length}건 (클릭하면 인덱싱)</p>
          {results.map((n, i) => (
            <div key={i} className="bg-gray-50 rounded-lg p-2.5 text-xs space-y-1.5">
              <p className="font-medium text-gray-800 leading-snug">{n.사업명 || '(제목 없음)'}</p>
              <p className="text-gray-500">{n.기관 || '미상'} · 마감: {n.마감일 || '미상'}</p>
              <button
                className="btn-primary w-full py-1 text-xs"
                onClick={() => handleIngest(n, i)}
                disabled={ingesting === i}
              >
                {ingesting === i ? '인덱싱 중...' : '인덱싱'}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
