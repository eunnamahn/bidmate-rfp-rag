import { useEffect, useState } from 'react'
import {
  naraSearch,
  naraIngest,
  isNaraNoContentError,
  NARA_NO_CONTENT_PREFIX,
  NARA_NO_CONTENT_LINK_TEXT,
} from '../api/nara'
import { fetchDocuments } from '../api/documents'
import { NaraNotice } from '../types'
import { useStore } from '../store/useStore'
import { useInfiniteReveal } from '../hooks/useInfiniteReveal'
import ProcurementCard from './ProcurementCard'

// UI에 노출하지 않는 내부 배치 크기 — 한 번에 넉넉히 받아 스크롤로 순차 공개한다.
const FETCH_SIZE = 100
const PAGE_SIZE = 12

export default function NaraSearch() {
  const setDocuments = useStore((s) => s.setDocuments)
  const setActiveFilename = useStore((s) => s.setActiveFilename)
  const setMainView = useStore((s) => s.setMainView)
  const [keyword, setKeyword] = useState('')
  const [days, setDays] = useState(3)
  const [results, setResults] = useState<NaraNotice[]>([])
  const [searching, setSearching] = useState(false)
  const [ingesting, setIngesting] = useState<number | null>(null)
  const [error, setError] = useState<
    { noContent: true; detailUrl?: string } | { noContent: false; message: string } | null
  >(null)

  const { visibleCount, sentinelRef, reset } = useInfiniteReveal(PAGE_SIZE)

  async function runSearch(q: string) {
    setError(null)
    setSearching(true)
    try {
      const notices = await naraSearch(q.trim(), days, FETCH_SIZE)
      setResults(notices)
      reset()
    } catch {
      setError({ noContent: false, message: '검색에 실패했습니다.' })
    } finally {
      setSearching(false)
    }
  }

  // 진입 시 키워드 없이 최신 공고를 바로 채워 보여준다.
  useEffect(() => {
    runSearch('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleIngest(notice: NaraNotice, idx: number) {
    setIngesting(idx)
    setError(null)
    try {
      const data = await naraIngest(notice.item)
      const docs = await fetchDocuments()
      setDocuments(docs)
      setActiveFilename(data.파일명)
      setMainView('chat')
    } catch (e) {
      if (isNaraNoContentError(e)) {
        setError({ noContent: true, detailUrl: (notice.item.bidNtceDtlUrl as string) || undefined })
      } else {
        setError({ noContent: false, message: '인덱싱에 실패했습니다.' })
      }
    } finally {
      setIngesting(null)
    }
  }

  const visibleResults = results.slice(0, visibleCount)

  return (
    <div className="space-y-4">
      <div className="flex gap-2 max-w-lg">
        <input
          className="input-field flex-1 text-sm"
          placeholder="공고명 키워드 (비워두면 최신 공고)"
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && runSearch(keyword)}
        />
        <input
          type="number" min={1} max={30}
          className="input-field w-16 text-sm text-center"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          title="최근 N일"
        />
        <button
          className="btn-primary px-4 text-sm flex-shrink-0"
          onClick={() => runSearch(keyword)}
          disabled={searching}
        >
          {searching ? '검색 중...' : '검색'}
        </button>
      </div>

      {error && (
        <p className="text-sm text-red-500">
          {error.noContent ? (
            <>
              {NARA_NO_CONTENT_PREFIX}
              {error.detailUrl ? (
                <a
                  href={error.detailUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline underline-offset-2 font-medium"
                >
                  {NARA_NO_CONTENT_LINK_TEXT}
                </a>
              ) : (
                NARA_NO_CONTENT_LINK_TEXT
              )}
            </>
          ) : (
            error.message
          )}
        </p>
      )}

      {searching && results.length === 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Array.from({ length: PAGE_SIZE }).map((_, i) => (
            <div key={i} className="h-56 bg-gray-100 rounded-xl animate-pulse" />
          ))}
        </div>
      )}

      {!searching && results.length === 0 && !error && (
        <p className="text-sm text-gray-400 text-center py-8">조건에 맞는 공고가 없습니다.</p>
      )}

      {visibleResults.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs text-gray-400">결과 {results.length}건</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {visibleResults.map((n, i) => (
              <ProcurementCard
                key={i}
                title={n.사업명 || '(제목 없음)'}
                organization={n.기관 || ''}
                deadline={n.마감일}
                budget={n.금액}
                contractMethod={(n.item.cntrctCnclsMthdNm as string) || null}
                region={(n.item.rgnLmtBidLocplcJdgmBssNm as string) || null}
                ingesting={ingesting === i}
                onIngest={() => handleIngest(n, i)}
              />
            ))}
          </div>
          {visibleCount < results.length && (
            <div ref={sentinelRef} className="py-4 text-center text-xs text-gray-300">
              더 불러오는 중...
            </div>
          )}
        </div>
      )}
    </div>
  )
}
