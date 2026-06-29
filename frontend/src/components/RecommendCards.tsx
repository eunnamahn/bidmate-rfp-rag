import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchRecommend } from '../api/recommend'
import { naraIngest } from '../api/nara'
import { fetchDocuments } from '../api/documents'
import { useStore } from '../store/useStore'

function hasCompanyInfo(): boolean {
  try { return !!localStorage.getItem('company_info') }
  catch { return false }
}

export default function RecommendCards() {
  const user = useStore((s) => s.user)
  const setDocuments = useStore((s) => s.setDocuments)
  const setActiveFilename = useStore((s) => s.setActiveFilename)
  const [ingesting, setIngesting] = useState<string | null>(null)
  const [ingestError, setIngestError] = useState<string | null>(null)

  const ready = !!user && hasCompanyInfo()

  const { data: cards = [], isLoading, isError } = useQuery({
    queryKey: ['recommend'],
    queryFn: fetchRecommend,
    staleTime: 5 * 60_000,
    retry: 1,
    enabled: ready,
  })

  async function handleIngest(card: typeof cards[0]) {
    if (ingesting === card.id) return
    setIngesting(card.id)
    setIngestError(null)
    try {
      const result = await naraIngest(card.item)
      const docs = await fetchDocuments()
      setDocuments(docs)
      setActiveFilename(result.파일명)
    } catch {
      setIngestError('첨부파일을 가져올 수 없습니다. 다른 공고를 선택해주세요.')
    } finally {
      setIngesting(null)
    }
  }

  if (!ready) {
    return (
      <div className="text-center py-5 space-y-2">
        <p className="text-xs text-gray-500 leading-relaxed">
          로그인 후 회사 정보를 입력하면
          <br />맞춤 공고를 추천해드립니다.
        </p>
        {!user ? (
          <Link
            to="/login"
            className="inline-block text-xs font-medium text-primary-600 hover:text-primary-700 underline underline-offset-2"
          >
            로그인하기
          </Link>
        ) : (
          <Link
            to="/company-setup"
            className="inline-block text-xs font-medium text-primary-600 hover:text-primary-700 underline underline-offset-2"
          >
            회사 정보 입력하기
          </Link>
        )}
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-20 bg-gray-100 rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  if (isError) {
    return (
      <p className="text-xs text-gray-400 text-center py-4">
        나라장터 연결에 실패했습니다.
        <br />API 키 설정을 확인해 주세요.
      </p>
    )
  }

  if (!cards.length) {
    return (
      <p className="text-xs text-gray-400 text-center py-4">
        최근 7일 내 공고가 없습니다.
      </p>
    )
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-gray-400 mb-2">나라장터 최근 공고 · 회사 프로필 매칭 Top 3</p>
      {ingestError && (
        <p className="text-xs text-red-500 bg-red-50 rounded-lg px-3 py-2">{ingestError}</p>
      )}
      {cards.map((card) => (
        <div
          key={card.id}
          className={`bg-white border rounded-lg p-3 transition-colors group cursor-pointer ${
            ingesting === card.id
              ? 'border-primary-300 bg-primary-50/30'
              : 'border-gray-100 hover:border-primary-300'
          }`}
          onClick={() => handleIngest(card)}
        >
          <div className="flex items-start justify-between gap-2 mb-1.5">
            <p className="text-xs font-medium text-gray-800 leading-snug flex-1 group-hover:text-primary-700">
              {card.title}
            </p>
            {ingesting === card.id ? (
              <svg className="flex-shrink-0 w-4 h-4 text-primary-500 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <span className="flex-shrink-0 text-xs font-bold text-primary-600 bg-primary-50 px-1.5 py-0.5 rounded">
                {card.matchScore}%
              </span>
            )}
          </div>
          <p className="text-xs text-gray-400">{card.organization} · ~{card.deadline}</p>
          <p className={`text-xs mt-1 leading-relaxed ${ingesting === card.id ? 'text-primary-600 font-medium' : 'text-gray-500'}`}>
            {ingesting === card.id ? '문서 인덱싱 중...' : card.reason}
          </p>
        </div>
      ))}
    </div>
  )
}
