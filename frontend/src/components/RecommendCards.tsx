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

  const ready = !!user && hasCompanyInfo()

  const { data: cards = [], isLoading, isError } = useQuery({
    queryKey: ['recommend'],
    queryFn: fetchRecommend,
    staleTime: 5 * 60_000,
    retry: 1,
    enabled: ready,
  })

  async function handleIngest(card: typeof cards[0]) {
    setIngesting(card.id)
    try {
      const data = await naraIngest(card.item)
      const docs = await fetchDocuments()
      setDocuments(docs)
      setActiveFilename(data.파일명)
    } catch {
      // silent
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
      {cards.map((card) => (
        <div
          key={card.id}
          className="bg-white border border-gray-100 rounded-lg p-3 hover:border-primary-300 transition-colors cursor-pointer group"
          onClick={() => handleIngest(card)}
        >
          <div className="flex items-start justify-between gap-2 mb-1.5">
            <p className="text-xs font-medium text-gray-800 leading-snug flex-1 group-hover:text-primary-700">
              {card.title}
            </p>
            <span className="flex-shrink-0 text-xs font-bold text-primary-600 bg-primary-50 px-1.5 py-0.5 rounded">
              {card.matchScore}%
            </span>
          </div>
          <p className="text-xs text-gray-400">{card.organization} · ~{card.deadline}</p>
          <p className="text-xs text-gray-500 mt-1 leading-relaxed">{card.reason}</p>
          {ingesting === card.id && (
            <p className="text-xs text-primary-600 mt-1.5 font-medium">인덱싱 중...</p>
          )}
        </div>
      ))}
    </div>
  )
}
