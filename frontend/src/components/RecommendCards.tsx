import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchRecommend } from '../api/recommend'
import { naraIngest } from '../api/nara'
import { fetchDocuments } from '../api/documents'
import { useStore } from '../store/useStore'

export default function RecommendCards() {
  const setDocuments = useStore((s) => s.setDocuments)
  const setActiveFilename = useStore((s) => s.setActiveFilename)
  const [ingesting, setIngesting] = useState<string | null>(null)

  const { data: cards = [], isLoading } = useQuery({
    queryKey: ['recommend'],
    queryFn: fetchRecommend,
    staleTime: 5 * 60_000,
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

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-20 bg-gray-100 rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-gray-400 mb-2">회사 정보 기반 추천 공고 Top 3</p>
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
