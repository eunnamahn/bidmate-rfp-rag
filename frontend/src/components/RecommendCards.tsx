import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchRecommend } from '../api/recommend'
import {
  naraIngest,
  isNaraNoContentError,
  NARA_NO_CONTENT_PREFIX,
  NARA_NO_CONTENT_LINK_TEXT,
} from '../api/nara'
import { fetchDocuments } from '../api/documents'
import { useStore } from '../store/useStore'
import { useInfiniteReveal } from '../hooks/useInfiniteReveal'
import ProcurementCard from './ProcurementCard'

const PAGE_SIZE = 12

export default function RecommendCards() {
  const user = useStore((s) => s.user)
  const setDocuments = useStore((s) => s.setDocuments)
  const setActiveFilename = useStore((s) => s.setActiveFilename)
  const setMainView = useStore((s) => s.setMainView)
  const [ingesting, setIngesting] = useState<string | null>(null)
  const [ingestError, setIngestError] = useState<
    { noContent: true; detailUrl?: string } | { noContent: false; message: string } | null
  >(null)

  // 회사 정보 등록 여부는 store의 user.hasCompanyInfo(백엔드 /company 기준)로 판단한다.
  const ready = !!user && !!user.hasCompanyInfo

  // queryKey에 user.id를 반드시 포함한다 — QueryClient가 앱 전역 싱글턴이라
  // 키가 유저별로 분리되지 않으면 로그아웃 후 다른 계정으로 로그인해도
  // 이전 계정의 추천 결과가 staleTime(5분) 동안 그대로 재사용된다.
  const { data: cards = [], isLoading, isError } = useQuery({
    queryKey: ['recommend', user?.id],
    queryFn: fetchRecommend,
    staleTime: 5 * 60_000,
    retry: 1,
    enabled: ready,
  })

  const { visibleCount, sentinelRef, reset } = useInfiniteReveal(PAGE_SIZE)
  useEffect(() => { reset() }, [cards, reset])

  async function handleIngest(card: typeof cards[0]) {
    if (ingesting === card.id) return
    setIngesting(card.id)
    setIngestError(null)
    try {
      const result = await naraIngest(card.item)
      const docs = await fetchDocuments()
      setDocuments(docs)
      setActiveFilename(result.파일명)
      setMainView('chat')
    } catch (e) {
      if (isNaraNoContentError(e)) {
        setIngestError({ noContent: true, detailUrl: (card.item.bidNtceDtlUrl as string) || undefined })
      } else {
        setIngestError({ noContent: false, message: '첨부파일을 가져올 수 없습니다. 다른 공고를 선택해주세요.' })
      }
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
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {Array.from({ length: PAGE_SIZE }).map((_, i) => (
          <div key={i} className="h-56 bg-gray-100 rounded-xl animate-pulse" />
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

  const visibleCards = cards.slice(0, visibleCount)

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-400 mb-1">나라장터 최근 공고 · 회사 프로필 매칭순</p>
      {ingestError && (
        <p className="text-sm text-red-500 bg-red-50 rounded-lg px-3 py-2">
          {ingestError.noContent ? (
            <>
              {NARA_NO_CONTENT_PREFIX}
              {ingestError.detailUrl ? (
                <a
                  href={ingestError.detailUrl}
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
            ingestError.message
          )}
        </p>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {visibleCards.map((card) => (
          <ProcurementCard
            key={card.id}
            title={card.title}
            organization={card.organization}
            deadline={card.deadline}
            budget={card.budget}
            contractMethod={card.contractMethod}
            region={card.region}
            matchScore={card.matchScore}
            extraTags={card.reasonTags}
            ingesting={ingesting === card.id}
            onIngest={() => handleIngest(card)}
          />
        ))}
      </div>
      {visibleCount < cards.length && (
        <div ref={sentinelRef} className="py-4 text-center text-xs text-gray-300">
          더 불러오는 중...
        </div>
      )}
    </div>
  )
}
