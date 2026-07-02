import { formatMoney } from '../utils/format'
import DDayBadge from './DDayBadge'
import IndexingProgress from './IndexingProgress'

interface ProcurementCardProps {
  title: string
  organization: string
  deadline?: string | null
  budget?: number | null
  contractMethod?: string | null
  region?: string | null
  matchScore?: number
  extraTags?: string[]
  ingesting: boolean
  onIngest: () => void
}

export default function ProcurementCard({
  title,
  organization,
  deadline,
  budget,
  contractMethod,
  region,
  matchScore,
  extraTags,
  ingesting,
  onIngest,
}: ProcurementCardProps) {
  // 사업금액·계약방식·수행지역 — 데이터에 없는 항목은 자연히 스킵된다.
  const pills = [formatMoney(budget), contractMethod, region].filter(
    (v): v is string => !!v,
  )

  return (
    <div
      className={`bg-white border rounded-xl p-6 flex flex-col gap-3 transition-colors ${
        ingesting ? 'border-primary-300 bg-primary-50/30' : 'border-gray-100'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-base font-bold text-gray-800 leading-snug line-clamp-2 flex-1">
          {title}
        </p>
        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          {matchScore !== undefined && (
            <span className="text-xs font-bold text-primary-600 bg-primary-50 px-1.5 py-0.5 rounded">
              {matchScore}%
            </span>
          )}
          <DDayBadge deadline={deadline} />
        </div>
      </div>

      <div className="flex items-center gap-1.5 text-sm text-gray-500">
        <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M3 21h18M5 21V7l8-4v18M19 21V11l-6-4M9 9h.01M9 12h.01M9 15h.01" />
        </svg>
        <span className="truncate">{organization || '미상'}</span>
      </div>

      {pills.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {pills.map((p, i) => (
            <span key={i} className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded-full">
              {p}
            </span>
          ))}
        </div>
      )}

      {extraTags && extraTags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {extraTags.map((tag) => (
            <span
              key={tag}
              className="text-[11px] font-medium text-primary-700 bg-primary-50 px-2 py-0.5 rounded-full"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      <button
        className="btn-primary w-full py-2 text-xs mt-auto"
        onClick={onIngest}
        disabled={ingesting}
      >
        {ingesting ? <IndexingProgress /> : '이 공고로 대화하기'}
      </button>
    </div>
  )
}
