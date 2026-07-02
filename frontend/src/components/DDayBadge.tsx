import { daysUntil } from '../utils/format'

export default function DDayBadge({ deadline }: { deadline?: string | null }) {
  const diff = daysUntil(deadline)
  if (diff === null) return null

  const label = diff > 0 ? `D-${diff}` : diff === 0 ? 'D-day' : '마감'
  // D-7 이하: 임박(빨강) · D-8~14: 주의(주황) · 그 외: 여유(회색)
  const colorClass =
    diff <= 7 ? 'bg-red-50 text-red-600' : diff <= 14 ? 'bg-orange-50 text-orange-600' : 'bg-gray-100 text-gray-500'

  return (
    <span className={`flex-shrink-0 text-xs font-bold px-1.5 py-0.5 rounded ${colorClass}`}>
      {label}
    </span>
  )
}
