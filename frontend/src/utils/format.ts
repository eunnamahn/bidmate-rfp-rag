export function formatMoney(amount: number | null | undefined): string | null {
  if (amount === null || amount === undefined || !Number.isFinite(amount) || amount <= 0) return null
  return `${amount.toLocaleString('ko-KR')}원`
}

// 마감일(YYYY-MM-DD 등)까지 남은 일수. 파싱 불가면 null.
export function daysUntil(deadline: string | null | undefined): number | null {
  if (!deadline) return null
  const target = new Date(deadline)
  if (Number.isNaN(target.getTime())) return null
  target.setHours(0, 0, 0, 0)
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  return Math.round((target.getTime() - today.getTime()) / 86_400_000)
}
