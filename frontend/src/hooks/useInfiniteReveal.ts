import { useCallback, useEffect, useRef, useState } from 'react'

// 이미 한 번에 넉넉히 받아온 배열을, 스크롤에 따라 pageSize씩 순차 공개한다.
// 화면이 커서 sentinel이 처음부터 보이면 IntersectionObserver가 즉시 다시 발화해
// 뷰포트를 채울 때까지 자연스럽게 이어서 공개된다.
export function useInfiniteReveal(pageSize: number) {
  const [visibleCount, setVisibleCount] = useState(pageSize)
  const sentinelRef = useRef<HTMLDivElement | null>(null)

  const reset = useCallback(() => setVisibleCount(pageSize), [pageSize])

  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setVisibleCount((v) => v + pageSize)
        }
      },
      { rootMargin: '300px' },
    )
    observer.observe(el)
    return () => observer.disconnect()
  })

  return { visibleCount, sentinelRef, reset }
}
