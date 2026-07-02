import { useEffect, useState } from 'react'

// 문서 파싱·청킹·임베딩은 파일 크기에 따라 수 초~수 분 걸릴 수 있다.
// 문구를 한동안 고정해두면 멈춘 것처럼 보이므로 단계를 진행 중인 것처럼 순차 전환한다.
// 실제 인덱싱 진행률과 동기화된 값은 아니고, 시간 경과에 따라 문구만 바뀐다.
const STAGES = [
  '대화를 위해 준비 중입니다. 잠시만 기다려주세요',
  '문서를 분석하고 있어요',
  '핵심 내용을 정리하고 있어요',
  '거의 다 됐어요',
  '조금만 더 기다려주세요',
]

const STAGE_INTERVAL_MS = 7000

// 마지막 두 단계 사이를 오가며 계속 움직이되, 앞 단계로 되돌아가는 것처럼 보이지 않게 한다.
function nextStage(current: number): number {
  if (current < STAGES.length - 1) return current + 1
  return STAGES.length - 2
}

export default function IndexingProgress({ className = '' }: { className?: string }) {
  const [stage, setStage] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setStage(nextStage), STAGE_INTERVAL_MS)
    return () => clearInterval(id)
  }, [])

  return (
    <span className={`inline-flex items-center justify-center gap-2 ${className}`}>
      <svg className="w-3.5 h-3.5 flex-shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      <span className="transition-opacity duration-500">{STAGES[stage]}</span>
    </span>
  )
}
