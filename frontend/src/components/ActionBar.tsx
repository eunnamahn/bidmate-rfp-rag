import { Agent } from '../types'

interface Props {
  agents: Agent[]
  onAction: (agent: Agent) => void
  disabled?: boolean
}

export default function ActionBar({ agents, onAction, disabled }: Props) {
  if (!agents.length) return null

  return (
    <div className="flex flex-wrap gap-1.5">
      {agents.map((a) => (
        <button
          key={a.id}
          onClick={() => onAction(a)}
          disabled={disabled || !a.implemented}
          title={a.description + (!a.implemented ? ' (준비 중)' : '')}
          className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium transition-colors border ${
            a.implemented
              ? 'bg-white hover:bg-primary-50 text-primary-700 border-primary-200 hover:border-primary-400 disabled:opacity-50'
              : 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed'
          }`}
        >
          {a.label}
          {!a.implemented && <span className="text-[10px] text-gray-400">(준비 중)</span>}
        </button>
      ))}
    </div>
  )
}
