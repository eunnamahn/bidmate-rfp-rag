"""에이전트 액션 레지스트리.

챗 UI의 '에이전트 액션 버튼'(적격성 판정·리스크 탐지·인포그래픽·체크리스트·평가 구조 등)과
백엔드 라우팅을 한곳에서 선언한다. 버튼/엔드포인트/라우팅이 모두 이 레지스트리를 참조하므로,
새 에이전트는 여기 엔트리 하나만 추가/갱신하면 UI에 자동으로 노출된다.

설계 의도(팀 협업):
  팀원이 각 에이전트 본체를 별도 PR로 추가한다. 충돌을 줄이기 위해 본체(로직)는
  자기 모듈(src/agents/<name>_agent.py)과 graph 노드로 분리하고, 공유 파일인 이 레지스트리에서는
  엔트리 1개만 손댄다.

에이전트 추가 절차(README/ARCHITECTURE 참고):
  1) src/agents/<name>_agent.py 에 본체 함수 작성(검색→LLM 생성/추출; metadata_agent.py 형태).
  2) src/graph.py 에 노드 _<name> + 조건 엣지 1줄 추가.
  3) 이 파일의 해당 AgentSpec 을 implemented=True, intent="<node>" 로 갱신
     (또는 register(...)로 새 엔트리 추가).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentSpec:
    """챗 액션 버튼 1개 = 에이전트 1개의 메타데이터.

    Attributes:
        id:          고유 식별자(백엔드 /chat 의 agent 인자로 전달).
        label:       버튼에 표시할 짧은 이름.
        description: 버튼 설명(툴팁/접근성). 무엇을 하는 에이전트인지 한 줄.
        intent:      라우팅할 graph 노드 이름(qa/summary/metadata/... ). 미구현이면 None.
        prompt:      버튼 클릭 시 백엔드로 보낼 기본 지시문(사용자가 따로 입력하지 않을 때).
        needs_company_info: 회사 정보(업종/규모/매출 등) 입력이 필요한 에이전트인지.
                            (예: 적격성 판정 — 멀티턴으로 회사 정보를 받아야 함)
        implemented: 본체가 연결돼 동작하는지. False면 버튼은 보이되 '준비 중' 안내만 응답.
    """

    id: str
    label: str
    description: str
    intent: Optional[str] = None
    prompt: str = ""
    needs_company_info: bool = False
    implemented: bool = False
    # 버튼이 호출할 백엔드 엔드포인트. 기본 "/chat"은 graph intent 라우팅을 탄다.
    # 구조화 출력(인포그래픽 등)은 전용 엔드포인트를 지정한다.
    endpoint: str = "/chat"
    # UI 렌더 방식 힌트: "text"(마크다운) | "infographic"(HTML 카드).
    # mermaid 다이어그램은 답변의 ```mermaid 블록으로 자동 감지하므로 별도 값이 필요 없다.
    render: str = "text"

    def to_frontend(self) -> Dict[str, object]:
        """UI 렌더에 필요한 필드만 dict로 노출한다."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "prompt": self.prompt,
            "needs_company_info": self.needs_company_info,
            "implemented": self.implemented,
            "endpoint": self.endpoint,
            "render": self.render,
        }


# ══════════════════════════════════════════════════════
# 레지스트리 시드
#   - 구현됨: 기존 graph 능력(summary/metadata)을 그대로 액션 버튼으로 노출.
#   - 미구현: 이미지의 5개 에이전트. 본체는 팀원 PR이 채운다(implemented=False).
# ══════════════════════════════════════════════════════

AGENTS: List[AgentSpec] = [
    AgentSpec(
        id="summary",
        label="요약",
        description="대상 기관·예산·마감일·핵심 요구조건·제출 방식을 양식으로 정리합니다.",
        intent="summary",
        prompt="이 RFP의 핵심을 대상 기관·예산·마감일·핵심 요구조건·제출 방식으로 정리해줘.",
        implemented=True,
    ),
    # 메타데이터 추출은 채팅 액션이 아니라 '문서 추가 시 data_list.csv 자동 기록'으로 쓴다
    # (backend /upload에서 metadata_agent.append_metadata_row 호출). 채팅 버튼에서는 제외.
    AgentSpec(
        id="eligibility",
        label="적격성 판정",
        description="회사 정보(업종·규모·창업연수·매출)를 받아 공고 참가자격과 대조해 충족 여부를 판정합니다.",
        intent="eligibility",
        prompt="이 공고의 참가자격을 기준으로 우리 회사의 적격성을 판정해줘.",
        needs_company_info=True,
        implemented=True,
    ),
    AgentSpec(
        id="risk",
        label="리스크 탐지",
        description="독소조항·과도한 자격요건·촉박한 일정·불리한 정산조건 등 리스크 신호를 추출합니다.",
        intent="risk",
        prompt="이 공고에서 주의해야 할 리스크 요소를 찾아줘.",
        implemented=True,
    ),
    AgentSpec(
        id="evaluation",
        label="평가 구조",
        description="평가배점 구조(기술/가격 등)를 Mermaid 다이어그램으로 시각화합니다.",
        intent="evaluation",
        prompt="이 공고의 평가배점 구조를 다이어그램으로 정리해줘.",
        implemented=True,
    ),
    AgentSpec(
        id="checklist",
        label="체크리스트",
        description="제출서류·자격요건·마감일정을 점검 가능한 준비 체크리스트로 정리합니다.",
        intent="checklist",
        prompt="이 공고 입찰 준비를 위한 체크리스트를 만들어줘.",
        implemented=True,
    ),
    # 인포그래픽은 HTML 카드(구조화 출력)라 graph 텍스트 응답이 아니라 전용 엔드포인트를 쓴다.
    AgentSpec(
        id="infographic",
        label="인포그래픽",
        description="평가배점·예산·일정 등 핵심 수치를 한 장짜리 HTML 카드로 시각화합니다.",
        prompt="이 공고의 핵심 수치를 인포그래픽으로 정리해줘.",
        implemented=True,
        endpoint="/infographic",
        render="infographic",
    ),
]

# id -> AgentSpec 빠른 조회용 인덱스(register로 갱신).
_INDEX: Dict[str, AgentSpec] = {a.id: a for a in AGENTS}


def register(spec: AgentSpec) -> None:
    """새 에이전트를 등록하거나 같은 id의 기존 엔트리를 교체한다.

    팀원 모듈에서 import 시 호출하는 확장 지점. AGENTS 본문을 직접 편집하지 않고
    register(...)로 추가하면 공유 파일 충돌을 더 줄일 수 있다.
    """
    existing = _INDEX.get(spec.id)
    if existing is not None:
        AGENTS[AGENTS.index(existing)] = spec
    else:
        AGENTS.append(spec)
    _INDEX[spec.id] = spec


def get(agent_id: str) -> Optional[AgentSpec]:
    """id로 AgentSpec을 조회한다(없으면 None)."""
    return _INDEX.get(agent_id)


def as_frontend() -> List[Dict[str, object]]:
    """UI가 액션 버튼을 렌더할 수 있도록 스펙 목록을 dict로 반환한다."""
    return [a.to_frontend() for a in AGENTS]
