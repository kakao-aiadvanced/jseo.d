import streamlit as st
from rag_agent import app

st.set_page_config(page_title="RAG 챗봇", page_icon="🤖", layout="wide")

# 제목과 설명
st.title("🤖 RAG 챗봇")
st.markdown(
    """
이 챗봇은 다음 주제들에 대해 답변할 수 있습니다:
- LLM 에이전트
- 프롬프트 엔지니어링
- LLM 적대적 공격
"""
)

# 사이드바에 정보 표시
with st.sidebar:
    st.header("정보")
    st.markdown(
        """
    이 챗봇은 다음 문서들을 기반으로 답변합니다:
    - [LLM Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
    - [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
    - [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/)
    """
    )

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 로딩 표시
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            # RAG 에이전트로 답변 생성
            response = None
            for output in app.stream({"question": prompt}):
                for key, value in output.items():
                    response = value

            if response and "generation" in response:
                answer = response["generation"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("답변을 생성하는 데 문제가 발생했습니다.")

# 채팅 초기화 버튼
if st.sidebar.button("채팅 초기화"):
    st.session_state.messages = []
    st.rerun()
