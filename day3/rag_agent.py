# RAG 에이전트 - 검색 기반 질문 응답 시스템
import getpass
import os
from pprint import pprint
from typing import Dict, List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
from typing_extensions import TypedDict

# API 키 설정
tavily_api_key = getpass.getpass()
tavily = TavilyClient(api_key=tavily_api_key)
openai_api_key = getpass.getpass()
os.environ["OPENAI_API_KEY"] = openai_api_key

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 벡터 데이터베이스 구축
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 웹 문서 로드 및 분할
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# 벡터 저장소 생성
vectorstore = Chroma.from_documents(
    documents=doc_splits, collection_name="rag-chroma", embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

# 프롬프트 정의 및 체인 구성
# 1. 질문 라우터 - 벡터 검색 또는 웹 검색 중 선택
router_system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
You do not need to be stringent with the keywords in the question related to these topics.
Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

question_router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_system_prompt),
        ("human", "question: {question}"),
    ]
)
question_router = question_router_prompt | llm | JsonOutputParser()

# 2. 검색 결과 평가기 - 검색된 문서가 질문과 관련있는지 평가
retrieval_grader_system_prompt = """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_grader_system_prompt),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)
retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

# 3. 응답 생성기 - 검색된 문서를 기반으로 답변 생성
generate_system_prompt = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise. After your answer, include the sources used in the format:

    Sources:
    - [Title or description] (URL)"""

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generate_system_prompt),
        ("human", "question: {question}\n\n context: {context}\n\n sources: {sources} "),
    ]
)
rag_chain = generate_prompt | llm | StrOutputParser()

# 4. 환각 평가기 - 생성된 답변이 문서 내용에 기반하는지 평가
hallucination_grader_system_prompt = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_system_prompt),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)
hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

# 5. 답변 평가기 - 생성된 답변이 질문에 유용한지 평가
answer_grader_system_prompt = """You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system_prompt),
        ("human", "question: {question}\n\n answer: {generation} "),
    ]
)
answer_grader = answer_grader_prompt | llm | JsonOutputParser()


# 그래프 상태 정의
class GraphState(TypedDict):
    """그래프의 상태를 표현하는 클래스"""

    question: str  # 사용자 질문
    generation: str  # 생성된 답변
    web_search: str  # 웹 검색 필요 여부
    documents: List[Document]  # 검색된 문서들
    sources: List[Dict[str, str]]  # 출처 정보 (URL, 제목)
    retry_count: int  # 재시도 횟수


# 노드 함수 정의
def retrieve(state):
    """벡터 저장소에서 문서 검색"""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(question)
    print(documents)
    # 벡터 저장소의 문서에는 별도 출처 정보가 없으므로 기본값 사용
    sources = [{"title": "LangChain Vector DB", "url": url} for url in urls]
    return {"documents": documents, "question": question, "sources": sources, "retry_count": 0}


def generate(state):
    """검색된 문서를 이용해 답변 생성"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    sources = state["sources"]
    retry_count = state.get("retry_count", 0)

    generation = rag_chain.invoke({"context": documents, "question": question, "sources": sources})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "sources": sources,
        "retry_count": retry_count,
    }


def grade_documents(state):
    """검색된 문서가 질문과 관련 있는지 평가"""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    sources = state.get("sources", [])

    filtered_docs = []
    filtered_sources = []

    for i, d in enumerate(documents):
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score["score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            if i < len(sources):
                filtered_sources.append(sources[i])

    # 관련 문서가 없으면 실패
    if not filtered_docs:
        print("---FAILED: NO RELEVANT DOCUMENTS FOUND---")
        return {"question": question, "generation": "Failed: No relevant documents found", "error": "not_relevant"}

    # 관련 문서가 있으면 생성 단계로
    return {"documents": filtered_docs, "question": question, "sources": filtered_sources, "retry_count": 0}


def web_search(state):
    """웹 검색으로 추가 정보 획득"""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    sources = state.get("sources", [])

    # Tavily 검색 실행
    search_results = tavily.search(query=question)["results"]

    # 웹 검색 결과를 문서로 변환
    web_docs = []
    web_sources = []

    for result in search_results:
        web_docs.append(Document(page_content=result["content"]))
        web_sources.append({"title": result["title"], "url": result["url"]})

    # 웹 검색 결과 평가
    filtered_web_docs = []
    filtered_web_sources = []

    for i, doc in enumerate(web_docs):
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score["score"]
        if grade.lower() == "yes":
            print("---GRADE: WEB DOCUMENT RELEVANT---")
            filtered_web_docs.append(doc)
            filtered_web_sources.append(web_sources[i])

    # 관련 문서가 없으면 실패 상태 반환
    if not filtered_web_docs and not documents:
        print("---FAILED: NO RELEVANT DOCUMENTS FOUND---")
        return {"question": question, "generation": "Failed: No relevant documents found", "error": "not_relevant"}

    # 기존 문서 및 출처와 통합
    all_docs = documents + filtered_web_docs
    all_sources = sources + filtered_web_sources

    return {"documents": all_docs, "sources": all_sources, "question": question, "retry_count": 0}


def route_question(state):
    """질문을 웹 검색 또는 벡터 검색으로 라우팅"""
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """문서 평가 결과에 따라 다음 단계 결정"""
    print("---ASSESS GRADED DOCUMENTS---")

    if "error" in state:
        print("---DECISION: NO RELEVANT DOCUMENTS, FAILED---")
        return "failed"

    print("---DECISION: RELEVANT DOCUMENTS FOUND, GENERATE---")
    return "generate"


def grade_generation_v_documents_and_question(state):
    """생성된 답변 평가"""
    print("---CHECK HALLUCINATIONS---")

    if "error" in state:
        return "failed"

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            if retry_count < 1:
                return "needs_improvement"
            else:
                return "failed"
    else:
        if retry_count < 1:
            return "needs_improvement"
        else:
            return "failed"


def increment_retry_counter(state):
    """재시도 카운터 증가"""
    new_state = state.copy()
    new_state["retry_count"] = state.get("retry_count", 0) + 1
    print(f"---RETRY ATTEMPT {new_state['retry_count']}---")
    return new_state


def mark_as_failed(state):
    """실패 상태 표시"""
    print("---TASK FAILED---")
    state["generation"] = "Failed: " + (state.get("error", "hallucination or not useful"))
    return state


# 워크플로우 그래프 구성
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("websearch", web_search)  # 웹 검색
workflow.add_node("retrieve", retrieve)  # 벡터 검색
workflow.add_node("grade", grade_documents)  # 문서 평가 (통합)
workflow.add_node("generate", generate)  # 답변 생성
workflow.add_node("retry", increment_retry_counter)  # 재시도
workflow.add_node("failed", mark_as_failed)  # 실패 처리

# 라우팅 분기
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

# 검색 결과 평가
workflow.add_edge("websearch", "grade")
workflow.add_edge("retrieve", "grade")

# 평가 결과에 따른 분기
workflow.add_conditional_edges("grade", decide_to_generate, {"generate": "generate", "failed": "failed"})

# 생성 및 재시도 로직
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {"useful": END, "needs_improvement": "retry", "failed": "failed"},
)

workflow.add_edge("retry", "generate")
workflow.add_edge("failed", END)

# 워크플로우 컴파일
app = workflow.compile()

# 테스트 실행
inputs = {"question": "What is prompt?"}
final_value = None

for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
        final_value = value

if final_value:
    pprint(final_value["generation"])

"""
# 그래프 시각화 생성
dot = graphviz.Digraph()
dot.attr(rankdir="LR")

# 노드 추가
dot.node("START", "Start", shape="circle")
dot.node("route", "Route Question", shape="diamond")
dot.node("websearch", "Web Search", shape="box")
dot.node("retrieve", "Vector Retrieve", shape="box")
dot.node("grade", "Grade Documents", shape="box")
dot.node("generate", "Generate Answer", shape="box")
dot.node("retry", "Retry", shape="box")
dot.node("failed", "Failed", shape="box", style="filled", fillcolor="lightpink")
dot.node("END", "End", shape="circle")

# 시작 및 라우팅
dot.edge("START", "route")
dot.edge("route", "websearch", "web_search")
dot.edge("route", "retrieve", "vectorstore")

# 검색 및 평가
dot.edge("websearch", "grade")
dot.edge("retrieve", "grade")

# 평가 결과에 따른 분기
dot.edge("grade", "generate", "relevant docs found")
dot.edge("grade", "failed", "no relevant docs")

# 생성 및 재시도
dot.edge("generate", "retry", "needs improvement")
dot.edge("generate", "END", "useful")
dot.edge("generate", "failed", "max retries exceeded")
dot.edge("retry", "generate")

# 실패 처리
dot.edge("failed", "END")

# 그래프 저장
dot.render("rag_workflow", format="png", cleanup=True)
print("Workflow diagram saved as 'rag_workflow.png'")
"""
