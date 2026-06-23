import time
from typing import List, Union

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings


class RateLimitedVoyageEmbeddings(VoyageAIEmbeddings):
    """무료 한도(3 RPM / 10K TPM) 초과 시 잠시 대기 후 자동 재시도하는 Voyage 임베딩.

    RateLimitError 가 나면 25초 기다렸다가 다시 시도하므로,
    빌드·검색 어디서 호출되든 한도에 걸려도 알아서 통과한다.
    (근본 해결은 결제수단 등록 — 무료 토큰은 그대로 적용된다.)
    """

    def _with_retry(self, fn, arg):
        for attempt in range(6):
            try:
                return fn(arg)
            except Exception as e:
                if "RateLimit" not in type(e).__name__:
                    raise
                print(f"  [rate limit] 25초 대기 후 재시도 ({attempt + 1}/6)")
                time.sleep(25)
        raise RuntimeError("재시도 초과 - 잠시 후 다시 실행하거나 결제수단을 등록하세요.")

    def embed_documents(self, texts):
        return self._with_retry(super().embed_documents, texts)

    def embed_query(self, text):
        return self._with_retry(super().embed_query, text)


def get_llm(
    model: str = "claude-haiku-4-5",
    temperature: float = 0.2,
    streaming: bool = False,
    callbacks: List = [],
    base_url: str = "10.99.16.87:11434",
    num_predict: int = 500,
) -> Union[ChatOpenAI, ChatAnthropic]:
    """
    주어진 모델 이름에 따라 적절한 LLM(언어 모델) 객체를 반환하는 함수.

    지원되는 모델:
    - 'gpt'로 시작하는 모델: ChatOpenAI 객체 반환
    - 'claude'로 시작하는 모델: ChatAnthropic 객체 반환
    - 'llama', 'gemma', 'mistral'로 시작하는 모델: ChatOllama 객체 반환

    Args:
        model (str): 사용할 LLM 모델 이름. 기본값은 'gpt-4o'
        temperature (float): 생성된 텍스트의 창의성 수준을 조절하는 값. 기본값은 0.2
        streaming (bool): 스트리밍 모드를 사용할지 여부. 기본값은 False
        callbacks (List): 생성 중 실행할 콜백 함수 목록. 기본값은 빈 리스트
        num_predict (int): Ollama 모델의 출력 토큰 최대 길이. 기본값은 300

    Returns:
        Union[ChatOpenAI, ChatAnthropic, ChatOllama]: 주어진 모델 이름에 해당하는 LLM 객체.

    Raises:
        ValueError: 지원되지 않는 모델 이름이 입력된 경우 발생합니다.
    """
    if model.startswith("gpt"):
        llm = ChatOpenAI(model=model, temperature=temperature, streaming=streaming, callbacks=callbacks)
    elif model.startswith("claude"):
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
        )
    elif model.startswith(("llama", "gemma", "cow/gemma2_tools", "mistral", "EEVE", "qwen")):
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
            base_url=base_url,
            num_predict=num_predict,
        )
    else:
        raise ValueError(f"지원되지 않는 모델: {model}")

    return llm


def get_embedding(
    model: str = "voyage-4-lite",
    base_url: str = "10.99.16.87:11434",
) -> Union[VoyageAIEmbeddings, OpenAIEmbeddings]:
    """
    주어진 모델 이름에 따라 적절한 Embeddings 객체를 반환하는 함수.

    지원되는 모델:
    - 'voyage'로 시작하는 모델: RateLimitedVoyageEmbeddings 객체 반환(무료 한도 자동 대응)
    - 'text-embedding'으로 시작하는 모델: OpenAIEmbeddings 객체 반환
    - 'mxbai-embed-large', 'nomic-embed-text', 'bge-m3', 'jeffh/intfloat': OllamaEmbeddings 객체 반환

    Args:
        model (str): 사용할 Embeddings 모델의 이름. 기본값은 'voyage-4-lite'

    Returns:
        Union[VoyageAIEmbeddings, OpenAIEmbeddings, OllamaEmbeddings]: 주어진 모델에 해당하는 Embeddings 객체.

    Raises:
        ValueError: 지원되지 않는 모델 이름이 주어졌을 때 발생.
    """
    if model.startswith("voyage"):
        embeddings = RateLimitedVoyageEmbeddings(model=model)
    elif model.startswith("text-embedding"):
        embeddings = OpenAIEmbeddings(model=model)
    elif model.startswith(("mxbai-embed-large", "nomic-embed-text", "bge-m3", "jeffh/intfloat")):
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=model, base_url=base_url)
    else:
        raise ValueError(f"지원되지 않는 모델: {model}")

    return embeddings
