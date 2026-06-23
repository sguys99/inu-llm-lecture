from typing import Any, Dict

import yaml
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate


def build_qa_prompt(
    system_message: str,
    add_context: bool = True,
    add_history: bool = True,
    examples: Dict = None,
) -> PromptTemplate:
    """
    시스템 메시지와 선택적인 컨텍스트 및 대화 기록을 사용하여 질문-답변(QA) 프롬프트 템플릿을 생성하는 함수.

    Args:
        system_message (str): 프롬프트의 시작 부분에 추가될 시스템 메시지 또는 지침.
        add_context (bool): True인 경우 프롬프트에 컨텍스트 정보를 포함합니다. 기본값은 True.
        add_history (bool): True인 경우 프롬프트에 대화 기록을 포함합니다. 기본값은 True.
        examples (Dict): FewShot프롬프트에 사용되는 examples. 기본값은 None.

    Returns:
        PromptTemplate: Langchain에서 사용할 수 있는 PromptTemplate 객체.
    """
    qa_template = ""

    if add_context:
        qa_template += "Context: {context}\n"

    if add_history:
        qa_template += "History: {chat_history}\n"

    qa_template += "Question: {question}"

    full_template = system_message + "\n\n" + qa_template

    if examples:
        example_template = PromptTemplate(
            template="Question: {question}\nAnswer: {answer}",
            input_variables=["question", "answer"],
        )
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix=system_message + "\n\n",
            suffix=qa_template,
            input_variables=["context", "chat_instory", "question"],
            example_separator="\n\n",
        )

    else:
        prompt = PromptTemplate(
            template=full_template,
            input_variables=["context", "chat_instory", "question"],
        )

    return prompt


def save_prompt(prompt: PromptTemplate, save_path: str) -> None:
    """
    주어진 PromptTemplate 객체를 YAML 파일로 저장하는 함수.

    Args:
        prompt (PromptTemplate): Langchain의 PromptTemplate 객체로, 저장할 프롬프트.
        save_path (str): YAML 파일을 저장할 경로를 나타내는 문자열.

    Returns:
        None: 이 함수는 파일을 저장하며, 반환값은 없음.
    """

    prompt_data = {
        "_type": "prompt",
        "input_variables": prompt.input_variables,
        "metadata": prompt.metadata,
        "name": prompt.name,
        "optional_variables": prompt.optional_variables,
        "output_parser": prompt.output_parser,
        "partial_variables": prompt.partial_variables,
        "tags": prompt.tags,
        "template": prompt.template,
        "template_format": prompt.template_format,
        "validate_template": prompt.validate_template,
    }

    with open(save_path, "w", encoding="utf-8") as file:
        yaml.dump(prompt_data, file, allow_unicode=True)


def save_fewshot_prompt(prompt: Any, save_path: str) -> None:
    """
    주어진 FewShotPromptTemplate 객체를 YAML 형식으로 저장하는 함수.

    Args:
        prompt (Any): FewShotPromptTemplate 객체로, 예제 프롬프트와 설정 정보를 포함.
        save_path (str): 저장할 파일 경로. YAML 형식으로 저장.

    Returns:
        None: 이 함수는 반환 값이 없습니다.
    """
    if hasattr(prompt.example_prompt, "__dict__"):
        example_prompt_dict = prompt.example_prompt.__dict__
    else:
        example_prompt_dict = prompt.example_prompt

    prompt_data = {
        "_type": "few_shot",
        "example_prompt": {
            "_type": "prompt",
            "input_variables": example_prompt_dict.get("input_variables", []),
            "metadata": example_prompt_dict.get("metadata", None),
            "name": example_prompt_dict.get("name", None),
            "optional_variables": example_prompt_dict.get("optional_variables", []),
            "output_parser": example_prompt_dict.get("output_parser", None),
            "partial_variables": example_prompt_dict.get("partial_variables", {}),
            "tags": example_prompt_dict.get("tags", None),
            "template": example_prompt_dict.get("template", ""),
            "template_format": example_prompt_dict.get("template_format", "f-string"),
            "validate_template": example_prompt_dict.get("validate_template", False),
        },
        "example_selector": prompt.example_selector,
        "example_separator": prompt.example_separator,
        "examples": prompt.examples,
        "input_variables": prompt.input_variables,
        "metadata": prompt.metadata,
        "name": prompt.name,
        "optional_variables": prompt.optional_variables,
        "output_parser": prompt.output_parser,
        "partial_variables": prompt.partial_variables,
        "prefix": prompt.prefix,
        "suffix": prompt.suffix,
        "tags": prompt.tags,
        "template_format": prompt.template_format,
        "validate_template": prompt.validate_template,
    }

    with open(save_path, "w", encoding="utf-8") as file:
        yaml.dump(prompt_data, file, allow_unicode=True)
