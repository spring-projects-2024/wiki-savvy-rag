import datasets
from datasets import load_dataset
from typing import Union, Optional
from backend.benchmark.constants import stem_subcategories, yahoo_stem_categories


def load_mmlu(
    split: str = "test", subset: Union[list, str, None] = "stem"
) -> datasets.Dataset:
    """
    Function to load the MMLU dataset.
    :param split: specifies the type of dataset split to be returned. Acceptable values are 'test', 'validation', 'dev', and 'auxiliary_train'.
    :param subset: determines the subset of the dataset to be returned. Acceptable values are None, 'stem', or a list of strings.
    If None, the function will return the entire dataset.
    If 'stem', the function will return only the subcategories related to Science, Technology, Engineering, and Mathematics (STEM).
    If a list of strings is provided, the function will return only the subcategories specified in the list.
    """
    dataset = load_dataset("cais/mmlu", "all")
    dataset = dataset[split]
    if subset is None:
        return dataset
    if isinstance(subset, str):
        if subset != "stem":
            raise ValueError("subset must be a list of strings, None, or 'stem'")
        subset = stem_subcategories
    dataset = dataset.filter(lambda x: x["subject"] in subset)
    return dataset


def load_mmlu_for_training(
    split: str = "test", subset: Union[list, str, None] = "stem"
) -> datasets.Dataset:
    dataset: datasets.Dataset = load_mmlu(split=split, subset=subset)

    def write_answer(example):
        example["answer_new"] = example["choices"][example["answer"]]
        return example

    dataset = dataset.map(write_answer)
    dataset = dataset.remove_columns("answer")
    dataset = dataset.rename_column("answer_new", "answer")
    dataset = dataset.rename_column("question", "query")
    return dataset


def load_yahoo_answers(subset: Union[list, str, None] = "stem") -> datasets.Dataset:
    """
    Function to load the Yahoo Answers QA dataset.
    :param subset: determines the subset of the dataset to be returned. Acceptable values are None, 'stem', or a list of strings.
    If None, the function will return the entire dataset.
    If 'stem', the function will return only the subcategories related to Science, Technology, Engineering, and Mathematics (STEM).
    If a list of strings is provided, the function will return only the subcategories specified in the list.
    """
    dataset = load_dataset("yahoo_answers_qa")
    dataset = dataset["train"]
    if subset is None:
        return dataset
    if isinstance(subset, str):
        if subset != "stem":
            raise ValueError("subset must be a list of strings, None, or 'stem'")
        subset = yahoo_stem_categories
    dataset = dataset.filter(lambda x: x["main_category"] in subset)
    dataset = dataset.rename_column("question", "query")
    return dataset


def _format_question(question: dict, include_answer: bool = False) -> str:
    prompt = f"Question: {question['question']}\n"
    for i, choice in enumerate(question["choices"]):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "Answer:"
    if include_answer:
        prompt += f" {chr(65 + question['answer'])}"
    return prompt


def craft_query(
    question: dict, chat=True, examples: Optional[list[dict]] = None
) -> str:
    """
    :param question: a dictionary with the following keys
    - question: a string with a question
    - choices: a list of 4 strings with the answer choices
    - subject: a string with the subject of the question
    - answer: a number between 0 and 3 inclusive with
    the index of the correct answer
    :param chat: whether to explicitly ask to only provide a letter as answer
    :param examples: a list of dictionaries like question. If provided, add
    examples at the beginning of the prompt (for few-shot evaluation).
    :return: a nicely formatted prompt for an llm.
    """
    prompt = f"The following are multiple choice questions (with answers) about {question['subject']}.\n"
    if chat:
        prompt += "Only provide the letter of the answer. Do not write anything else!\n"
    prompt += "\n"
    if examples:
        for example in examples:
            prompt += _format_question(example, include_answer=True)
            prompt += "\n\n"
    prompt += _format_question(question, include_answer=False)

    prompt += "\n\nAnswer (only the letter):\n"
    return prompt


if __name__ == "__main__":
    dataset = load_yahoo_answers("stem")

    question = dataset[0]
    print(question)


if __name__ == "__main__":
    dataset = load_mmlu_for_training("test", None)
