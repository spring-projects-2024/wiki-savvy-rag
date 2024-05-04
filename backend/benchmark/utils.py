import datasets
from datasets import load_dataset
from typing import Union, Optional
from constants import stem_subcategories


def load_mmlu(split: str = "test", subset: Union[list, str, None] = "stem") -> datasets.Dataset:
    """
    :param split: one of test, validation, dev, auxiliary_train
    :param subset: one of None, 'stem', or a list of strings. If None,
    return the entire dataset. If 'stem', return only the STEM subcategories.
    If a list of strings, return only the subcategories in the list.
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


def format_question(question: dict, include_answer: bool = False) -> str:
    prompt = f"Question: {question['question']}\n"
    for i, choice in enumerate(question["choices"]):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "Answer:"
    if include_answer:
        prompt += f" {chr(65 + question['answer'])}"
    return prompt


def craft_query(
    question: dict, chat=False, examples: Optional[list[dict]] = None
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
        prompt += "Only provide the letter of the answer.\n"
    prompt += "\n"
    if examples:
        for example in examples:
            prompt += format_question(example, include_answer=True)
            prompt += "\n\n"
    prompt += format_question(question, include_answer=False)
    return prompt


if __name__ == "__main__":
    dataset = load_mmlu(split="test", subset="stem")
    question = dataset[5]
    examples = [dataset[i] for i in range(5)]
    prompt = craft_query(question, chat=True, examples=examples)
    print(prompt)
