import datasets
from datasets import load_dataset
from typing import Union, Optional
from backend.benchmark.constants import stem_subcategories, yahoo_stem_categories

CHOICES = ["A", "B", "C", "D"]


def load_mmlu(
    split: str = "test",
    subset: Union[list, str, None] = "stem",
    num_samples: Optional[int] = None,
) -> datasets.Dataset:
    """
    Function to load the MMLU dataset.
    :param split: specifies the type of dataset split to be returned. Acceptable values are 'test', 'validation', 'dev', and 'auxiliary_train'.
    :param subset: determines the subset of the dataset to be returned. Acceptable values are None, 'stem', or a list of strings.
    If None, the function will return the entire dataset.
    If 'stem', the function will return only the subcategories related to Science, Technology, Engineering, and Mathematics (STEM).
    If a list of strings is provided, the function will return only the subcategories specified in the list.
    """
    if num_samples is not None:
        split = f"{split}[:{num_samples}]"
    dataset = load_dataset("cais/mmlu", "all", split=split)
    if subset is None:
        return dataset
    if isinstance(subset, str):
        if subset != "stem":
            raise ValueError("subset must be a list of strings, None, or 'stem'")
        subset = stem_subcategories
    dataset = dataset.filter(lambda x: x["subject"] in subset)
    return dataset


def load_mmlu_for_training(
    split: str = "test",
    subset: Union[list, str, None] = "stem",
    num_samples: Optional[int] = None,
) -> datasets.Dataset:
    dataset: datasets.Dataset = load_mmlu(
        split=split, subset=subset, num_samples=num_samples
    )

    def write_answer(example):
        example["answer_new"] = example["choices"][example["answer"]]
        return example

    dataset = dataset.map(write_answer)
    dataset = dataset.remove_columns("answer")
    dataset = dataset.rename_column("answer_new", "answer")
    dataset = dataset.rename_column("question", "query")
    return dataset


def load_yahoo_answers(
    subset: Union[list, str, None] = "stem", num_samples: Optional[int] = None
) -> datasets.Dataset:
    """
    Function to load the Yahoo Answers QA dataset.
    :param subset: determines the subset of the dataset to be returned. Acceptable values are None, 'stem', or a list of strings.
    If None, the function will return the entire dataset.
    If 'stem', the function will return only the subcategories related to Science, Technology, Engineering, and Mathematics (STEM).
    If a list of strings is provided, the function will return only the subcategories specified in the list.
    """
    split = "train"
    if num_samples is not None:
        split += f"[:{num_samples}]"
    dataset = load_dataset("yahoo_answers_qa", split=split)
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
    example = (
        "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
        + question["question"]
        + "\n"
    )
    for i, choice in enumerate(question["choices"]):
        example += f"{chr(65 + i)}.{choice}\n"

    return example


def format_example(line, include_answer=True):
    example = "Question: " + line["question"]

    for i, choice in enumerate(line["choices"]):
        example += f"\n{chr(65 + i)}. {choice}"

    if include_answer:
        example += f"\nAnswer:{chr(65 + line['answer'])}"
    else:
        example += "\nAnswer:"
    return example


def craft_query(
    question: dict, preface=True, examples: Optional[list[dict]] = None
) -> str:
    """
    :param question: a dictionary with the following keys
    - question: a string with a question
    - choices: a list of 4 strings with the answer choices
    - subject: a string with the subject of the question
    - answer: a number between 0 and 3 inclusive with
    the index of the correct answer
    :param preface: whether to explicitly ask to only provide a letter as answer
    :param examples: a list of dictionaries like question. If provided, add
    examples at the beginning of the prompt (for few-shot evaluation).
    :return: a nicely formatted prompt for an llm.
    """
    prompt = ""
    if preface:
        prompt += f"The following are multiple choice questions (with answers) about {question['subject']}.\n"
        prompt += "Only provide the letter of the answer. Do not write anything else!\n"

    prompt += "\n"
    if examples:
        for example in examples:
            prompt += _format_question(example, include_answer=True)
            prompt += "\n\n"
    prompt += _format_question(question, include_answer=False)

    prompt += "Provide only the letter of the answer:\n"
    return prompt


def format_example_0_shot(line):
    example = (
        "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the "
        "answer to this question. Clearly state the letter of the correct answer. DO NOT INCLUDE ANYTHING ELSE IN THE ANSWER.\n"
        "Acceptable answers are 'A.', 'B.', 'C.', or 'D.'\n\n"
        + line["question"]
        + "\n"
    )
    for i, choice in enumerate(line["choices"]):
        example += f"{chr(65 + i)}.{choice}\n"
    return example


def craft_few_shot_prompt(subject, examples: list[dict]):
    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )

    for example in examples:
        prompt += format_example(
            example,
            include_answer=True,
        )

    prompt += (
        "\nClearly state the letter of the correct answer. If the question contains whether "
        "two statements are true or false, provide only the letter of the answer containing the right combination."
        "Do not write which statement is false and which is true\n\n"
    )
    return prompt


if __name__ == "__main__":
    dataset = load_yahoo_answers("stem")
