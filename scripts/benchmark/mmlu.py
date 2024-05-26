import time
import json
from backend.benchmark.utils import craft_query, load_mmlu, craft_few_shot_prompt, format_example, format_example_0_shot
import argparse
import yaml
import datasets
import os
from backend.model.rag_handler import RagHandler
import re
from thefuzz import process

choices = ["A", "B", "C", "D"]
def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def process_before_extraction(gen, choice_dict):
    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen

def extract_answer(response, question):
    gen = process_before_extraction(
        response, {choice: answ for choice, answ in zip(choices, question["choices"])}
    )
    pred = extract_choice(gen, question["choices"])
    return pred



def evaluate(
    rag_handler: RagHandler,
    dataset: datasets.Dataset,
    log_answers: bool = False,
    n_docs_retrieved: int = 10,
    k_shot: int = 0,
    batch_size: int = 1,
    n_samples: int = None,
    inference_type: str = "replug",
):
    metrics = {
        "correct": 0,
        "total": 0,
    }

    if log_answers:
        metrics["answers"] = []

    examples = [dataset[i] for i in range(k_shot)]  # k-shot evaluation

    if k_shot > 0:
        few_shot_prompt: str = craft_few_shot_prompt(
            subject=examples[0]["subject"],
            examples=examples,
        )
    else:
        few_shot_prompt: str = ""

    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples + k_shot, len(dataset))

    i = k_shot
    while i + batch_size < n_samples:
        batch = [dataset[i + j] for j in range(batch_size)]
        if k_shot > 0:
            queries = [
                few_shot_prompt + format_example(question, include_answer=False)
                for question in batch
            ]
        else:
            queries = [format_example_0_shot(question) for question in batch]


        responses = [
            rag_handler.inference(
                query=query,
                n_docs_retrieved=n_docs_retrieved,
                return_generator=False,
                return_prompt=False,
                inference_type=inference_type,
            )[0]
            for query in queries
        ]
        for question, response in zip(batch, responses):
            response = response.strip()
            complete_response = response

            # Almonds is a correct answer a fourth of the time (asymptotically)
            pred = extract_answer(response, question)

            pred_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
            pred = pred_to_num[pred]
            if "answer" in question:
                correct = 1 if pred == question["answer"] else 0
                if correct:
                    metrics["correct"] += 1

            # target = chr(ord("a") + question["answer"])
            # if response == target:
            #     metrics["correct"] += 1
            metrics["total"] += 1

            if log_answers:
                metrics["answers"].append(complete_response)

        i += batch_size

    metrics["accuracy"] = metrics["correct"] / metrics["total"]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subset", type=str, default="stem")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--config_path", type=str, default="configs/llm.yaml")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--use_rag", type=bool, default=False)
    parser.add_argument("--n_docs_retrieved", type=int, default=10)
    parser.add_argument("--log_answers", type=bool, default=False)
    parser.add_argument("--inference_type", type=str, default="replug")

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = load_mmlu(split=args.split, subset=args.subset)
    print(
        f"Loaded {len(dataset)} questions from the {args.subset} subset of the {args.split} split of the MMLU dataset."
    )

    device = config["device"]
    model_name = config["model_name"]
    model_kwargs = config.get("model_kwargs", None)
    tokenizer_kwargs = config.get("tokenizer_kwargs", None)
    faiss_kwargs = config.get("faiss_kwargs", None)
    rag_kwargs = config.get("rag_kwargs", {})

    print("Creating RAGHandler...")
    rag_handler = RagHandler(
        model_name=model_name,
        device=device,
        use_rag=args.use_rag,
        llm_generation_config=rag_kwargs,
        llm_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        faiss_kwargs=faiss_kwargs,
    )

    print("Starting evaluation...")
    metrics = evaluate(
        rag_handler,
        dataset,
        args.log_answers,
        n_docs_retrieved=args.n_docs_retrieved,
        k_shot=args.k_shot,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        inference_type=args.inference_type,
    )
    print("Evaluation done.")

    if args.output:
        output = args.output
    else:
        os.makedirs("scripts/benchmark/out", exist_ok=True)
        output = f"scripts/benchmark/out/mmlu_{time.strftime('%Y-%m-%d-%H-%M-%S')}.json"

    result = {
        "metrics": metrics,
        "config": config,
        "args": vars(args),
    }

    with open(output, "w") as f:
        print(f"Saving metrics to {output}")
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
