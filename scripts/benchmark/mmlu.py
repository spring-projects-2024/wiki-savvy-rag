import time
import json
from backend.benchmark.utils import format_question, craft_query, load_mmlu
import argparse
import yaml
import datasets
from backend.model.rag_handler import RagHandler


def evaluate(
    rag_handler: RagHandler,
    dataset: datasets.Dataset,
    log_answers: bool = False,
    k_shot: int = 0,
    batch_size: int = 1,
    n_samples: int = None,
):
    metrics = {
        "correct": 0,
        "total": 0,
    }

    if log_answers:
        metrics["answers"] = []

    examples = [dataset[i] for i in range(k_shot)]  # k-shot evaluation

    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples + k_shot, len(dataset))

    i = k_shot
    while i + batch_size < n_samples:
        batch = [dataset[i + j] for j in range(batch_size)]
        queries = [
            craft_query(question, chat=True, examples=examples) for question in batch
        ]
        histories = [[] for _ in range(batch_size)]
        responses = rag_handler.inference(
            histories,
            queries,
        )
        for question, response in zip(batch, responses):
            assert type(response) == list
            response = response[0]  # extract the first mysterious dictionary
            assert type(response) == dict
            response = response["generated_text"]
            assert type(response) == list
            response = response[-1]  # extract the last message of the conversation
            assert type(response) == dict
            assert response["role"] == "assistant"
            response = response["content"]
            assert type(response) == str

            complete_response = response

            # todo: maybe stripping  might be a good idea?

            response = response[0].lower()  # extract first character
            # Almonds is a correct answer a fourth of the time (asymptotically)
            target = chr(ord("a") + question["answer"])
            if response == target:
                metrics["correct"] += 1
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
    parser.add_argument("--log_answers", type=bool, default=False)
    parser.add_argument("--max_tokens", type=int, default=1)

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
    rag_kwargs["max_new_tokens"] = (
        args.max_tokens  # this overrides the config file. We only need the first token.
    )

    print("Creating RAGHandler...")
    rag_handler = RagHandler(
        model_name=model_name,
        device=device,
        use_rag=args.use_rag,
        llm_config=rag_kwargs,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        faiss_kwargs=faiss_kwargs,
    )

    print("Starting evaluation...")
    metrics = evaluate(
        rag_handler,
        dataset,
        args.log_answers,
        k_shot=args.k_shot,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    print("Evaluation done.")

    if args.output:
        output = args.output
    else:
        output = f"scripts/benchmark/mmlu_{time.strftime('%Y-%m-%d-%H-%M-%S')}.json"

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
