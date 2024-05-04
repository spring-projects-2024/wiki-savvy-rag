import json
from backend.benchmark.utils import format_question, craft_query, load_mmlu
import argparse
import yaml
import datasets
from backend.model.rag_handler import RagHandler


def evaluate(
        rag_handler: RagHandler, 
        dataset: datasets.Dataset, 
        k_shot: int = 0, 
        batch_size: int = 1
    ):
    metrics = {}
    examples = [dataset[i] for i in range(k_shot)]  # k-shot evaluation

    i = 0
    while i < len(dataset):
        batch = [dataset[i + j] for j in range(batch_size)]
        queries = [craft_query(question, chat=True, examples=examples) for question in batch]
        responses = rag_handler.inference([], queries)
        for question, response in zip(batch, responses):
            assert type(response) == str and len(response) == 1
            target = chr(65 + question["answer"])
            if response == target:
                metrics["correct"] += 1
            metrics["total"] += 1
        i += batch_size

    metrics["accuracy"] = metrics["correct"] / metrics["total"]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subset", type=str, default="stem")
    parser.add_argument("--output", type=str, default="mmlu.json")
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--config_path", type=str, default="configs/llm.yaml")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = load_mmlu(split=args.split, subset=args.subset)
    print(f"Loaded {len(dataset)} questions from the {args.subset} subset of the {args.split} split of the MMLU dataset.")

    device = config["device"]
    model_name = config["model_name"]
    model_kwargs = config.get("model_kwargs", None)
    tokenizer_kwargs = config.get("tokenizer_kwargs", None)
    faiss_kwargs = config.get("faiss_kwargs", None)

    rag_kwargs = config.get("rag_kwargs", {})
    rag_kwargs["max_new_tokens"] = 1  # this overrides the config file. We only need the first token.

    print("Creating RAGHandler...")
    rag_handler = RagHandler(
        model_name=model_name,
        device=device,
        rag_config=rag_kwargs,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        faiss_kwargs=faiss_kwargs,
    )

    print("Starting evaluation...")
    metrics = evaluate(rag_handler, dataset, k_shot=args.k_shot, batch_size=args.batch_size)
    print("Evaluation done.")

    with open(args.output, "w") as f:
        print(f"Saving metrics to {args.output}")
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
