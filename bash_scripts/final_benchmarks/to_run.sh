# without rag
python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 0
python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 5

# with rag naive
python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 0 --use_rag 1 --inference_type "naive" --n_docs_retrieved 1
python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 0 --use_rag 1 --inference_type "naive" --n_docs_retrieved 4

python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 5 --use_rag 1 --inference_type "naive" --n_docs_retrieved 1
python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 5 --use_rag 1 --inference_type "naive" --n_docs_retrieved 4

# with replug
python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 0 --use_rag 1 --inference_type "replug" --n_docs_retrieved 1
python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 0 --use_rag 1 --inference_type "replug" --n_docs_retrieved 4

python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 5 --use_rag 1 --inference_type "replug" --n_docs_retrieved 1
python3 scripts/benchmark/mmlu.py --config_path "configs/llm_vm.yaml" --log_answers True --k_shot 5 --use_rag 1 --inference_type "replug" --n_docs_retrieved 4

