from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# TODO: does pipe handle eval mode?


def get_default_gen_args():
    return {
        "max_new_tokens": 500,
        "return_full_text": False,
        # "temperature": 0.0,
        "do_sample": False,
    }


DEFAULT_MODEL = "microsoft/phi-3-mini-128k-instruct"


class LLMHandler:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        model_kwargs: Optional[dict] = None,  # torch_dtype
        tokenizer_kwargs: Optional[dict] = None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    @torch.inference_mode()
    def inference(self, messages, generation_args: Dict):
        """
        Example of messages structure:
            messages = [
                {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
                {"role": "assistant",
                    "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
                {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
            ]
        """
        return self.pipe(messages, **generation_args)  # TODO: get structure of output

    def load_weights(self, path):
        print(f"Loading weights from {path}")
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
