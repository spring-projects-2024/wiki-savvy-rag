from typing import Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_default_gen_args():
    return {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }


MODEL_NAME = "microsoft/phi-3-mini-128k-instruct"


class LLMHandler:
    def __init__(self, device, weight_path=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        # todo: add weight loading from disk and test it

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def inference(self, messages, generation_args: Dict | None = None):
        """
        Example of messages structure:
            messages = [
                {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
                {"role": "assistant",
                    "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
                {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
            ]
        """
        if generation_args is None:
            generation_args = get_default_gen_args()
        return self.pipe(messages, **generation_args)  # todo: get structure of output
