from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from transformers.utils import ModelOutput
from torch import nn

DEFAULT_MODEL = "Qwen/Qwen1.5-0.5B-Chat"


class LLMHandler(nn.Module):
    """
    Handler for a language model.

    Attributes:
    - model: a pretrained language model.
    - tokenizer: a pretrained tokenizer.
    - pipe: a pipeline for text generation.
    - device: the device on which the model is loaded.
    - use_qlora: whether to use qLoRA quantization.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        pretrained_model_path: Optional[str] = None,
        llm_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        use_qlora: bool = False,
    ):
        """

        :param model_name:
        :param device:
        :param pretrained_model_path:
        :param quantization_config_path:
        :param llm_kwargs:
        :param tokenizer_kwargs:
        :param use_qlora:
        """
        super().__init__()
        self.device = device
        if llm_kwargs is None:
            llm_kwargs = {}
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        self.use_qlora = use_qlora
        if self.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        self.quantization_config = quantization_config

        if pretrained_model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_path,
                device_map=device,
                trust_remote_code=True,
                **llm_kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                quantization_config=quantization_config,
                **llm_kwargs,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, **tokenizer_kwargs
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def prepare_input(self, prompts: list[str], **kwargs) -> dict[str, torch.Tensor]:
        """
        Tokenize prompts and return a dictionary with keys "input_ids", "attention_mask".
        "input_ids" is a tensor of shape (batch_size, seq_len), where seq_len is the
        maximum sequence length in the batch (padding=True).
        "attention_mask" is a tensor of shape (batch_size, seq_len) with 1s in positions
        corresponding to tokens and 0s in positions corresponding to padding tokens.
        """
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            **kwargs,
        )

    def forward(self, input_ids: dict[str, torch.Tensor]) -> ModelOutput:
        """
        The output is an instance of a subclass of ModelOutput.
        Its attributes can be accessed with the dot notation.
        With phi-3-mini-128k-instruct, the output has the attributes
        'logits' and 'past_key_values' (for caching hidden states when
        decoding autoregressively).
        """
        return self.model(**input_ids)

    def get_logits(self, x: Union[dict[str, torch.Tensor], list[str]]) -> torch.Tensor:
        """
        Get the logits of the model output.
        Input can be a list of strings, which will be tokenized and padded,
        or a dictionary with keys "input_ids" and "attention_mask" ready for
        the model forward pass (like the output of prepare_input()).
        """
        if isinstance(x, dict):
            return self.forward(x).logits
        elif isinstance(x, list):
            return self.forward(self.prepare_input(x)).logits
        else:
            raise ValueError(f"Invalid input type: {type(x)}")

    @torch.inference_mode()
    def inference(self, messages, generation_args: Dict) -> List[str] | str:
        """
        Generate text based on a list of messages.
        Example of messages structure:
            messages = [
                {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
                {"role": "assistant",
                    "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
                {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
            ]
        """

        outputs = self.pipe(messages, return_full_text=False, **generation_args)
        if type(messages[0]) == list:
            return [output[0]["generated_text"] for output in outputs]
        else:
            return outputs[0]["generated_text"]

    def load_weights(self, path):
        print(f"Loading weights from {path}")
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def save_weights(self, path):
        self.model.save_pretrained(path)
