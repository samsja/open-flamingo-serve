from jina import Executor, requests

from transformers import LlamaTokenizer
import torch

from transformers.tokenization_utils_base import BatchEncoding
import transformers


from typing import List, Union, TypeVar
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

from docarray import BaseDoc
from docarray import DocArray

from docarray.typing import ImageUrl, ImageBytes, TorchTensor

transformers.LLaMATokenizer = LlamaTokenizer  # ugly hack

ImgSource = Union[ImageUrl, ImageBytes]


class Prompt(BaseDoc):  # input schema
    images: List[ImgSource]
    prompt: str


num_images = TypeVar('num_images')


class PromptLoaded(BaseDoc):
    class Config:
        arbitrary_types_allowed = True

    images: TorchTensor[1, 'num_images', 1, 3, 224, 224]
    prompt: BatchEncoding


class Response(BaseDoc):
    generated: str


class FlamingoExec(Executor):
    def __init__(self, device='cuda', half=True, **kwargs):
        super().__init__(**kwargs)

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path='ViT-L-14',
            clip_vision_encoder_pretrained='openai',
            lang_encoder_path='decapoda-research/llama-7b-hf',
            tokenizer_path='decapoda-research/llama-7b-hf',
            cross_attn_every_n_layers=4,
        )
        self.tokenizer.padding_side = 'left'

        checkpoint_path = hf_hub_download(
            'openflamingo/OpenFlamingo-9B', 'checkpoint.pt'
        )
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)

        self.half = half
        if half:
            self.model = self.model.half()
        self.model = self.model.to(device)
        self.device = device

    @requests
    def generate(self, docs: DocArray[Prompt], **kwargs) -> DocArray[Response]:
        return DocArray[Response](
            [Response(generated=self.generate_one_doc(doc)) for doc in docs]
        )

    def generate_one_doc(self, prompt_input: Prompt, **kwargs) -> str:
        prompt = self.load_prompt_images(prompt_input)
        if self.half:
            prompt.images = prompt.images.half()

        prompt.images = prompt.images.to(self.device)
        prompt.prompt['input_ids'] = prompt.prompt['input_ids'].to(self.device)
        prompt.prompt['attention_mask'] = prompt.prompt['attention_mask'].to(
            self.device
        )

        generated_text = self.model.generate(
            vision_x=prompt.images,
            lang_x=prompt.prompt['input_ids'],
            attention_mask=prompt.prompt['attention_mask'],
            max_new_tokens=20,
            num_beams=3,
        )

        return self.tokenizer.decode(generated_text[0])

    def load_prompt_images(self, prompt: Prompt) -> PromptLoaded:
        images = [
            self.image_processor(img.load_pil()).unsqueeze(0) for img in prompt.images
        ]
        images = torch.cat(images, dim=0)
        images = images.unsqueeze(1).unsqueeze(0)

        lang_x = self.tokenizer([prompt.prompt], return_tensors='pt')

        return PromptLoaded(images=images, prompt=lang_x)


