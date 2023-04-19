from typing import List, Union, TypeVar

from docarray import BaseDoc
from docarray import DocList, DocVec
from docarray.typing import ImageUrl, ImageBytes, TorchTensor

from jina import Executor, requests

from transformers import LlamaTokenizer
import torch
import transformers
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download


transformers.LLaMATokenizer = LlamaTokenizer  # ugly hack

ImgSource = Union[ImageUrl, ImageBytes]


class Prompt(BaseDoc):  # input schema
    images: List[ImgSource]
    prompt: str


num_images = TypeVar('num_images')


class Tokens(BaseDoc):
    input_ids: TorchTensor = None
    attention_mask: TorchTensor = None


class PromptLoaded(BaseDoc):
    images: TorchTensor['num_images', 1, 3, 224, 224]
    prompt: Tokens


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

    def load_prompt_images(self, prompt: Prompt) -> PromptLoaded:
        images = [
            self.image_processor(img.load_pil()).unsqueeze(0) for img in prompt.images
        ]
        images = torch.cat(images, dim=0)
        images = images.unsqueeze(1)

        lang_x = self.tokenizer([prompt.prompt])

        return PromptLoaded(images=images, prompt=Tokens(**lang_x))

    def get_doc_vec(self, docs: DocList[Prompt]) -> DocVec[PromptLoaded]:
        da = DocVec[PromptLoaded]([self.load_prompt_images(doc) for doc in docs])
        da.prompt.to(self.device)
        da.images = da.images.half()
        da.to(self.device)

        da.prompt.input_ids = da.prompt.input_ids.squeeze(1)
        da.prompt.attention_mask = da.prompt.attention_mask.squeeze(1)
        return da

    @requests
    def generate(self, docs: DocList[Prompt], **kwargs) -> DocList[Response]:

        da = self.get_doc_vec(docs)

        generated_text = self.model.generate(
            vision_x=da.images,
            lang_x=da.prompt.input_ids,
            attention_mask=da.prompt.attention_mask,
            max_new_tokens=20,
            num_beams=3,
        )

        return DocList[Response](
            [Response(generated=self.tokenizer.decode(txt)) for txt in generated_text]
        )
