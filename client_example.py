from typing import List, Union
from docarray import BaseDoc, DocArray
from docarray.typing import ImageUrl, ImageBytes
from jina import Client

ImgSource = Union[ImageUrl, ImageBytes]


class Prompt(BaseDoc):
    images: List[ImgSource]
    prompt: str


class Response(BaseDoc):
    generated: str


images = ['https://cdn.pariscityvision.com/library/image/5449.jpg']
prompt = Prompt(
    images=images,
    prompt='<image> What is the name of this painting ?.<|endofchunk|><image>This is ',
)

docs = DocArray[Prompt]([prompt])

client = Client(port=12347)
resp = client.post(
    on='/', inputs=DocArray[Prompt]([prompt]), return_type=DocArray[Response]
)

print(resp[0].generated)
