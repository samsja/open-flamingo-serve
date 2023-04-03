from jina import DocumentArray, Executor, requests


class MyExecutor22(Executor):
    """OpenFlamingo serving. MLLM, text and image as input and generate text out of it"""
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass