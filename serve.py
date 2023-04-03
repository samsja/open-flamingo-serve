from jina import Deployment
from open_flamingo_serve import FlamingoExec

if __name__ == '__main__':
    with Deployment(uses=FlamingoExec, port=12347) as dep:
        dep.block()