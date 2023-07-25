from uuid import uuid4


def uid():
    return str(uuid4())[:8]
