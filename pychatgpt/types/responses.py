class ChatCompletion:
    def __init__(self, **kwargs) -> None:
        for key, item in kwargs.items():
            setattr(self, key, item)
