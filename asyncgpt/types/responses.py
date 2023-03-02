class ChatCompletion:
    def __init__(self, **kwargs) -> None:
        for key, item in kwargs.items():
            setattr(self, key, item)
    
    def __str__(self) -> str:
        return self.choices[0]["message"]["content"]
