class Message:
    def __init__(self, text: str, role: str):
        self.text = text
        self.role = role

    def __str__(self):
        return f"{self.role}: {self.text}"

class Context:
    def __init__(self):
        self.messages: list[Message] = []

    def add_message(self, role: str, text: str):
        self.messages.append(Message(text=text, role=role))
