class ConversationContext:
    def __init__(self, max_history=3):
        self.max_history = max_history
        self.history = []

    def add_exchange(self, question, answer):
        self.history.append({"question": question, "answer": answer})

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context_string(self):
        if not self.history:
            return ""

        context_parts = ["previous conversation:"]
        for i, exchange in enumerate(self.history, 1):
            context_parts.append(f"Q{i}: {exchange['question']}")
            context_parts.append(f"A{i}: {exchange['answer']}")

        return "\n".join(context_parts)

    def clear(self):
        self.history = []

    def get_history(self):
        return self.history
