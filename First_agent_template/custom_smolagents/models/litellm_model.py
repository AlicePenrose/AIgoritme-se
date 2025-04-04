import litellm

class LitellmModel:
    def __init__(
        self,
        model_id: str = "ollama/llava",
        temperature: float = 0.5,
        max_tokens: int = 1024,
        stop: list = None,
        **kwargs
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop or []

    def run(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )
        return response.choices[0].message["content"]
