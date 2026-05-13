from groq import Groq
from core.prompts import VALIDATOR_PROMPT

class ValidatorAgent:
    def __init__(self, api_key, model_id):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id

    def validate(self, answer, context_chunks):
        context_text = "\n\n".join([c['text'] for c in context_chunks])
        prompt = f"Context:\n{context_text}\n\nAnswer to Validate:\n{answer}\n\n{VALIDATOR_PROMPT}"
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return "VALID" in response.choices[0].message.content.upper()
