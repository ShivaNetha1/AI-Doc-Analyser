from groq import Groq
from core.prompts import ANSWER_PROMPT

class RetrievalAgent:
    def __init__(self, api_key, model_id):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id

    def generate_answer(self, query, context_chunks):
        context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['text']}" for c in context_chunks])
        prompt = ANSWER_PROMPT.format(context=context_text, question=query)
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
