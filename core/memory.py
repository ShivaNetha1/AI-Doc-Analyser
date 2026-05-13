from groq import Groq
from core.prompts import CONDENSE_PROMPT

class MemoryManager:
    def __init__(self, api_key, model_id):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id

    def condense_query(self, query, history):
        if not history:
            return query
            
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])
        prompt = f"{CONDENSE_PROMPT}\n\nHistory:\n{history_text}\n\nQuestion: {query}"
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
