from groq import Groq
import os
from core.prompts import ROUTER_PROMPT

class RouterAgent:
    def __init__(self, api_key, model_id):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id

    def route(self, query, history):
        messages = [{"role": "system", "content": ROUTER_PROMPT}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": query})
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages
        )
        return response.choices[0].message.content.strip().upper()
