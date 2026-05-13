from groq import Groq

class CalculatorAgent:
    def __init__(self, api_key, model_id):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id

    def calculate(self, query, context_chunks):
        context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['text']}" for c in context_chunks])
        prompt = f"Extract numerical data from the context and solve: {query}\n\nContext:\n{context_text}\n\nProvide the result and mention which document the numbers were found in."
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
