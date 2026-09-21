from groq import Groq
from core.prompts import VALIDATOR_PROMPT

class ValidatorAgent:
    def __init__(self, api_key, model_id):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id

    def validate(self, answer, context_chunks, question=None):
        context_text = "\n\n".join([c['text'] for c in context_chunks])
        prompt = f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer to Validate:\n{answer}\n\n{VALIDATOR_PROMPT}"
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip().upper()
        
        print(f"\n\n\n--- [INPUTS PASSED TO VALIDATOR] ---")
        print(f"Context:\n\n\n\n\n{context_text}\n\n\n\n\n")
        print(f"Answer to Validate:\n\n\n\n{answer}")
        
        print(f"\n--- [VALIDATOR AGENT LOG] ---")
        print(f"\n\n\n\n\nRaw LLM Response: {content}")
        print(f'\n\n\nraww response :{response.choices}')
        
        REJECT_KEYWORDS = {"INVALID", "NO", "FALSE", "0", "UNSUPPORTED", "HALLUCINATION"}

        # If ANY negative word is in content -> Reject
        if any(neg in content for neg in REJECT_KEYWORDS):
            return False

        # if "INVALID" in content:
        #     print("Validation Decision: REJECTED (False)")
        #     return False
        APPROVE_KEYWORDS = {"VALID", "YES", "TRUE", "1", "SUPPORTED"}

        # If ANY positive word is in content -> Approve
        if any(pos in content for pos in APPROVE_KEYWORDS):
            return True

        # Default fallback if response is ambiguous -> Reject (Safe default)
        return False

            
        # is_valid = "VALID" in content

        # print(f"Validation Decision: {'APPROVED (True)' if is_valid else 'REJECTED (False)'}")
        # return is_valid





