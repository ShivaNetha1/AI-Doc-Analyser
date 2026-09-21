import json
from groq import Groq
from pydantic import BaseModel, Field

class ValidationSchema(BaseModel):
    is_valid: bool = Field(description="True if the answer is completely supported by the retrieved context, False if it contains ungrounded information or hallucinations.")
    reasoning: str = Field(description="Brief explanation justifying why the answer was marked as valid or invalid.")

class ValidatorAgent:
    def __init__(self, api_key, model_id):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id

    def validate(self, answer, context_chunks, question=None):
        context_text = "\n\n".join([c['text'] for c in context_chunks])
        
        prompt = f"""Question: {question}

Context:
{context_text}

Answer to Validate:
{answer}

You are a strict Document Validation Agent. Your job is to check if the AI's answer is accurate, truthful to the Question, and fully supported by the Context.

Validation Rules:
1. ENTITY & QUESTION ALIGNMENT: The answer MUST specifically answer the subject asked in the Question. If the Question asks about a specific entity/topic (e.g. "Microsoft"), but the Context is about a different entity, providing facts about the wrong entity is INVALID (is_valid: false).
2. FACTUAL GROUNDING: Every claim in the Answer must be directly supported by the Context.
3. CORRECT REFUSAL: If the Answer correctly states "I could not find this information in the provided documents" because the requested topic is absent from the Context, mark is_valid: true.

You MUST respond ONLY with a valid JSON object matching this exact JSON schema:
{{
  "is_valid": true or false,
  "reasoning": "Brief explanation of why the answer is valid or invalid"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Parse and validate against Pydantic schema
            validation_result = ValidationSchema.model_validate_json(raw_content)
            
            print(f"\n\n\n\n\n--- [STRUCTURED VALIDATOR AGENT LOG] ---\n\n\n\n\n")
            print(f"\n\n\n\n\n Reasoning: {validation_result.reasoning}\n\n\n\n\n")
            print(f"\n\n\n\n\n Validation Decision: {'APPROVED (True)' if validation_result.is_valid else 'REJECTED (False)'}\n\n\n\n\n")
            
            return validation_result.is_valid

        except Exception as e:
            print(f"\n\n\n\n\n\n--- [VALIDATOR ERROR / FALLBACK LOG] ---\n\n\n\n\n")
            print(f"\n\n\n\n\n Validation failed with exception: {e}\n\n\n\n\n")
            return False
