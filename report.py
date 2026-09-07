import os
from groq import Groq
import google.generativeai as genai

class AutopsyOrchestrator:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")

    def generate_report(self, model_evidence: dict, data_evidence: dict) -> str:
        prompt = f"""
        You are an expert ML Reliability Engineer. Analyze the following model inspection and data profiling evidence to produce a structured Autopsy Report.
        
        Model Evidence: {model_evidence}
        Data Evidence: {data_evidence}
        
        Provide the output with:
        1. Findings
        2. Evidence Summary
        3. Severity Level (Low/Medium/High/Critical)
        4. Recommended Fixes
        """
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as groq_error:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as gemini_error:
                return f"Both Groq and Gemini failed. Groq Error: {groq_error} | Gemini Error: {gemini_error}"
