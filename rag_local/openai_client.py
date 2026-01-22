import openai,os
from dotenv import load_dotenv
load_dotenv()
 
client = openai.OpenAI(api_key=os.getenv("KITVIEW_DESKTOP_OPENAI_API_KEY"))

def call_llm(prompt: str, model="gpt-4.1-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Tu es un assistant IA professionnel spécialisé en analyse documentaire pour Kitview."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600
    )
    
    return response.choices[0].message.content.strip()
