import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful customer support assistant. Answer the user's question based on the FAQ entries provided. Be concise, friendly, and professional. Do not mention FAQ entries in your answer."""

def get_ai_answer(user_query, faq_candidates, chat_history=None):
    if faq_candidates:
        context = "Relevant FAQ entries:\n\n"
        for i, faq in enumerate(faq_candidates, 1):
            context += f"[{i}] Q: {faq['question']}\n    A: {faq['answer']}\n\n"
    else:
        context = "No relevant FAQ entries found.\n"

    prompt = f"{context}\nUser question: {user_query}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": [f["question"] for f in faq_candidates[:3]],
        "used_fallback": not faq_candidates,
        "model": "llama-3.1-8b-instant",
    }
