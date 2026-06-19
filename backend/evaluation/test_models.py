import asyncio
from backend.llm.client import LLMClient
from backend.pipeline.rewriter import _COMPRESS_SYSTEM

async def test():
    client = LLMClient()
    message = "I have 1000 documents to download from a website. So as not to overload the servers 1) at what rate should I download? Just pick a good rate for the sake of the question then answer:2)  how long will it take to download all the files?"
    
    print("Testing Llama 3.1 8B:")
    try:
        r1 = await client.async_complete(
            model="groq/llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM},
                {"role": "user", "content": f"<query>\n{message}\n</query>"}
            ],
            temperature=0.3
        )
        print(r1.content)
    except Exception as e:
        print("Llama failed:", e)

    print("\n----------------\nTesting Gemini 3.5 Flash:")
    try:
        r2 = await client.async_complete(
            model="gemini/gemini-3.5-flash",
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM},
                {"role": "user", "content": f"<query>\n{message}\n</query>"}
            ],
            temperature=0.3
        )
        print(r2.content)
    except Exception as e:
        print("Gemini failed:", e)

asyncio.run(test())
