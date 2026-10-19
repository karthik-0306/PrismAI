"""
backend/evaluation/test_models.py

Runs the rewriter's compression prompt against every configured model, one at a
time (no fallback), to compare their output side by side.

    python -m backend.evaluation.test_models
"""

import asyncio

from backend import config
from backend.llm.client import LLMClient
from backend.pipeline.rewriter import _COMPRESS_SYSTEM

MESSAGE = (
    "I have 1000 documents to download from a website. So as not to overload the servers "
    "1) at what rate should I download? Just pick a good rate for the sake of the question "
    "then answer:2)  how long will it take to download all the files?"
)


async def main():
    client = LLMClient()
    for model in config.ALL_MODELS:
        print(f"\n---------------- {config.model_label(model)} ({model})")
        try:
            result = await client.complete(
                [model],
                [{"role": "system", "content": _COMPRESS_SYSTEM}, {"role": "user", "content": MESSAGE}],
                temperature=0.3,
                max_tokens=200,
                low_reasoning=True,
            )
            print(result.content)
        except Exception as e:
            print("FAILED:", e)


if __name__ == "__main__":
    asyncio.run(main())
