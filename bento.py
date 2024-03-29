import bentoml

from typing import Optional, AsyncGenerator, List

MAX_TOKENS = 512
PROMPT_TEMPLATE = """{user_prompt}"""

@bentoml.service(
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "memory": "16Gi",
    },
)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        ENGINE_ARGS = AsyncEngineArgs(
            model='ugshanyu/mongol-mistral-3',
            max_model_len=MAX_TOKENS
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.request_id = 0

    @bentoml.api
    async def generate(self, prompt: str = "Explain superconductors like I'm five years old", tokens: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=MAX_TOKENS)
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
        stream = await self.engine.add_request(self.request_id, prompt, SAMPLING_PARAM, prompt_token_ids=tokens)
        self.request_id += 1
        async for request_output in stream:
            yield request_output.outputs[0].text
