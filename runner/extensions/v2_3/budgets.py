"""Reserve prompt/feedback tokens inside the existing total context window."""
import asyncio


async def prompt_tokens(client, args, messages):
    request = {"model": args.model, "messages": messages, "add_generation_prompt": True,
               "add_special_tokens": False}
    if args.disable_thinking:
        request["chat_template_kwargs"] = {"enable_thinking": False}
    response = await client.post(args.vllm_url + "/tokenize", json=request)
    response.raise_for_status()
    value = response.json()
    count, tokens = value.get("count"), value.get("tokens")
    if (type(count) is not int or not 0 < count < args.max_context_tokens
            or not isinstance(tokens, list) or count != len(tokens)
            or any(type(t) is not int or t < 0 for t in tokens)):
        raise ValueError("Invalid served chat prompt token evidence")
    return count


async def prepare_prompts(problems, args, client):
    async def one(problem):
        messages = [{"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": problem["problem"]}]
        return {**problem, "prompt_tokens": await prompt_tokens(client, args, messages)}
    return await asyncio.gather(*(one(p) for p in problems))
