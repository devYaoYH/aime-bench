"""Wrong-verdict feedback only; never obtains reference answers."""
import json

POLICY = {
    "id": "wrong-verdict-queued-batch-v2.2",
    "trigger": "actual boolean false grader verdict",
    "defer": "queued unique candidates and detected candidates awaiting validation",
    "prefix": "exact prompt/output token IDs plus server-tokenized feedback, no BOS",
    "budget": "correction continuations count toward four generation requests",
    "scope": "unsolved question only; retain existing prompt, parser and scheduler",
}


def feedback_text(answers):
    return ("\n\n[External grader feedback]\nThe grader rejected these submitted "
            "answers to the original question as incorrect: "
            + json.dumps(answers, ensure_ascii=False)
            + ". Reconsider the reasoning and correct the answer. "
            "Continue solving the original question; emit a new prospective exact "
            "answer in a box when ready.\n[/External grader feedback]\n\n")


async def tokenize_feedback(client, args, text):
    response = await client.post(args.vllm_url + "/tokenize", json={
        "model": args.model, "prompt": text, "add_special_tokens": False,
    })
    response.raise_for_status()
    value = response.json()
    tokens = value.get("tokens")
    if (not isinstance(tokens, list) or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or value.get("count") != len(tokens)):
        raise RuntimeError("Invalid exact feedback token IDs from vLLM")
    return tokens
