# Complete naive pass@4 baseline

All 30 questions and four rollouts each, BF16 VibeThinker-3B at 95% memory and 16K total context. Standard naive prompt and final-only extraction; no sibling or target cancellation.

57/120 correct samples, 18/30 pass@4 and unique-plurality voting, 13/30 strict three-of-four voting. All 63 capped samples had no eligible final answer. See [batch results](../../results/post_freeze/measurements-v1-20261004T104200Z/summary.md) for exact definitions, timings and the historical 80% timing reference.
