"""Frozen procedural corpus and byte n-gram evaluator for exercising the engine.

No neural network, LLM proposals, GPU, or five-minute claim is involved in this demo.
Train, validation and test sets are disjoint sentences generated from one grammar.
"""

import itertools
import math
import random
import sys
import time
from collections import Counter, defaultdict


def corpus():
    sentences = [
        f"the {adjective} {animal} {verb} near the {place}.\n"
        for adjective, animal, verb, place in itertools.product(
            ["small", "bright", "quiet", "quick", "gentle", "sleepy"],
            ["fox", "bird", "cat", "dog", "mouse", "horse"],
            ["waits", "plays", "sleeps", "runs", "sings"],
            ["river", "forest", "garden", "house", "hill"],
        )
    ]
    random.Random(947).shuffle(sentences)
    return sentences[:600], sentences[600:750], sentences[750:]


def run(order, alpha, backoff):
    started = time.monotonic()
    if not isinstance(order, int) or not 1 <= order <= 5 or alpha <= 0 or not 0 < backoff < 1:
        raise ValueError("invalid n-gram parameters")
    train, validation, test = corpus()
    counts = [defaultdict(Counter) for _ in range(order)]
    for sentence in train:
        data = bytes(order) + sentence.encode()
        for i in range(order, len(data)):
            for n in range(order):
                counts[n][data[i - n : i]][data[i]] += 1
    totals = [
        {context: sum(counter.values()) for context, counter in table.items()} for table in counts
    ]
    training_seconds = time.monotonic() - started
    heldout = "--heldout" in sys.argv
    examples = test if heldout else validation
    bits = tokens = 0
    for sentence in examples:
        data = bytes(order) + sentence.encode()
        for i in range(order, len(data)):
            probability = 1 / 256
            for n in range(order):
                context = data[i - n : i]
                count = counts[n].get(context, {}).get(data[i], 0)
                estimate = (count + alpha) / (totals[n].get(context, 0) + alpha * 256)
                probability = backoff * probability + (1 - backoff) * estimate
            bits -= math.log2(probability)
            tokens += 1
    print(f"{'test_bpb' if heldout else 'val_bpb'}: {bits / tokens:.8f}")
    print(f"training_seconds: {training_seconds:.6f}")
    print("peak_vram_mb: 0")
    print(f"total_seconds: {time.monotonic() - started:.6f}")
