"""A small, real byte-level language-model experiment, runnable on any CPU."""

from benchmark import run

ORDER = 1
ALPHA = 1.0
BACKOFF = 0.5

if __name__ == "__main__":
    run(ORDER, ALPHA, BACKOFF)
