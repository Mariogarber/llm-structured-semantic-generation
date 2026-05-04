from __future__ import annotations

from typing import Any


def mean_pool_generated_hidden_states(last_hidden_state: Any, prompt_token_count: int) -> Any | None:
    """Average the final-layer hidden states corresponding to generated tokens only."""

    if prompt_token_count < 0:
        raise ValueError("prompt_token_count must be non-negative")
    if last_hidden_state.ndim != 2:
        raise ValueError("last_hidden_state must have shape [sequence_length, hidden_size]")
    if prompt_token_count > last_hidden_state.shape[0]:
        raise ValueError("prompt_token_count cannot exceed sequence_length")

    generated_hidden = last_hidden_state[prompt_token_count:]
    if generated_hidden.shape[0] == 0:
        return None
    return generated_hidden.mean(dim=0)


def mean_pool_generate_hidden_states(generate_hidden_states: Any) -> Any | None:
    """Average final-layer hidden states returned by `model.generate(..., output_hidden_states=True)`.

    Hugging Face generation returns one entry per decoding step. Each step contains
    the hidden states for every layer. We take the last layer and the last position
    of each step, which corresponds to the newly generated token at that step.
    """

    if not generate_hidden_states:
        return None

    generated_vectors = []
    for step_hidden_states in generate_hidden_states:
        if not step_hidden_states:
            continue
        last_layer = step_hidden_states[-1]
        if last_layer.ndim != 3:
            raise ValueError("generation last-layer hidden states must have shape [batch, seq, hidden]")
        generated_vectors.append(last_layer[:, -1, :])

    if not generated_vectors:
        return None

    if len(generated_vectors) == 1:
        return generated_vectors[0][0]

    import torch

    stacked = torch.cat(generated_vectors, dim=0)
    return stacked.mean(dim=0)
