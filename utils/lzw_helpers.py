from __future__ import annotations

import numpy as np


RANDOM_BASELINE_SYMBOLS = np.array(["1", "2", "3", "4"])


def _make_rng(rng: np.random.Generator | int | None = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _compress_codes(uncompressed: str) -> tuple[list[int], list[str]]:
    alphabet = list(dict.fromkeys(uncompressed))
    dictionary = {ch: idx for idx, ch in enumerate(alphabet)}
    dict_size = len(dictionary)

    w = ""
    result: list[int] = []

    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    if w:
        result.append(dictionary[w])

    return result, alphabet


def lzw_compress(
    uncompressed: str,
    return_ratio: bool = False,
    return_alphabet: bool = False,
    return_random_ratio: bool = False,
    return_random_lengths: bool = False,
    n_random_samples: int = 100,
    rng: np.random.Generator | int | None = None,
):
    """Compress a string into LZW integer codes.

    Parameters
    ----------
    uncompressed:
        Input string to compress.
    return_ratio:
        If True, also return len(uncompressed) / len(compressed_codes).
    return_alphabet:
        If True, also return the initial alphabet used for compression.
    return_random_ratio:
        If True, also return the ratio between the compressed random sequence
        length and the compressed original sequence length.
    return_random_lengths:
        If True, also return the distribution of compressed random sequences
    n_random_samples:
        Number of random baseline sequences to sample.
    rng:
        Optional NumPy random generator or seed for reproducible shuffles.
    """
    result, alphabet = _compress_codes(uncompressed)

    ratio = None
    if return_ratio:
        ratio = 1.0 if len(uncompressed) == 0 else len(uncompressed) / len(result)

    random_ratio = None
    random_lengths: list[int] = []
    if return_random_ratio or return_random_lengths:
        if n_random_samples < 1:
            raise ValueError("n_random_samples must be at least 1")

        generator = _make_rng(rng)
        original_len = len(result)
        input_len = len(uncompressed)

        for _ in range(n_random_samples):
            random_seq = "".join(generator.choice(RANDOM_BASELINE_SYMBOLS, size=input_len, replace=True))
            random_codes, _ = _compress_codes(random_seq)
            random_lengths.append(len(random_codes))

        if return_random_ratio:
            if original_len == 0:
                random_ratio = 1.0
            else:
                random_ratio = float(np.mean([length / original_len for length in random_lengths]))

    outputs = [result]

    if return_ratio:
        outputs.append(ratio)

    if return_random_ratio:
        outputs.append(random_ratio)

    if return_alphabet:
        outputs.append(alphabet)

    if return_random_lengths:
        outputs.append(np.array(random_lengths))

    if len(outputs) == 1:
        return result
    if len(outputs) == 2:
        return outputs[0], outputs[1]
    if len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2]
    if len(outputs) == 4:
        return outputs[0], outputs[1], outputs[2], outputs[3]
    return tuple(outputs)


def lzw_decompress(compressed: list[int], alphabet: list[str] | None = None) -> str:
    """Decompress LZW integer codes back into a string.

    If ``alphabet`` is provided, it should be the initial alphabet returned by
    ``lzw_compress(..., return_alphabet=True)``.
    """
    if alphabet is None:
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
    else:
        dict_size = len(alphabet)
        dictionary = {i: ch for i, ch in enumerate(alphabet)}

    if not compressed:
        return ""

    if compressed[0] not in dictionary:
        raise ValueError(f"Bad first compressed code: {compressed[0]}")
    w = dictionary[compressed[0]]
    result = [w]

    for code in compressed[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = w + w[0]
        else:
            raise ValueError(f"Bad compressed code: {code}")

        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        w = entry

    return "".join(result)
