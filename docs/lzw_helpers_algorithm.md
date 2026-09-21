# LZW Helper Algorithms

This document explains the algorithms implemented in [utils/lzw_helpers.py](../utils/lzw_helpers.py).

## Overview

The module provides four functions:

- `_make_rng`: normalizes random number generator input.
- `_compress_codes`: core LZW compression routine using a dynamic alphabet.
- `lzw_compress`: user-facing compressor with optional summary statistics.
- `lzw_decompress`: decoder for code sequences produced by this implementation.

The key idea is LZW dictionary growth: previously seen substrings are assigned integer codes so repeated patterns become fewer output symbols.

## 1) Random Generator Setup

### Function

- `_make_rng(rng)`

### Algorithm

1. If `rng` is already a NumPy `Generator`, return it unchanged.
2. Otherwise call `np.random.default_rng(rng)`.

This allows:

- `None` for non-deterministic behavior,
- an integer seed for reproducibility,
- or a pre-created generator object.

## 2) Core Compression

### Function

- `_compress_codes(uncompressed: str) -> (codes, alphabet)`

### Initial Dictionary

The initial alphabet is derived from first occurrence order in the input string:

1. Iterate the input once and keep first-seen characters.
2. Assign codes `0..k-1` to these characters.

So if input starts as `A B A C`, initial alphabet is `[A, B, C]` and dictionary is `{A:0, B:1, C:2}`.

### Encoding Loop

Let `w` be the current matched phrase and `c` the next character.

1. Form candidate `wc = w + c`.
2. If `wc` exists in dictionary, set `w = wc` (extend the match).
3. Else:
   - emit code for `w`,
   - add `wc` to dictionary with next free code,
   - reset `w = c`.
4. After the loop, emit `w` if non-empty.

### Output

- `codes`: list of emitted integer codes.
- `alphabet`: initial alphabet required for exact decode compatibility.

## 3) Public Compression Wrapper

### Function

- `lzw_compress(...)`

### Base Behavior

Always computes:

- `result, alphabet = _compress_codes(uncompressed)`

### Optional Metrics

#### Compression ratio (`return_ratio=True`)

Returns:

$$
\text{ratio} =
\begin{cases}
1.0, & |s|=0 \\
\frac{|s|}{|C(s)|}, & |s|>0
\end{cases}
$$

where:

- $s$ is the original string,
- $C(s)$ is the compressed code list,
- $|\cdot|$ is length.

#### Random baseline ratio (`return_random_ratio=True`)

The function samples `n_random_samples` synthetic random strings with:

- length exactly equal to `len(uncompressed)`, and
- symbols drawn from the fixed set `['1', '2', '3', '4']`.

1. For each sample, generate a random string of length $|s|$ from `['1','2','3','4']`.
2. Compress the random string.
3. Store the compressed length.

Then compute the average ratio:

$$
	ext{random\_ratio} = \frac{1}{N} \sum_{i=1}^{N} \frac{|C(r_i)|}{|C(s)|}
$$

where each $r_i$ is a random sequence in $\{1,2,3,4\}^{|s|}$.

Special case:

- if `|C(s)| = 0`, returns `1.0`.

This metric compares how compressible the observed sequence is relative to fixed-alphabet random sequences of the same length.

#### Random length distribution (`return_random_lengths=True`)

When requested, the function appends a NumPy array containing the sampled compressed lengths for those fixed-alphabet random sequences.

### Return Packing Strategy

The function builds an output list starting with `result` and appends optional outputs in this order:

1. `ratio` (if requested),
2. `random_ratio` (if requested),
3. `alphabet` (if requested),
4. `random_lengths` (if requested).

It then returns either:

- a single object (`result`) if only one value is present,
- a tuple for 2 or more outputs.

## 4) Decompression

### Function

- `lzw_decompress(compressed, alphabet=None) -> str`

### Dictionary Initialization

Two modes are supported:

1. `alphabet is None`: initialize legacy dictionary with 256 ASCII entries (`0..255`).
2. `alphabet provided`: initialize dictionary from that alphabet with codes `0..k-1`.

### Decode Loop

1. Read first code, map to first string fragment `w`.
2. For each next code:
   - if code exists in dictionary, use it,
   - else if code equals next dictionary index (`dict_size`), use LZW edge case `w + w[0]`,
   - otherwise raise `ValueError`.
3. Append decoded fragment to result.
4. Add new dictionary entry `w + entry[0]`.
5. Set `w = entry` and continue.

Finally join fragments into the recovered string.

## Complexity Notes

Let $n$ be input length.

- Core compression: approximately $O(n)$ average behavior for typical dictionary access.
- Decompression: approximately $O(m)$ where $m$ is number of compressed codes.
- Random baseline mode adds $N$ extra compressions, so roughly $O(Nn)$.

## Practical Interpretation of Ratios

- Larger `ratio = |s|/|C(s)|` means stronger compression of the observed sequence.
- `random_ratio > 1` means fixed-alphabet random sequences tend to compress to longer code streams than the observed sequence.
- `random_ratio ≈ 1` suggests observed order is not much more compressible than this fixed-alphabet random baseline.
