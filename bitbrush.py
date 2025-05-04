"""
bitbrush.py

A novel, high-performance and readable bit manipulation toolkit with optional NumPy acceleration.

This module introduces a unique approach to "bit brushing" —
conceptual and practical code to systematically explore, test,
and manipulate bits using expressive primitives. It supports
both pure‑Python and NumPy‑backed operations for large‑scale
bit pattern generation.
"""

import numpy as np
from typing import Generator, Union, Literal, List


class BitBrush:
    """
    BitBrush: A high‑performance utility for bit‑level pattern manipulation.

    Supports both pure‑Python and NumPy backends for key operations.
    """

    def __init__(self, width: int = 32, backend: Literal['python', 'numpy'] = 'python'):
        """
        Initialize a BitBrush instance.

        Args:
            width (int):
                Number of bits to operate on. Default is 32.
            backend (str):
                Which implementation to use: 'python' or 'numpy'.
                Default is 'python'.

        Attributes:
            width (int): Configured bit‑width.
            mask (int): Bitmask of all 1s for the given width.
            backend (str): Selected backend mode.
            _mirror_lut (List[int]): Lookup table for fast byte‑wise bit reversal.
        """
        self.width = width
        self.mask = (1 << width) - 1
        self.backend = backend
        self._mirror_lut = self._build_mirror_lut()

    def _build_mirror_lut(self) -> List[int]:
        """
        Build a lookup table (LUT) for 8‑bit reversed values.

        This table maps each byte value 0–255 to its bit‑reversed counterpart,
        enabling fast byte‑wise reversal for larger integers.

        Returns:
            List[int]: A 256‑element list where index i yields the bits of i reversed.
        """
        lut = [0] * 256
        for i in range(256):
            b = i
            rev = 0
            for _ in range(8):
                rev = (rev << 1) | (b & 1)
                b >>= 1
            lut[i] = rev
        return lut

    def sweep_ones(self) -> Union[Generator[int, None, None], np.ndarray]:
        """
        Generate values with a single '1' bit sweeping from LSB to MSB.

        Python backend yields each pattern lazily; NumPy backend returns
        a vector of all patterns at once.

        Returns:
            Generator[int, None, None] or np.ndarray:
                - Python: generator of ints 1<<0, 1<<1, …, 1<<(width-1)
                - NumPy : np.ndarray of dtype uint64 containing same values
        """
        if self.backend == 'numpy':
            return np.left_shift(np.uint64(1), np.arange(self.width, dtype=np.uint64))
        return (1 << i for i in range(self.width))

    def sweep_zeros(self) -> Union[Generator[int, None, None], np.ndarray]:
        """
        Generate values with all bits set except one '0' sweeping LSB→MSB.

        Returns:
            Generator[int, None, None] or np.ndarray:
                - Python: generator of mask^(1<<i) for i in 0…width-1
                - NumPy : np.ndarray of dtype uint64 with same values
        """
        if self.backend == 'numpy':
            ones = np.left_shift(np.uint64(1), np.arange(self.width, dtype=np.uint64))
            return np.bitwise_xor(self.mask, ones)
        return (self.mask ^ (1 << i) for i in range(self.width))

    def toggle_sparse(self, step: int = 3) -> Union[Generator[int, None, None], np.ndarray]:
        """
        Generate values with bits cumulatively toggled at fixed intervals.

        For step=k, bits at positions 0, k, 2k… are set one by one.

        Args:
            step (int): Interval between toggled bits. Default is 3.

        Returns:
            Generator[int, None, None] or np.ndarray:
                - Python: generator yielding cumulative OR of 1<<i at each step
                - NumPy : np.ndarray of dtype uint64 with same sequence
        """
        if self.backend == 'numpy':
            indices = np.arange(0, self.width, step, dtype=np.uint64)
            arr = np.zeros(indices.shape, dtype=np.uint64)
            for idx in range(indices.size):
                bit = np.left_shift(np.uint64(1), indices[idx])
                arr[idx] = arr[idx - 1] | bit if idx > 0 else bit
            return arr

        val = 0
        for i in range(0, self.width, step):
            val |= 1 << i
            yield val

    def mirror_mask(self, value: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Reverse the bit order of the input within the configured width.

        Uses a byte‑wise lookup table for efficiency. Supports both single
        integer and NumPy array inputs when backend='numpy'.

        Args:
            value (int or np.ndarray):
                Input bits to mirror. If ndarray, must be dtype uint64.

        Returns:
            int or np.ndarray:
                Bit‑reversed output, same type as input.
        """
        bytes_needed = (self.width + 7) // 8

        # NumPy path
        if self.backend == 'numpy' and isinstance(value, np.ndarray):
            result = np.zeros_like(value, dtype=np.uint64)
            for i in range(bytes_needed):
                chunk = np.right_shift(value, np.uint64(i * 8)) & np.uint64(0xFF)
                rev = np.take(self._mirror_lut, chunk)
                result |= np.left_shift(rev.astype(np.uint64),
                                       np.uint64((bytes_needed - 1 - i) * 8))
            return np.right_shift(result, np.uint64(bytes_needed * 8 - self.width))

        # Pure‑Python path
        result = 0
        for i in range(bytes_needed):
            byte = (value >> (i * 8)) & 0xFF
            rev = self._mirror_lut[byte]
            result |= rev << ((bytes_needed - 1 - i) * 8)
        return result >> (bytes_needed * 8 - self.width)

    def scan_patterns(self) -> Union[Generator[int, None, None], np.ndarray]:
        """
        Generate symmetric bit patterns expanding from center outward.

        Patterns have 1s at equal distance from center. Length of sequence
        is floor(width/2)+1.

        Returns:
            Generator[int, None, None] or np.ndarray:
                - Python: generator yielding int patterns
                - NumPy : np.ndarray of dtype uint64 with same patterns
        """
        center = self.width // 2

        if self.backend == 'numpy':
            radii = np.arange(center + 1, dtype=np.uint64)
            left = np.left_shift(np.uint64(1), np.uint64(center) - radii)
            right = np.where(center + radii < self.width,
                             np.left_shift(np.uint64(1), np.uint64(center) + radii),
                             np.uint64(0))
            return np.bitwise_or(left, right)

        for radius in range(center + 1):
            left = 1 << (center - radius)
            right = 1 << (center + radius) if center + radius < self.width else 0
            yield left | right

    def count_ones(self, value: int) -> int:
        """
        Count the number of set bits (1s) in an integer.

        Args:
            value (int): Input integer to analyze.

        Returns:
            int: Number of bits set to 1 within the configured width.
        """
        return bin(int(value) & self.mask).count('1')

    def visualize(self, value: int) -> str:
        """
        Return a zero‑padded binary string representation of the value.

        Args:
            value (int): Input integer to convert.

        Returns:
            str: Binary string of length `width`, with leading zeros.
        """
        return format(int(value) & self.mask, f'0{self.width}b')
