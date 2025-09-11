#------------------------------------------------------------------------------
# caesar_cipher.py
#
# This module provides basic utilities for working with Caesar's cipher and
# related key-recovery experiments. It includes functions for cleaning input
# text, decrypting with a given key, deriving a key from a known plaintext–
# ciphertext pair (crib), and converting integers to binary strings.
#
# Constants:
# - ALPH: Alphabet used for Caesar's cipher (A–Z).
# - A2I: Mapping from alphabet letters to integers.
# - I2A: Mapping from integers to alphabet letters.
#
# Functions included:
# - clean_text(s): Strips invalid characters from a string.
# - caesar_decrypt(ciphertext, key): Decrypts a ciphertext given the key.
# - key_from_crib(plain_char, cipher_char): Infers the Caesar key from a
#   plaintext–ciphertext character pair.
# - int_to_bits(x, n): Converts an integer into an n-bit binary string.
#
# These utilities support cryptographic experiments, toy cipher analysis,
# and demonstrations of quantum/classical key-recovery strategies.
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------


ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A2I = {c: i for i, c in enumerate(ALPH)}
I2A = {i: c for i, c in enumerate(ALPH)}


def clean_text(s):
    """Strips invalid characters from a string, keeping only A–Z.

    Args:
        s (str): Input string.

    Returns:
        str: Cleaned uppercase string containing only valid letters.
    """
    return "".join(ch for ch in s.upper() if ch in A2I)


def caesar_decrypt(ciphertext: str, key: int) -> str:
    """Decrypts a ciphertext using Caesar's cipher with the given key.

    Args:
        ciphertext (str): Encrypted text.
        key (int): Shift amount used for decryption (0–25).

    Returns:
        str: Decrypted plaintext.
    """
    c = clean_text(ciphertext)
    return "".join(I2A[(A2I[ch] - key) % 26] for ch in c)


def key_from_crib(plain_char: str, cipher_char: str) -> int:
    """Derives the Caesar key from a known plaintext–ciphertext letter pair.

    Args:
        plain_char (str): Plaintext character (A–Z).
        cipher_char (str): Ciphertext character (A–Z).

    Returns:
        int: Inferred key value in the range 0–25.
    """
    return (A2I[cipher_char.upper()] - A2I[plain_char.upper()]) % 26


def int_to_bits(x: int, n: int) -> str:
    """Converts an integer into an n-bit binary string.

    Args:
        x (int): Integer value.
        n (int): Number of bits in the output string.

    Returns:
        str: Binary representation of x with length n.
    """
    return format(x, f"0{n}b")
