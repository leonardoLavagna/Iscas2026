#------------------------------------------------------------------------------
# improved_aes.py
#
# This module provides the AES S-box and its inverse (FIPS-197 tables),
# together with a toy AES-like construction operating on 16-byte blocks.
#
# Original toy primitive:
#   c = S_BOX[p ⊕ k]
#
# This version extends the primitive by adding an AES-style diffusion
# layer (ShiftRows + MixColumns), while keeping a single-byte key and
# a simple public API.
#
# Encryption on each 16-byte block:
#   state = SubBytes(state ⊕ k)
#   state = ShiftRows(state)
#   state = MixColumns(state)
#
# Decryption applies the inverse steps:
#   state = InvMixColumns(state)
#   state = InvShiftRows(state)
#   state = InvSubBytes(state) ⊕ k
#
# Blocks are interpreted as a 4×4 AES state in column-major order.
#
# Padding:
#   Zero-byte padding is used to reach a multiple of 16 bytes.
#   For plaintexts that never contain '\x00', unpadding via rstrip
#   is safe and deterministic.
#
#
# © Giacomo Vittori 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------

# AES S-box (FIPS-197, Table 4.1)
S_BOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
]

# AES inverse S-box
S_INV = [0] * 256
for i, v in enumerate(S_BOX):
    S_INV[v] = i


def aes_sbox(x: int) -> int:
    """
    Forward AES S-box mapping.

    Args:
        x (int): Byte value 0..255.
    Returns:
        int: S_BOX[x].
    """
    return S_BOX[x & 0xff]


def aes_sbox_inv(y: int) -> int:
    """
    Inverse AES S-box mapping.

    Args:
        y (int): Byte value 0..255.
    Returns:
        int: Inverse S-box output S_INV[y].
    """
    return S_INV[y & 0xff]


# ---------------- AES-like diffusion layer ----------------

def _xtime(a: int) -> int:
    """
    Multiply a byte by x (i.e., by 2) in GF(2^8).

    Args:
        a (int): Byte value 0..255.
    Returns:
        int: Resulting byte.
    """
    a &= 0xff
    return ((a << 1) & 0xff) ^ (0x1b if (a & 0x80) else 0x00)


def _gf_mul(a: int, b: int) -> int:
    """
    Multiply two bytes in the AES finite field GF(2^8).

    Args:
        a (int): First byte.
        b (int): Second byte.
    Returns:
        int: Product in GF(2^8).
    """
    a &= 0xff
    b &= 0xff
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        a = _xtime(a)
        b >>= 1
    return res & 0xff


def _shift_rows(state: list[int]) -> list[int]:
    """
    AES-style ShiftRows operation.

    Args:
        state (list[int]): 16-byte state (column-major).
    Returns:
        list[int]: Shifted state.
    """
    out = state[:]
    for r in range(4):
        row = [state[r + 4*c] for c in range(4)]
        row = row[r:] + row[:r]
        for c in range(4):
            out[r + 4*c] = row[c]
    return out


def _inv_shift_rows(state: list[int]) -> list[int]:
    """
    Inverse AES-style ShiftRows operation.

    Args:
        state (list[int]): 16-byte state (column-major).
    Returns:
        list[int]: Unshifted state.
    """
    out = state[:]
    for r in range(4):
        row = [state[r + 4*c] for c in range(4)]
        row = row[-r:] + row[:-r] if r else row
        for c in range(4):
            out[r + 4*c] = row[c]
    return out


def _mix_columns(state: list[int]) -> list[int]:
    """
    AES-style MixColumns diffusion layer.

    Args:
        state (list[int]): 16-byte state (column-major).
    Returns:
        list[int]: Mixed state.
    """
    out = state[:]
    for c in range(4):
        col = [state[r + 4*c] for r in range(4)]
        a0, a1, a2, a3 = col
        m = [
            _gf_mul(a0,2) ^ _gf_mul(a1,3) ^ a2 ^ a3,
            a0 ^ _gf_mul(a1,2) ^ _gf_mul(a2,3) ^ a3,
            a0 ^ a1 ^ _gf_mul(a2,2) ^ _gf_mul(a3,3),
            _gf_mul(a0,3) ^ a1 ^ a2 ^ _gf_mul(a3,2),
        ]
        for r in range(4):
            out[r + 4*c] = m[r] & 0xff
    return out


def _inv_mix_columns(state: list[int]) -> list[int]:
    """
    Inverse AES-style MixColumns operation.

    Args:
        state (list[int]): 16-byte state (column-major).
    Returns:
        list[int]: State after inverse mixing.
    """
    out = state[:]
    for c in range(4):
        col = [state[r + 4*c] for r in range(4)]
        a0, a1, a2, a3 = col
        m = [
            _gf_mul(a0,0x0e) ^ _gf_mul(a1,0x0b) ^ _gf_mul(a2,0x0d) ^ _gf_mul(a3,0x09),
            _gf_mul(a0,0x09) ^ _gf_mul(a1,0x0e) ^ _gf_mul(a2,0x0b) ^ _gf_mul(a3,0x0d),
            _gf_mul(a0,0x0d) ^ _gf_mul(a1,0x09) ^ _gf_mul(a2,0x0e) ^ _gf_mul(a3,0x0b),
            _gf_mul(a0,0x0b) ^ _gf_mul(a1,0x0d) ^ _gf_mul(a2,0x09) ^ _gf_mul(a3,0x0e),
        ]
        for r in range(4):
            out[r + 4*c] = m[r] & 0xff
    return out


def _pad16(data: bytes) -> bytes:
    """
    Pad data with zero bytes to a multiple of 16.

    Args:
        data (bytes): Input data.
    Returns:
        bytes: Zero-padded data.
    """
    return data + b"\x00" * ((-len(data)) % 16)


def _unpad16_zeros(data: bytes) -> bytes:
    """
    Remove trailing zero-byte padding.

    Args:
        data (bytes): Padded data.
    Returns:
        bytes: Unpadded data.
    """
    return data.rstrip(b"\x00")


def _encrypt_block(block: bytes, k: int) -> bytes:
    """
    Encrypt a single 16-byte block.

    Args:
        block (bytes): Plaintext block (16 bytes).
        k (int): Byte key.
    Returns:
        bytes: Encrypted block.
    """
    state = [aes_sbox(b ^ k) for b in block]
    state = _shift_rows(state)
    state = _mix_columns(state)
    return bytes(state)


def _decrypt_block(block: bytes, k: int) -> bytes:
    """
    Decrypt a single 16-byte block.

    Args:
        block (bytes): Ciphertext block (16 bytes).
        k (int): Byte key.
    Returns:
        bytes: Decrypted block.
    """
    state = list(block)
    state = _inv_mix_columns(state)
    state = _inv_shift_rows(state)
    state = [aes_sbox_inv(b) ^ k for b in state]
    return bytes(state)


# ---------------- Public API ----------------

def encrypt_sbox_xor(plaintext: str, k: int) -> bytes:
    """
    AES-like toy encryption on 16-byte blocks.

    Args:
        plaintext (str): Input text (ASCII / no '\\x00' recommended).
        k (int): Byte key 0..255.
    Returns:
        bytes: Ciphertext bytes (multiple of 16).
    """
    data = plaintext.encode("latin-1")
    data = _pad16(data)

    out = bytearray()
    for i in range(0, len(data), 16):
        out += _encrypt_block(data[i:i+16], k)
    return bytes(out)


def decrypt_sbox_xor(cipher: bytes, k: int) -> str:
    """
    AES-like toy decryption on 16-byte blocks.

    Args:
        cipher (bytes): Ciphertext bytes (multiple of 16).
        k (int): Byte key 0..255.
    Returns:
        str: Decrypted plaintext.
    """
    if len(cipher) % 16 != 0:
        raise ValueError("Ciphertext length must be multiple of 16")

    out = bytearray()
    for i in range(0, len(cipher), 16):
        out += _decrypt_block(cipher[i:i+16], k)

    out = _unpad16_zeros(bytes(out))
    return out.decode("latin-1")

if __name__ == "__main__":
    secret_k = 0x3A
    pt = "hello AES-like toy!"
    ct = encrypt_sbox_xor(pt, secret_k)
    rt = decrypt_sbox_xor(ct, secret_k)

    print("k:", secret_k, "(hex:", hex(secret_k), ")")
    print("cipher len:", len(ct))
    print("round-trip ok?", rt == pt)
    print("decrypted:", rt)