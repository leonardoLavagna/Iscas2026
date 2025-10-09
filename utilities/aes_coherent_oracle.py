from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import XGate
from utilities.aes_sbox import aes_sbox
from qiskit.circuit.library import ZGate



def u_sbox_xor_on_output(x_reg: QuantumRegister, y_reg: QuantumRegister) -> QuantumCircuit:
    """
    Appends y ^= SBox(x). Implementation: for each x0 in {0..255}, 
    multi-control on x==x0 and toggle output bits y[j] where SBox(x0)_j == 1.
    Uses mcx with dirty ancillas (ok for sim; small n only).
    """
    n = len(x_reg); m = len(y_reg)
    assert n == 8 and m == 8, "Expect 8-bit S-box."
    sbox = aes_sbox()  # list of 256 ints
    qc = QuantumCircuit(x_reg, y_reg, name="U_SBOX_XOR")
    # For each pattern x0, condition on x==x0
    for x0, val in enumerate(sbox):
        bits = [(x0 >> i) & 1 for i in range(n)]  # LSB-first
        # Build controls: flip X on |0>-controls to make them |1>-controls
        for i, b in enumerate(bits):
            if b == 0: qc.x(x_reg[i])
        # Apply to each output bit where val has 1
        for j in range(m):
            if ((val >> j) & 1) == 1:
                qc.mcx(list(x_reg), y_reg[j])  # multi-control Toffoli
        # Unflip controls
        for i, b in enumerate(bits):
            if b == 0: qc.x(x_reg[i])
    return qc


def xor_register(target: QuantumRegister, value: int) -> QuantumCircuit:
    """Apply X to target[j] where value bit j is 1."""
    n = len(target)
    qc = QuantumCircuit(target, name="XOR_CONST")
    for j in range(n):
        if (value >> j) & 1: qc.x(target[j])
    return qc


def u_encrypt_Ek_xor_sbox(m_reg, k_reg, c_reg):
    """
    Compute c ^= S(m XOR k). Leaves (m,k) untouched; writes into c_reg.
    """
    qc = QuantumCircuit(m_reg, k_reg, c_reg, name="U_E")
    # temp = m XOR k  (do in-place on m, then uncompute)
    for i in range(len(m_reg)):
        qc.cx(k_reg[i], m_reg[i])
    # y ^= S(m)
    qc.append(u_sbox_xor_on_output(m_reg, c_reg).to_gate(), list(m_reg)+list(c_reg))
    # uncompute temp (restore m)
    for i in range(len(m_reg)):
        qc.cx(k_reg[i], m_reg[i])
    return qc

def u_decrypt_check(m_reg, k_reg, c_reg, flag_reg, c_const: int):
    """
    Phase-marking helper: compute c ^= S(m XOR k), compare to constant c_const,
    flip phase (Z on |flag>) if equal, optionally keep c_reg (non-isolated) or uncompute (isolated).
    """
    pass  # implemented via wrappers below
