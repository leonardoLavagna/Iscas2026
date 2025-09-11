from qiskit import QuantumCircuit, transpile
from qiskit.result import marginal_counts
from qiskit.circuit.library import MCXGate, SwapGate
import numpy as np


def grover_diffusion_gate(n):
    """Grover diffuser on n qubits (reflection about uniform superposition |s>."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    # multi-controlled Z on the last qubit (about |11...1>)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n-1)
        qc.append(MCXGate(n-1), list(range(n)))
        qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    return qc.to_gate(label="Diff[coin]")


def phase_oracle_on_register(n_total, pos_start, pos_len, marked_bitstring):
    """
    Phase-flip oracle that acts only on the position register and marks the
    solution state.

    Args: 
      n_total (int): total qubits in the circuit
      pos_start (int): starting index of position register in the circuit
      pos_len (int): number of qubits in the position register
      marked_bitstring (str): e.g. '101'

    Returns:

    """
    qc = QuantumCircuit(n_total)
    # Map MSB...LSB to Qiskit's little-endian order within the block
    bits = marked_bitstring[::-1]
    # X on zeros to turn |marked> into |11..1>
    for i, b in enumerate(bits):
        if b == '0':
            qc.x(pos_start + i)
    # Multi-controlled Z on position register
    if pos_len == 1:
        qc.z(pos_start)
    else:
        qc.h(pos_start + pos_len - 1)
        qc.append(MCXGate(pos_len - 1),
                  list(range(pos_start, pos_start + pos_len)))
        qc.h(pos_start + pos_len - 1)
    # Undo X
    for i, b in enumerate(bits):
        if b == '0':
            qc.x(pos_start + i)
    return qc.to_gate(label=f"Oracle|{marked_bitstring}>[pos]")


def flipflop_shift_swap(n, coin_start, pos_start):
    """Flip-flop shift S as a full SWAP between coin and position registers, thus
       coupling the coin and position spaces.
    """
    qc = QuantumCircuit(2*n)
    for i in range(n):
        qc.append(SwapGate(), [coin_start + i, pos_start + i])
    return qc.to_gate(label="Shift=SWAP")


def coined_grover_walk_search(n, marked_state, steps=None, measure=True):
    """
    Coined quantum walk Grover search on 2^n items.
    Registers: [coin (n qubits)] + [pos (n qubits)]
    Walk step: S * (C_coin ⊗ I_pos) * O_pos
    """
    N = 2**n
    n_total = 2*n
    coin_start = 0
    pos_start = n
    qc = QuantumCircuit(n_total, n_total if measure else 0)
    # Init to |s>_coin ⊗ |s>_pos
    qc.h(range(n_total))
    # Gates
    C_coin = grover_diffusion_gate(n)
    O_pos = phase_oracle_on_register(n_total, pos_start, n, marked_state)
    S_swap = flipflop_shift_swap(n, coin_start, pos_start)
    # Number of Grover-like steps
    if steps is None:
        steps = int(np.floor(np.pi/4*np.sqrt(N)))
    for _ in range(steps):
        qc.append(O_pos, range(n_total))
        qc.append(C_coin, range(coin_start, coin_start + n))
        qc.append(S_swap, range(n_total))
    if measure:
        qc.measure(range(n_total), range(n_total))
    return qc


def run_walk_and_get_position_counts(n, marked_state, backend=None, shots=2048):
    if backend is None:
        backend = Aer.get_backend("qasm_simulator")
    qc = coined_grover_walk_search(n, marked_state)
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=shots)
    result = job.result()
    pos_counts = marginal_counts(result, indices=list(range(n, 2*n))).get_counts()
    return qc, pos_counts
