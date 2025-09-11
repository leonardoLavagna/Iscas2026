#------------------------------------------------------------------------------
# grover_walk.py
#
# This module implements a unified circuit framework where Grover's algorithm 
# is expressed as a coined quantum walk. It provides building blocks for the 
# Grover diffusion operator, oracle construction, shift operator, and a full 
# coined Grover-walk search routine. An execution helper is also included to 
# run the walk circuit and extract position-register statistics.
#
# Functions included:
# - grover_diffusion_gate(n): Constructs the Grover diffusion operator.
# - phase_oracle_on_register(n_total, pos_start, pos_len, marked_bitstring): 
#   Builds a phase-flip oracle marking the solution state.
# - flipflop_shift_swap(n, coin_start, pos_start): Implements the flip-flop 
#   shift as SWAP operations between coin and position registers.
# - coined_grover_walk_search(n, marked_state, steps=None, measure=True): 
#   Builds the coined Grover-walk search circuit.
# - run_walk_and_get_position_counts(n, marked_state, backend=None, shots=2048): 
#   Executes the walk circuit and returns measurement counts for the position 
#   register.
#
# These functions support experiments in quantum search, cryptanalysis 
# (e.g., Caesar’s cipher key recovery), and analysis of Grover vs. quantum 
# walk formulations in both fault-tolerant and NISQ settings.
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------


from qiskit import QuantumCircuit, transpile
from qiskit.result import marginal_counts
from qiskit.circuit.library import MCXGate, SwapGate
import numpy as np


def grover_diffusion_gate(n):
    """Constructs the Grover diffusion operator on n qubits.

    Args:
        n (int): Number of qubits.

    Returns:
        qiskit.circuit.Gate: Grover diffusion gate.
    """
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
    """Constructs a phase-flip oracle acting on the position register.

    Args:
        n_total (int): Total number of qubits in the circuit.
        pos_start (int): Starting index of the position register.
        pos_len (int): Number of qubits in the position register.
        marked_bitstring (str): Bitstring of the marked state (e.g., '101').

    Returns:
        qiskit.circuit.Gate: Oracle gate marking the specified state.
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
    """Implements the flip-flop shift as a SWAP between coin and position registers.

    Args:
        n (int): Number of qubits in each register.
        coin_start (int): Starting index of the coin register.
        pos_start (int): Starting index of the position register.

    Returns:
        qiskit.circuit.Gate: Shift operator gate.
    """
    qc = QuantumCircuit(2*n)
    for i in range(n):
        qc.append(SwapGate(), [coin_start + i, pos_start + i])
    return qc.to_gate(label="Shift=SWAP")


def coined_grover_walk_search(n, marked_state, steps=None, measure=True):
    """Builds a coined quantum walk circuit for Grover search.

    Args:
        n (int): Number of qubits in coin and position registers (total 2n).
        marked_state (str): Bitstring of the marked state.
        steps (int, optional): Number of walk steps. Defaults to optimal if None.
        measure (bool, optional): Whether to add measurement operations.

    Returns:
        qiskit.QuantumCircuit: The coined Grover-walk search circuit.
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
    """Runs the coined Grover-walk circuit and collects position register counts.

    Args:
        n (int): Number of qubits in coin and position registers (total 2n).
        marked_state (str): Bitstring of the marked state.
        backend (qiskit.providers.Backend, optional): Backend to run on.
        shots (int, optional): Number of measurement shots.

    Returns:
        tuple: (QuantumCircuit, dict) The circuit and position register counts.
    """
    if backend is None:
        backend = Aer.get_backend("qasm_simulator")
    qc = coined_grover_walk_search(n, marked_state)
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=shots)
    result = job.result()
    pos_counts = marginal_counts(result, indices=list(range(n, 2*n))).get_counts()
    return qc, pos_counts
    
