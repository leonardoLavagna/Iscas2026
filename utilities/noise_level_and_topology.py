#------------------------------------------------------------------------------
# noise_level_and_topology.py
#
# This module provides functions for analyzing the performance of Grover-walk
# circuits under noise. It includes utilities to sweep success probabilities
# across different step counts, detect amplification peaks, estimate damping
# factors, and summarize backend performance with respect to optimal query
# counts and noise effects.
#
# Imports:
# - depolarizing_error, ReadoutError: Qiskit Aer noise models.
#
# Functions included:
# - success_prob_for_steps(marked_key, steps, shots, backend, n_key_qubits=5):
#   Runs the walk circuit for a fixed number of steps and returns success
#   probability.
# - sweep_success(marked_key, steps_list, shots, backend): Computes success
#   probability over multiple step counts.
# - first_two_peaks(xs, ys): Finds the first two local maxima in a curve.
# - estimate_kappa(peaks): Estimates damping constant from successive peaks.
# - analyze_backend(backend, label, ...): Runs analysis on a backend and prints
#   performance metrics.
#
# These utilities are useful for quantifying how noise impacts Grover/walk
# algorithms, especially in NISQ settings where coherence and gate errors limit
# achievable amplification.
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------

from qiskit_aer.noise import depolarizing_error, ReadoutError
from qiskit import transpile
from qiskit.result import marginal_counts
from utilities.grover_walk import coined_grover_walk_search
import numpy as np


def success_prob_for_steps(marked_key, steps, shots,
                           backend, n_key_qubits= 5):
    """Runs the Grover-walk circuit and returns success probability.

    Args:
        marked_key (int): Target key/state to be searched.
        steps (int): Number of walk steps.
        shots (int): Number of measurement shots.
        backend (qiskit.providers.Backend): Backend used for simulation/execution.
        n_key_qubits (int, optional): Number of qubits for the key register.
            Defaults to 5.

    Returns:
        float: Success probability of measuring the marked state.
    """
    marked_bits = format(marked_key, f"0{n_key_qubits}b")
    qc = coined_grover_walk_search(n_key_qubits, marked_bits, steps=steps)
    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=shots)
    result = job.result()
    pos = marginal_counts(result, indices=list(range(n_key_qubits, 2 * n_key_qubits))).get_counts()
    return pos.get(marked_bits, 0) / shots


def sweep_success(marked_key: int, steps_list, shots, backend):
    """Computes success probabilities across a range of Grover-walk steps.

    Args:
        marked_key (int): Key/state to be searched.
        steps_list (iterable): List of step counts to evaluate.
        shots (int): Number of measurement shots per run.
        backend (qiskit.providers.Backend): Backend used for simulation/execution.

    Returns:
        numpy.ndarray: Success probabilities for each step count.
    """
    return np.array([success_prob_for_steps(marked_key, k, shots, backend) for k in steps_list])


def first_two_peaks(xs, ys):
    """Finds the first two local maxima in a sequence of values.

    Args:
        xs (list): X-axis values.
        ys (list): Y-axis values.

    Returns:
        list[tuple]: Up to two (x, y) pairs corresponding to peaks.
    """
    peaks = []
    for i in range(1, len(ys) - 1):
        if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1]:
            peaks.append((xs[i], ys[i]))
    return peaks[:2]


def estimate_kappa(peaks):
    """Estimates the damping constant from two successive amplification peaks.

    Args:
        peaks (list[tuple]): List of (step, probability) pairs.

    Returns:
        float: Estimated damping constant, or NaN if not computable.
    """
    if len(peaks) < 2:
        return np.nan
    (k1, p1), (k2, p2) = peaks[0], peaks[1]
    if p2 <= 0 or p1 <= p2:
        return np.nan
    return (k2 - k1) / np.log(p1 / p2)


def analyze_backend(backend, label, marked_key=5, steps_range=range(1, 15), shots=4096, N=26):
    """Analyzes Grover-walk performance on a given backend.

    Args:
        backend (qiskit.providers.Backend): Backend used for simulation/execution.
        label (str): Label used for printing results.
        marked_key (int, optional): Key/state to be searched. Defaults to 5.
        steps_range (iterable, optional): Range of step counts to test. Defaults to 1–14.
        shots (int, optional): Number of measurement shots per run. Defaults to 4096.
        N (int, optional): Size of the search space. Defaults to 26.

    Returns:
        tuple: (xs, ys, peaks, kappa, k_opt, p_opt) containing step values, 
        success probabilities, detected peaks, estimated damping, optimal step, 
        and optimal success probability.
    """
    xs = np.array(list(steps_range))
    ys = sweep_success(marked_key, xs, shots, backend)
    peaks = first_two_peaks(xs, ys)
    kappa = estimate_kappa(peaks)
    k_opt = xs[np.argmax(ys)]
    p_opt = np.max(ys)
    print(f"[{label}] optimal steps = {k_opt},  max success = {p_opt:.3f}")
    if len(peaks) >= 1:
        print(f"[{label}] first peak at k={peaks[0][0]} with p={peaks[0][1]:.3f}")
    if len(peaks) >= 2:
        print(f"[{label}] second peak at k={peaks[1][0]} with p={peaks[1][1]:.3f}")
    print(f"[{label}] estimated damping kappa ≈ {kappa:.2f}\n")
    return xs, ys, peaks, kappa, k_opt, p_opt

