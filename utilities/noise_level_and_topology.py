#------------------------------------------------------------------------------
# noise_level_and_topology.py
#
# This module provides functions for analyzing the performance of Grover-walk
# circuits under noise and restricted hardware topologies. It supports sweeping
# success probabilities across different step counts, detecting amplification
# peaks, estimating damping factors, and simulating the effect of noise models
# and qubit connectivity constraints.
#
# Imports:
# - AerSimulator, NoiseModel, depolarizing_error, ReadoutError: Qiskit Aer tools
#   for noise simulation.
# - transpile, marginal_counts: Qiskit utilities for circuit compilation and
#   measurement analysis.
# - coined_grover_walk_search: Circuit generator from utilities.grover_walk.
# - numpy: Numerical computations.
#
# Functions included:
# - success_prob_for_steps(marked_key, steps, shots, backend, n_key_qubits=5):
#   Runs the walk circuit for a fixed number of steps and returns success
#   probability.
# - sweep_success(marked_key, steps_list, shots, backend): Computes success
#   probabilities over multiple step counts.
# - first_two_peaks(xs, ys): Finds the first two local maxima in a curve.
# - estimate_kappa(peaks): Estimates damping constant from successive peaks.
# - simple_noise_model(n_qubits, p1, p2, p_meas): Builds a uniform depolarizing
#   and readout noise model.
# - make_backend_with_noise(n_qubits, p1, p2, p_meas, coupling_map=None):
#   Creates a simulator backend with the given noise model and optional topology.
# - analyze_backend(...): Runs performance analysis on a backend, optionally
#   considering coupling maps and initial layouts.
#
# These utilities are useful for quantifying how both noise and connectivity
# impact Grover/walk algorithms in NISQ settings, where limited coherence,
# gate errors, and restricted topologies constrain amplification.
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import depolarizing_error, ReadoutError
from qiskit import transpile
from qiskit.result import marginal_counts
from utilities.grover_walk import coined_grover_walk_search
import numpy as np


def success_prob_for_steps(marked_key, steps, shots, backend, n_key_qubits=5):
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


def simple_noise_model(n_qubits: int, p1=1e-3, p2=5e-3, p_meas=2e-2):
    """Builds a simple uniform noise model with depolarizing and readout errors.

    Args:
        n_qubits (int): Number of qubits.
        p1 (float, optional): Single-qubit depolarizing error rate. Defaults to 1e-3.
        p2 (float, optional): Two-qubit depolarizing error rate. Defaults to 5e-3.
        p_meas (float, optional): Readout error probability. Defaults to 2e-2.

    Returns:
        qiskit_aer.noise.NoiseModel: Constructed noise model.
    """
    nm = NoiseModel()
    # gate errors
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)
    nm.add_all_qubit_quantum_error(err1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
    nm.add_all_qubit_quantum_error(err2, ['cx'])
    # readout errors
    ro = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    for q in range(n_qubits):
        nm.add_readout_error(ro, [q])
    return nm


def make_backend_with_noise(n_qubits: int, p1, p2, p_meas, coupling_map=None):
    """Creates a noisy simulator backend with optional topology constraints.

    Args:
        n_qubits (int): Number of qubits.
        p1 (float): Single-qubit depolarizing error rate.
        p2 (float): Two-qubit depolarizing error rate.
        p_meas (float): Readout error probability.
        coupling_map (list, optional): Connectivity map for qubits.

    Returns:
        qiskit_aer.AerSimulator: Simulator backend with noise and connectivity.
    """
    nm = simple_noise_model(n_qubits, p1, p2, p_meas)
    return AerSimulator(noise_model=nm,
                        basis_gates=nm.basis_gates,
                        coupling_map=coupling_map)


def analyze_backend(backend, label, marked_key=5, steps_range=range(1, 26),
                    shots=1024, N=26, topology=None, initial_layout=None):
    """Analyzes Grover-walk performance on a given backend.

    Args:
        backend (qiskit.providers.Backend): Backend used for simulation/execution.
        label (str): Label used for printing results.
        marked_key (int, optional): Key/state to be searched. Defaults to 5.
        steps_range (iterable, optional): Range of step counts to test. Defaults to 1–25.
        shots (int, optional): Number of measurement shots per run. Defaults to 1024.
        N (int, optional): Size of the search space. Defaults to 26.
        topology (bool, optional): If True, analyze with connectivity constraints.
        initial_layout (list, optional): Initial layout for transpilation.

    Returns:
        tuple: (xs, ys, peaks, kappa, k_opt, p_opt) containing step values, 
        success probabilities, detected peaks, estimated damping, optimal step, 
        and optimal success probability.
    """
    xs = np.array(list(steps_range))
    if topology:
        ys = []
        for k in xs:
            qc = coined_grover_walk_search(5, format(marked_key, "05b"), steps=k)
            tqc = transpile(qc, backend, initial_layout=initial_layout, optimization_level=3)
            job = backend.run(tqc, shots=shots)
            result = job.result()
            pos = marginal_counts(result, indices=list(range(5, 10))).get_counts()
            ys.append(pos.get(format(marked_key, "05b"), 0) / shots)
        ys = np.array(ys)
    else:
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
