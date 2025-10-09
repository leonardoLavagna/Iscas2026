from typing import Callable, Dict, List, Tuple, Optional
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector
from utilities.oracles.aes_coherent_oracle import (
    AES_SBOX,
    oracle_isolated,
    oracle_nonisolated,
    channel_unitary_identity,
    channel_unitary_tag_phase,
    u_encrypt_Ek_xor_sbox, 
)


def compute_c_const(m_val: int, k_star: int) -> int:
    """c = S(m XOR k*), using the same 8-bit LSB-first convention as the module."""
    return AES_SBOX[(m_val ^ k_star) & 0xFF]


def bitstr_lsb_first(x: int, n: int) -> str:
    """LSB-first bitstring (q[0] is least significant)."""
    return ''.join('1' if (x >> i) & 1 else '0' for i in range(n))


def success_prob_from_state(sv: Statevector, key_reg: QuantumRegister, k_star: int) -> float:
    """
    Marginalize to the key register and read probability for k_star.
    Robust to endianness by trying both orders.
    """
    pmap = sv.probabilities_dict(qargs=list(key_reg))
    bs = bitstr_lsb_first(k_star, len(key_reg))
    if bs in pmap:
        return float(pmap[bs])
    rb = bs[::-1]
    if rb in pmap:
        return float(pmap[rb])
    # As a last resort, look for integer key (some Qiskit versions use int keys)
    if k_star in pmap: 
        return float(pmap[k_star])
    raise KeyError("Could not locate key bitstring in probabilities_dict; check qubit ordering.")


def diffusion_on_keys(qc: QuantumCircuit, key_reg: QuantumRegister):
    """Standard Grover diffusion on the key register only."""
    for q in key_reg: qc.h(q); qc.x(q)
    # multi-controlled Z implemented via H on last qubit + MCX + H
    last = key_reg[-1]
    qc.h(last)
    qc.mcx(list(key_reg[:-1]), last)
    qc.h(last)
    for q in key_reg: qc.x(q); qc.h(q)
        

def build_search_circuit(
    *,
    m_val: int,
    k_init_superposition: bool = True,
    nbits: int = 8,
    steps: int,
    oracle_kind: str,                      # 'isolated' | 'nonisolated'
    c_const: Optional[int] = None,
    channel: str = "identity",             # 'identity' | 'tag_phase'
    n_env: int = 1,
    n_anc: int = 6,
) -> Tuple[QuantumCircuit, Dict[str, QuantumRegister]]:
    """
    Build a full Grover circuit with pluggable isolated/non-isolated oracles.
    Returns (circuit, registers).
    """
    # Registers
    m_reg   = QuantumRegister(nbits, "m")
    k_reg   = QuantumRegister(nbits, "k")
    c_reg   = QuantumRegister(nbits, "c")
    phase   = QuantumRegister(1,      "phase")
    env_reg = QuantumRegister(n_env,  "env") if (oracle_kind == "nonisolated" and n_env > 0) else None
    anc     = AncillaRegister(max(0, n_anc), "anc") if n_anc > 0 else None
    regs = {"m": m_reg, "k": k_reg, "c": c_reg, "phase": phase}
    if env_reg is not None: regs["env"] = env_reg
    if anc is not None:     regs["anc"] = anc
    # Circuit
    qregs = [m_reg, k_reg, c_reg, phase]
    if env_reg is not None: qregs.append(env_reg)
    if anc is not None:     qregs.append(anc)
    qc = QuantumCircuit(*qregs, name=f"Grover_{oracle_kind}")
    # Initialize message to |m_val>, others to |0...0>
    for i in range(nbits):
        if (m_val >> i) & 1: qc.x(m_reg[i])
    # Prepare key superposition (unless you want to test fixed key behavior)
    if k_init_superposition:
        for q in k_reg: qc.h(q)
    # Prepare phase |-> (for phase kickback)
    qc.x(phase[0]); qc.h(phase[0])
    # Determine target ciphertext
    if c_const is None:
        # (for evaluation you know k*, but in a real run the oracle hides it)
        raise ValueError("c_const must be provided (or compute it from (m,k*) with compute_c_const).")
    # Oracle selector
    if oracle_kind == "isolated":
        def apply_oracle():
            qc.append(
                oracle_isolated(m_reg, k_reg, c_reg, phase, c_const=c_const, anc_reg=anc).to_gate(),
                list(m_reg)+list(k_reg)+list(c_reg)+list(phase)+(list(anc) if anc else [])
            )
    elif oracle_kind == "nonisolated":
        chan = channel_unitary_identity if channel == "identity" else channel_unitary_tag_phase
        def apply_oracle():
            qc.append(
                oracle_nonisolated(m_reg, k_reg, c_reg, phase, c_const=c_const,
                                   channel_unitary=chan, env_reg=env_reg, anc_reg=anc).to_gate(),
                list(m_reg)+list(k_reg)+list(c_reg)+list(phase)
                + (list(env_reg) if env_reg else []) + (list(anc) if anc else [])
            )
    else:
        raise ValueError("oracle_kind must be 'isolated' or 'nonisolated'")
    # Grover iterations
    for _ in range(steps):
        apply_oracle()
        diffusion_on_keys(qc, k_reg)
    return qc, regs


def sweep_grover_steps(
    *,
    m_val: int,
    k_star: int,
    steps_range: range,
    oracle_kind: str,              # 'isolated' | 'nonisolated'
    channel: str = "identity",
    nbits: int = 8,
    n_env: int = 1,
    n_anc: int = 6,
) -> List[Dict]:
    """
    Build & simulate for each t in steps_range. Returns a list of dict rows:
    {'steps': t, 'P_succ': ..., 'oracle': ..., 'channel': ...}
    """
    c_const = compute_c_const(m_val, k_star)
    rows: List[Dict] = []
    for t in steps_range:
        qc, regs = build_search_circuit(
            m_val=m_val, nbits=nbits, steps=t, oracle_kind=oracle_kind,
            c_const=c_const, channel=channel, n_env=n_env, n_anc=n_anc
        )
        sv = Statevector.from_instruction(qc)
        p = success_prob_from_state(sv, regs["k"], k_star)
        rows.append({
            "steps": t,
            "P_succ": p,
            "oracle": oracle_kind,
            "channel": channel,
        })
    return rows
