# oracles/aes_coherent_oracle.py

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, AncillaRegister
from qiskit.circuit.library import MCXGate

# AES S-box (FIPS-197, Table 4.1). LSB-first bit convention throughout.
AES_SBOX = [
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

def xor_register(target: QuantumRegister, value: int) -> QuantumCircuit:
    """Apply X to target[j] where value bit j is 1 (LSB-first)."""
    qc = QuantumCircuit(target, name="XOR_CONST")
    for j in range(len(target)):
        if (value >> j) & 1:
            qc.x(target[j])
    return qc

def u_sbox_xor_on_output(x_reg: QuantumRegister, y_reg: QuantumRegister,
                         anc_reg: AncillaRegister | None = None,
                         mcx_mode: str = "v-chain") -> QuantumCircuit:
    """
    Append y ^= SBox(x) using a 256-case selector on x and toggling y-bits accordingly.
    Uses multi-controlled X with optional ancillas (recommended).
    """
    n = len(x_reg); m = len(y_reg)
    assert n == 8 and m == 8, "Expect 8-bit S-box."
    sbox = AES_SBOX
    qc = QuantumCircuit(x_reg, y_reg, *( [anc_reg] if anc_reg else [] ), name="U_SBOX_XOR")

    for x0, val in enumerate(sbox):
        # Configure controls to fire when x == x0
        bits = [(x0 >> i) & 1 for i in range(n)]  # LSB-first
        # Turn 0-controls into 1-controls via X trick
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(x_reg[i])
        # MCX target each y[j] where bit is 1
        for j in range(m):
            if (val >> j) & 1:
                if anc_reg is not None and mcx_mode == "v-chain":
                    qc.mcx(list(x_reg), y_reg[j], ancilla_qubits=list(anc_reg), mode="v-chain")
                else:
                    qc.mcx(list(x_reg), y_reg[j])  # may be slower/unsupported for >4 ctrls
        # Undo the X trick
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(x_reg[i])
    return qc

def u_encrypt_Ek_xor_sbox(m_reg: QuantumRegister, k_reg: QuantumRegister,
                          c_reg: QuantumRegister,
                          anc_reg: AncillaRegister | None = None) -> QuantumCircuit:
    """
    Compute c ^= S(m XOR k). Restores m, leaves k untouched.
    """
    qc = QuantumCircuit(m_reg, k_reg, c_reg, *( [anc_reg] if anc_reg else [] ), name="U_E")
    # temp: m ^= k
    for i in range(len(m_reg)):
        qc.cx(k_reg[i], m_reg[i])
    # c ^= S(m)
    qc.append(
        u_sbox_xor_on_output(m_reg, c_reg, anc_reg=anc_reg).to_gate(),
        list(m_reg) + list(c_reg) + (list(anc_reg) if anc_reg else [])
    )
    # uncompute temp
    for i in range(len(m_reg)):
        qc.cx(k_reg[i], m_reg[i])
    return qc

# --- Oracles ---

def _mark_if_equal_zero(qc: QuantumCircuit, reg_zero: QuantumRegister, phase_qubit):
    """
    Flip phase (via X on |-> target) iff reg_zero == 0^n.
    Implements: X on all qubits -> MCX -> X on all qubits.
    """
    for q in reg_zero: qc.x(q)
    qc.mcx(list(reg_zero), phase_qubit)
    for q in reg_zero: qc.x(q)

def oracle_isolated(m_reg: QuantumRegister, k_reg: QuantumRegister, c_reg: QuantumRegister,
                    phase_reg: QuantumRegister,  # must be size 1
                    c_const: int,
                    anc_reg: AncillaRegister | None = None) -> QuantumCircuit:
    """
    Clean phase oracle: compute c ^= S(m^k), compare to c_const, phase-flip if equal,
    then UNCOMPUTE (isolation).
    Assumes phase_reg[0] is prepared in |-> at call time.
    """
    assert len(phase_reg) == 1, "phase_reg must have size 1"
    qc = QuantumCircuit(m_reg, k_reg, c_reg, phase_reg, *( [anc_reg] if anc_reg else [] ), name="O_iso")
    # compute c ^= S(m^k)
    qc.append(
        u_encrypt_Ek_xor_sbox(m_reg, k_reg, c_reg, anc_reg=anc_reg).to_gate(),
        list(m_reg)+list(k_reg)+list(c_reg)+(list(anc_reg) if anc_reg else [])
    )
    # compare to constant (all-zero test after XOR)
    qc.append(xor_register(c_reg, c_const).to_gate(), list(c_reg))
    _mark_if_equal_zero(qc, c_reg, phase_reg[0])
    qc.append(xor_register(c_reg, c_const).to_gate(), list(c_reg))
    # UNCOMPUTE to isolate
    qc.append(
        u_encrypt_Ek_xor_sbox(m_reg, k_reg, c_reg, anc_reg=anc_reg).to_gate().inverse(),
        list(m_reg)+list(k_reg)+list(c_reg)+(list(anc_reg) if anc_reg else [])
    )
    return qc

def channel_unitary_identity(ciph_reg: QuantumRegister,
                             env_reg: QuantumRegister | None = None,
                             k_reg: QuantumRegister | None = None) -> QuantumCircuit:
    """Identity channel placeholder."""
    return QuantumCircuit(ciph_reg, *( [env_reg] if env_reg else [] ), *( [k_reg] if k_reg else [] ), name="CHAN_ID")

def channel_unitary_tag_phase(ciph_reg: QuantumRegister,
                              env_reg: QuantumRegister,
                              k_reg: QuantumRegister | None = None) -> QuantumCircuit:
    """
    Toy non-isolated 'channel': tag environment with the PARITY of the ciphertext
    (keeps dependence on ciphertext, not key).
    """
    qc = QuantumCircuit(ciph_reg, env_reg, name="CHAN_TAG")
    # Parity of ciphertext into env[0]
    for q in ciph_reg:
        qc.cx(q, env_reg[0])
    return qc

def oracle_nonisolated(m_reg: QuantumRegister, k_reg: QuantumRegister, c_reg: QuantumRegister,
                       phase_reg: QuantumRegister,  # size 1
                       c_const: int,
                       channel_unitary=channel_unitary_identity,
                       env_reg: QuantumRegister | None = None,
                       anc_reg: AncillaRegister | None = None) -> QuantumCircuit:
    """
    Phase oracle that LEAVES ciphertext register populated (non-isolated) and
    optionally applies a reversible 'channel' interacting with env_reg.
    Assumes phase_reg[0] is prepared in |->.
    """
    assert len(phase_reg) == 1, "phase_reg must have size 1"
    qc = QuantumCircuit(m_reg, k_reg, c_reg, phase_reg, *( [env_reg] if env_reg else [] ),
                        *( [anc_reg] if anc_reg else [] ), name="O_niso")

    # compute c ^= S(m^k)
    qc.append(
        u_encrypt_Ek_xor_sbox(m_reg, k_reg, c_reg, anc_reg=anc_reg).to_gate(),
        list(m_reg)+list(k_reg)+list(c_reg)+(list(anc_reg) if anc_reg else [])
    )
    # compare & phase flip
    qc.append(xor_register(c_reg, c_const).to_gate(), list(c_reg))
    _mark_if_equal_zero(qc, c_reg, phase_reg[0])
    qc.append(xor_register(c_reg, c_const).to_gate(), list(c_reg))

    # DO NOT uncompute: retain c_reg as quantum data
    # Optional reversible 'channel' on (ciphertext, env, [key?]) — pass explicit qubit list
    chan_circ = (channel_unitary(c_reg, env_reg, k_reg) if env_reg is not None
                 else channel_unitary(c_reg))
    # Determine expected order from the channel circuit signature
    if chan_circ.num_qubits > 0:
        if env_reg is not None and 'CHAN_TAG' in chan_circ.name:
            # expects [c_reg..., env_reg...]
            qc.append(chan_circ.to_gate(), list(c_reg) + list(env_reg))
        else:
            # default: just ciphertext (and possibly env/key if the channel uses them)
            # Try to match by name length heuristics; safest is explicit:
            qubits_for_channel = []
            # include ciph first
            qubits_for_channel += list(c_reg)
            # include env if present and channel declares it
            if env_reg is not None and env_reg.size > 0 and chan_circ.num_qubits >= len(c_reg) + len(env_reg):
                qubits_for_channel += list(env_reg)
            # include key if channel was built with it
            if chan_circ.num_qubits > len(qubits_for_channel):
                qubits_for_channel += list(k_reg)[:chan_circ.num_qubits - len(qubits_for_channel)]
            qc.append(chan_circ.to_gate(), qubits_for_channel)

    return qc
