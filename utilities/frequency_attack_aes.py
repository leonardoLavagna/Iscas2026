#------------------------------------------------------------------------------
# frequency_attack_aes.py
#
# Classical frequency-analysis attack on simplified AES S-box XOR cipher.
# Provides utilities to repeat the attack and measure success probability
# as a function of the number of trials.
#
# Functions:
# - classical_recover_key(cipher): try all 256 keys, pick χ²-minimizer.
# - run_attack_once(pt, k): encrypt, attack, decrypt, return stats dict.
# - run_attack_trials(pt, k, n_trials, threshold): run multiple trials,
#   compute similarity vs ground truth, and track the number of trials
#   needed to reach a given success probability threshold.
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------

from utilities.simplified_aes import encrypt_sbox_xor, decrypt_sbox_xor
from collections import Counter

ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENG_FREQ = {'E':12.702,'T':9.056,'A':8.167,'O':7.507,'I':6.966,'N':6.749,'S':6.327,'H':6.094,
            'R':5.987,'D':4.253,'L':4.025,'C':2.782,'U':2.758,'M':2.406,'W':2.360,'F':2.228,
            'G':2.015,'Y':1.974,'P':1.929,'B':1.492,'V':0.978,'K':0.772,'J':0.153,'X':0.150,
            'Q':0.095,'Z':0.074}
ENGP = {c: ENG_FREQ[c]/100.0 for c in ALPH}

def chi2_score(text: str) -> float:
    """Chi-square statistic comparing text frequencies vs English."""
    text = "".join(ch for ch in text if ch in ALPH)
    n = len(text)
    if n == 0:
        return float('inf')
    cnt = Counter(text)
    s = 0.0
    for c in ALPH:
        obs = cnt.get(c,0)
        exp = ENGP[c]*n
        if exp > 0:
            s += (obs-exp)**2/exp
    return s

def classical_recover_key(cipher: bytes) -> int:
    """Recover key by brute-force + frequency χ² score."""
    best_k, best_s = None, float('inf')
    for k in range(256):
        cand = decrypt_sbox_xor(cipher, k)
        s = chi2_score(cand)
        if s < best_s:
            best_s, best_k = s, k
    return best_k

def similarity(s1: str, s2: str) -> float:
    """Return fraction of matching characters between two strings."""
    n = min(len(s1), len(s2))
    if n == 0: return 0.0
    return sum(a==b for a,b in zip(s1,s2)) / n

def run_attack_once(pt: str, k: int):
    """
    Encrypt with AES S-box XOR, run classical frequency attack, return stats.

    Returns:
        dict with fields: true_key, recovered_key, recovered_pt,
                          similarity, success (bool exact match).
    """
    ct = encrypt_sbox_xor(pt, k)
    k_hat = classical_recover_key(ct)
    pt_hat = decrypt_sbox_xor(ct, k_hat)
    sim = similarity(pt, pt_hat)
    return {
        "true_key": k,
        "recovered_key": k_hat,
        "recovered_pt": pt_hat,
        "similarity": sim,
        "success": (pt_hat == pt)
    }

def run_attack_trials(pt: str, k: int, n_trials: int=100, threshold: float=0.9):
    """
    Run the attack multiple times and track empirical success probability.

    Args:
        pt (str): plaintext
        k (int): secret key
        n_trials (int): number of attack repetitions
        threshold (float): success probability target

    Returns:
        results (list of dict): trial results
        steps_to_threshold (int or None): number of trials needed to exceed threshold
    """
    results = []
    successes = 0
    steps_to_threshold = None

    for t in range(1, n_trials+1):
        out = run_attack_once(pt, k)
        results.append(out)
        if out["success"]:
            successes += 1
        p_succ = successes / t
        if steps_to_threshold is None and p_succ >= threshold:
            steps_to_threshold = t

    return results, steps_to_threshold
