from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeVigoV2

# Constants
shots = 1024
marked_key = 5
ks = list(range(1, 26))

# Backends
backend = AerSimulator()
fake = FakeVigoV2()
ideal_backend = AerSimulator(method="statevector")
noise_model = NoiseModel.from_backend(fake)
noisy_backend = AerSimulator(noise_model=noise_model,basis_gates=noise_model.basis_gates)
