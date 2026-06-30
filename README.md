# Iscas2026

Implementations extending the [project presented at Iscas 2025](https://github.com/leonardoLavagna/Iscas2025) for [Iscas 2026](https://2026.ieee-iscas.org/).

## What's in here?
Here you can find the code we used in the project.
* `utilities` contains functions to carry out random walks, Grover searches and frequency attacks on the Caesar's cipher and the AES cipher in a unified (NISQ) framework.
* `notebooks` contains the script we used to get the main results in the project. There are 4 notebooks: the first two are about the unification of (NISQ) cryptographic attacks under the Grover search and quantum walk models, the last two are about time-space tradeoffs in symmetric (NISQ) protocols with the presence of quantum channels.
* `config.py` is a configuration file used to specify some settings (e.g. the noise model).
* `requirements.txt` contains the requirements (install the file before using the code in this repository).
* `LICENSE.txt`: MIT License.

## Use this repository
If you want to use the code in this repository in your projects, please cite explicitely our work, and
* Clone the repository with `git clone https://github.com/leonardoLavagna/Iscas2026`
* Install the requirements with `pip install -r requirements.txt`

## Contributing
We welcome contributions to enhance the functionality and performance of the models. Please submit pull requests or open issues for any improvements or bug fixes.

## License
This project is licensed under the MIT License.

## Citation
Cite this repository or one of the associated papers
```
@INPROCEEDINGS{11562366,
  author={Lavagna, Leonardo and Vittori, Giacomo and Rosato, Antonello and Panella, Massimo},
  booktitle={2026 IEEE International Symposium on Circuits and Systems (ISCAS)}, 
  title={A Unified Quantum Circuit Framework for Grover Search and Quantum Walks in NISQ Cryptanalysis}, 
  year={2026},
  volume={},
  number={},
  pages={837-841},
  keywords={Legged locomotion;Circuits;Noise;Modeling;Printing;Quantum circuit;Ciphers;Probability;Qubit;Registers},
  doi={10.1109/ISCAS66217.2026.11562366}}
```

or 

```
@INPROCEEDINGS{11562218,
  author={Lavagna, Leonardo and Vittori, Giacomo and Rosato, Antonello and Panella, Massimo},
  booktitle={2026 IEEE International Symposium on Circuits and Systems (ISCAS)}, 
  title={Additive Resource Scaling in Quantum Circuits for Search in Cryptanalysis}, 
  year={2026},
  volume={},
  number={},
  pages={2758-2762},
  keywords={Modeling;Printing;Timing;Probability;International trade;Qubit;Registers;Algorithms;Circuits;Quantum channels},
  doi={10.1109/ISCAS66217.2026.11562218}}
```

