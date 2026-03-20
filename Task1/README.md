### Task I: Quantum Computing Part 

In this section, I will present the development of the first task. This task requires the following:

1. Implement a simple quantum operation with Cirq or Pennylane:
    - a) With 5 qubits 
    - b) Apply Hadamard operation on every qubit 
    - c) Apply CNOT operation on (0, 1), (1,2), (2,3), (3,4) 
    - d) SWAP (0, 4) 
    - e) Rotate X with $\pi/2$ on any qubit 
    - f) Plot the circuit 

2. Implement a second circuit with a framework of your choice:
    - a) Apply a Hadamard gate to the first qubit
    - b) Rotate the second qubit by $\pi/3$ around X
    - c) Apply Hadamard gate to the third and fourth qubit
    - d) Perform a swap test between the states of the first and second qubit $|q_1 q_2\rangle$ and the third and fourth qubit $|q_3 q_4\rangle$


### 1. Implement a simple quantum operation with Cirq or Pennylane

```python
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import cirq
import pandas as pd

