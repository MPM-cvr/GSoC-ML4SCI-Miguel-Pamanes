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

For this task, I chose to use PennyLane instead of Cirq.

First, I import the necessary packages:

```python
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

Then, we create the circuit with PennyLane. 

First, we define a device with 5 qubits. After that, we construct the quantum node applying the Hadamard gates, the CNOTs, the SWAP, and the RX rotation as requested:

```
dev = qml.device('default.qubit', wires = 5, shots = 10000) #Para fines educativos probamos solo 10,000


#Creamos Nuestro Circuito conforme los lineamientos que se piden
@qml.qnode(dev)
def circuito():
    for i in range(5):
        qml.Hadamard(wires=i)

    qml.Barrier()       #Solo por estética

    qml.CNOT(wires = [0, 1])
    qml.CNOT(wires = [1, 2])
    qml.CNOT(wires = [2, 3])
    qml.CNOT(wires = [3, 4])

    qml.Barrier()       #Solo por estética

    qml.SWAP(wires = [0, 4])
    qml.RX(np.pi/2, wires=0)

    return qml.probs(wires=range(5)) , qml.sample(wires=range(5))

#Calculamos las probabilidades y las muestras aqui muestras se refiere a los posibles estados 
probabilidades, muestras = circuito()

```
We obtain the following results:

Muestras:
[[1 0 0 1 0]
 [1 0 1 0 1]
 [0 1 1 1 1]
 [0 1 0 1 0]
 [1 1 1 1 1]
 [1 0 0 0 1]
 [0 0 1 1 1]
 [0 1 0 0 0]
 [0 0 0 0 1]
 [1 0 0 0 0]
 [0 0 1 1 1]
 [0 0 1 0 1]
 [1 1 0 0 1]
 [1 0 1 0 1]
 [1 0 0 1 1]
 [0 1 0 1 0]
 [1 1 1 0 1]
 [1 0 1 1 1]
 [1 1 0 1 0]
 [0 1 1 0 0]
 [1 1 0 0 0]
 [1 0 0 1 1]
 [1 1 0 0 1]
 [1 0 1 0 0]
 [1 0 0 0 0]
 [0 0 1 1 0]
 [1 0 1 1 1]
 [1 1 0 0 0]
 [1 1 1 0 1]
 [1 1 1 1 0]
 [1 1 0 1 1]
 [0 1 1 0 0]]

Then we check the probability associated to each state 



