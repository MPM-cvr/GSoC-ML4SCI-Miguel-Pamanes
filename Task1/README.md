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

Then, I create the circuit with PennyLane. 

First, I define a device with 5 qubits. After that, I construct the quantum node applying the Hadamard gates, the CNOTs, the SWAP, and the RX rotation as requested:

```python
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
I obtain the following results:
```Python
#Revisamos las muestras
print("Muestras:")
print(muestras[:32])
```

```text
Muestras:
[1 0 0 1 0]
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
[0 1 1 0 0]
```

Then, I check the probability associated with each state:

```python
for i, p in enumerate(probabilidades):
    estado_binario = format(i, '05b') 
    print(f"Estado |{estado_binario}> : {p:.4f} ({p*100:.2f}%)")
```

Which gave me the following results:

```text
Estado |00000> : 0.0312 (3.12%)
Estado |00001> : 0.0312 (3.12%)
Estado |00010> : 0.0312 (3.12%)
Estado |00011> : 0.0312 (3.12%)
Estado |00100> : 0.0312 (3.12%)
Estado |00101> : 0.0312 (3.12%)
Estado |00110> : 0.0312 (3.12%)
Estado |00111> : 0.0312 (3.12%)
Estado |01000> : 0.0312 (3.12%)
Estado |01001> : 0.0312 (3.12%)
Estado |01010> : 0.0312 (3.12%)
Estado |01011> : 0.0312 (3.12%)
Estado |01100> : 0.0312 (3.12%)
Estado |01101> : 0.0312 (3.12%)
Estado |01110> : 0.0312 (3.12%)
Estado |01111> : 0.0312 (3.12%)
Estado |10000> : 0.0312 (3.12%)
Estado |10001> : 0.0312 (3.12%)
Estado |10010> : 0.0312 (3.12%)
Estado |10011> : 0.0312 (3.12%)
Estado |10100> : 0.0312 (3.12%)
Estado |10101> : 0.0312 (3.12%)
Estado |10110> : 0.0312 (3.12%)
Estado |10111> : 0.0312 (3.12%)
Estado |11000> : 0.0312 (3.12%)
Estado |11001> : 0.0312 (3.12%)
Estado |11010> : 0.0312 (3.12%)
Estado |11011> : 0.0312 (3.12%)
Estado |11100> : 0.0312 (3.12%)
Estado |11101> : 0.0312 (3.12%)
Estado |11110> : 0.0312 (3.12%)
Estado |11111> : 0.0312 (3.12%)
```

Last but not least, we plot the circuit:

```python

#Graficamos el circuto

fig, ax = qml.draw_mpl(circuito, scale=0.8)()
fig.suptitle("Circuito", fontsize=14)
plt.show()



```
<img width="1220" height="619" alt="image" src="https://github.com/user-attachments/assets/f384ae7c-5aee-4bfa-879b-47c453daf6a7" />


### 2. Implement a second circuit with a framework of your choice:

Then I create the second circuit, also in Pennylane

```Python

#Creamos el segundo circuito
dev_2 = qml.device('default.qubit', wires = 5)

@qml.qnode(dev_2)
def circuito_2():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=3)
    qml.Hadamard(wires=4)

    qml.RX(np.pi/3, wires=2)

    qml.CSWAP(wires=[0, 1, 3])
    qml.CSWAP(wires=[0, 2, 4])

    qml.Hadamard(wires=0)

    return qml.probs(wires=0)
```

Now I check the probabilities associated with each state:

```Python

probas = circuito_2()

for i, p in enumerate(probas):
    estado_binario = format(i, '05b') 
    print(f"Estado |{estado_binario}> : {p:.4f} ({p*100:.2f}%)")

```
That gave me the following results:

```text
Estado |00000> : 0.7500 (75.00%)
Estado |00001> : 0.2500 (25.00%)
```











