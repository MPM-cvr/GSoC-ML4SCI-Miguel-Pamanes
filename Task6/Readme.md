# Task VI: Quantum representation learning

In this section I will show you my procedure tu solve Task 6. 

Task 6 ask the following

In this task you should implement a simple representation learning scheme based on a contrastive loss:
- Load the MNIST dataset
- Write a function which takes an image and prepares a quantum state. This function should have trainable parameters which we want to learn in order to have good quantum representations
- Create a circuit with which takes two images and embeds both as quantum states with the function you wrote before. Afterwards the circuit should perform a SWAP test between the two states. In the end the measurement should give the fidelity of the quantum states.
- Train the circuit parameters with a contrastive loss: For two MNIST images in the same class the fidelity should be maximized, while for images of different classes the fidelity should be minimized.

### Objective

This code trains a quantum circuit to learn how to compare images of handwritten numbers (MNIST dataset) and determine if they are of the same number or not.

### Code


First, I imported all the necessary packages

```Python

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math
import pennylane as qml
import tensorflow as tf
import random
import torch.optim as optim
```
To better understand the data, I first displayed an individual sample and then a series of original images for comparison.

```Python
transform_original = transforms.Compose([
    transforms.ToTensor() 
])


train_dataset_original = datasets.MNIST(root='./data', 
                                        train=True, 
                                        download=True, 
                                        transform=transform_original)


imagen_original, etiqueta_original = train_dataset_original[0]

plt.figure()
imagen_prueba = imagen_original.squeeze()
plt.imshow(imagen_prueba, cmap=plt.cm.binary)
plt.colorbar
plt.show()


plt.figure(figsize=(10,10))

for i in range(25):
    imagen_original, etiqueta_original = train_dataset_original[i]
    
    plt.subplot(5, 5, i+1)
    
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(imagen_original.squeeze(), cmap=plt.cm.binary)
    plt.title(f"Etiqueta: {etiqueta_original}")

plt.show()


```
This is a single example. <br>
<img width="416" height="413" alt="image" src="https://github.com/user-attachments/assets/ecfad9c3-e100-4c16-8a92-980f3a1525fa" /> <br>

And these are a set of examples.
<img width="794" height="812" alt="image" src="https://github.com/user-attachments/assets/e682fff2-6df0-4ad2-85d7-6fc525da4267" /> <br>


This block acts as a preprocessing pipeline that prepares classical images for quantum integration. First, the original image is downsampled to an 8×8 resolution to manage simulator constraints. This grid is then flattened into a 1D vector of 64 values. Finally, these values undergo amplitude normalization, ensuring the total magnitude is exactly 1 to satisfy the requirements of a quantum state.

```Python


transformar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((8, 8)),   
    transforms.Lambda(lambda x: torch.flatten(x)),
    transforms.Lambda(lambda x: x / (torch.norm(x) + 1e-8))
])



train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               download=True, 
                               transform=transformar)

imagen, etiqueta = train_dataset[0]

#Vamos a probar con un solo dato
print(f"La etiqueta es: {etiqueta}")
print(f"La forma de la imagen es: {imagen.shape}")

```
And for this example this block gave me:
```Text
La etiqueta es: 5
La forma de la imagen es: torch.Size([64])
```
This block configures a 13 qubit quantum circuit to load and compare two images simultaneously. Use 6 qubits for the first image and 6 for the second, translating the list of 64 numbers (the pixels) into quantum states by rotating. During this embedding process, it also incorporates the theta parameters and intertwine the qubits to capture the patterns of the images.


To compare both images, the code uses qubit number 0 (the ancilla) applying an algorithm called Swap Test. This algorithm puts the ancilla in superposition to try to exchange the states of both images at the same time; in the end, this judge qubit is measured.

```Python


num_qubits = 6
total_qubits = 2 * num_qubits + 1 
ancilla = 0 

dev = qml.device("default.qubit", wires=total_qubits)

def quantum_embedding(x, theta, qubits):
    
    L = len(x)
    Q = len(qubits)
    
    for i in range(L):
        q = qubits[i % Q]
        qml.RY(x[i], wires=q)
    
    for i in range(Q):
        qml.RY(theta[i], wires=qubits[i])
    
    for i in range(Q - 1):
        qml.CNOT(wires=[qubits[i], qubits[i+1]])

@qml.qnode(dev, interface="torch")
def quantum_comparator(img_a, img_b, theta):
    
    qubits_a = list(range(1, 1 + num_qubits))
    qubits_b = list(range(1 + num_qubits, 1 + 2 * num_qubits))
    
    quantum_embedding(img_a, theta, qubits_a)
    quantum_embedding(img_b, theta, qubits_b)
    
    qml.Hadamard(wires=ancilla)
    
    for i in range(len(qubits_a)):
        qml.CSWAP(wires=[ancilla, qubits_a[i], qubits_b[i]])
        
    qml.Hadamard(wires=ancilla)
    
    return qml.probs(wires=ancilla)

```


To visualize the circuit, I generate a list of 16 random numbers. These are not actual images from the MNIST dataset or real quantum weights; they are simply synthetic data used to initialize the simulator and render the circuit diagram.

```Python
img_a_prueba = torch.rand(16)
img_b_prueba = torch.rand(16)
theta_prueba = torch.rand(16)

fig, ax = qml.draw_mpl(quantum_comparator)(img_a_prueba, img_b_prueba, theta_prueba)

plt.show()
```


That gave me this image

<img width="1920" height="1420" alt="image" src="https://github.com/user-attachments/assets/1f453d2a-6903-4029-8545-cf9d2671200f" />


Then I realize the training phase of the model. Initially, the trainable parameters of the circuit are defined and the Adam optimization algorithm is configured. The iterative process consists of 200 steps in which an auxiliary generator randomly selects pairs of images from the data set, alternating in a structured way between pairs of the same class and pairs of different classes, guaranteeing balanced sampling.


The circuit evaluates the pair of images and calculates their level of similarity, expressed as quantum fidelity. To guide network learning, a conditional loss function is established: if the images belong to the same class, the function minimizes the error by bringing fidelity closer to 1.0; on the contrary, if they are from different classes, it minimizes the error by forcing fidelity to 0.0.

```Python

theta = torch.rand(6, requires_grad=True)

optimizador = optim.Adam([theta], lr=0.1)

def obtener_par_imagenes(dataset, misma_clase=True):
    idx1, idx2 = random.randint(0, 1000), random.randint(0, 1000)
    img1, label1 = dataset[idx1]
    img2, label2 = dataset[idx2]
    
    while (misma_clase and label1 != label2) or (not misma_clase and label1 == label2):
        idx2 = random.randint(0, 1000)
        img2, label2 = dataset[idx2]
        
    return img1, img2, label1, label2

pasos = 200

for paso in range(pasos):
    optimizador.zero_grad() 
    
    misma_clase = (paso % 2 == 0)
    img_a, img_b, label_a, label_b = obtener_par_imagenes(train_dataset, misma_clase)
    
    probabilidades = quantum_comparator(img_a, img_b, theta)
    
    prob_0 = probabilidades[0]
    fidelidad = 2.0 * prob_0 - 1.0
    fidelidad = torch.clamp(fidelidad, 0.0, 1.0) 
    
    if misma_clase:
        loss = 1.0 - fidelidad
    else:
        loss = fidelidad
        
    loss.backward()
    optimizador.step()
    
    tipo = "Misma" if misma_clase else "Diferente"
    print(f"Paso {paso+1:02d} | Clases: {label_a} y {label_b} ({tipo}) | Fidelidad: {fidelidad.item():.4f} | Loss: {loss.item():.4f}")

print("Los valores finales de theta son:", theta.detach().numpy())

```
The following results were obtained from this stage of the simulation:

```Text

Paso 01 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9986 | Loss: 0.0014
Paso 02 | Clases: 2 y 7 (Diferente) | Fidelidad: 0.9580 | Loss: 0.9580
Paso 03 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9724 | Loss: 0.0276
Paso 04 | Clases: 4 y 0 (Diferente) | Fidelidad: 0.8600 | Loss: 0.8600
Paso 05 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9785 | Loss: 0.0215
Paso 06 | Clases: 7 y 8 (Diferente) | Fidelidad: 0.8378 | Loss: 0.8378
Paso 07 | Clases: 4 y 4 (Misma) | Fidelidad: 0.9616 | Loss: 0.0384
Paso 08 | Clases: 9 y 6 (Diferente) | Fidelidad: 0.9193 | Loss: 0.9193
Paso 09 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9809 | Loss: 0.0191
Paso 10 | Clases: 4 y 8 (Diferente) | Fidelidad: 0.9851 | Loss: 0.9851
Paso 11 | Clases: 6 y 6 (Misma) | Fidelidad: 0.9914 | Loss: 0.0086
Paso 12 | Clases: 7 y 1 (Diferente) | Fidelidad: 0.8900 | Loss: 0.8900
Paso 13 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9563 | Loss: 0.0437
Paso 14 | Clases: 1 y 4 (Diferente) | Fidelidad: 0.9579 | Loss: 0.9579
Paso 15 | Clases: 1 y 1 (Misma) | Fidelidad: 0.8969 | Loss: 0.1031
Paso 16 | Clases: 5 y 8 (Diferente) | Fidelidad: 0.9796 | Loss: 0.9796
Paso 17 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9533 | Loss: 0.0467
Paso 18 | Clases: 3 y 6 (Diferente) | Fidelidad: 0.8919 | Loss: 0.8919
Paso 19 | Clases: 4 y 4 (Misma) | Fidelidad: 0.9810 | Loss: 0.0190
Paso 20 | Clases: 8 y 4 (Diferente) | Fidelidad: 0.9497 | Loss: 0.9497
Paso 21 | Clases: 8 y 8 (Misma) | Fidelidad: 0.9783 | Loss: 0.0217
Paso 22 | Clases: 5 y 4 (Diferente) | Fidelidad: 0.8748 | Loss: 0.8748
Paso 23 | Clases: 2 y 2 (Misma) | Fidelidad: 0.9681 | Loss: 0.0319
Paso 24 | Clases: 2 y 9 (Diferente) | Fidelidad: 0.8895 | Loss: 0.8895
Paso 25 | Clases: 2 y 2 (Misma) | Fidelidad: 0.8964 | Loss: 0.1036
Paso 26 | Clases: 0 y 7 (Diferente) | Fidelidad: 0.8500 | Loss: 0.8500
Paso 27 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9802 | Loss: 0.0198
Paso 28 | Clases: 7 y 4 (Diferente) | Fidelidad: 0.9508 | Loss: 0.9508
Paso 29 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9753 | Loss: 0.0247
Paso 30 | Clases: 6 y 7 (Diferente) | Fidelidad: 0.9821 | Loss: 0.9821
Paso 31 | Clases: 0 y 0 (Misma) | Fidelidad: 0.9669 | Loss: 0.0331
Paso 32 | Clases: 9 y 3 (Diferente) | Fidelidad: 0.9657 | Loss: 0.9657
Paso 33 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9776 | Loss: 0.0224
Paso 34 | Clases: 1 y 3 (Diferente) | Fidelidad: 0.8937 | Loss: 0.8937
Paso 35 | Clases: 9 y 9 (Misma) | Fidelidad: 0.9951 | Loss: 0.0049
Paso 36 | Clases: 5 y 0 (Diferente) | Fidelidad: 0.8357 | Loss: 0.8357
Paso 37 | Clases: 6 y 6 (Misma) | Fidelidad: 0.9739 | Loss: 0.0261
Paso 38 | Clases: 1 y 3 (Diferente) | Fidelidad: 0.9268 | Loss: 0.9268
Paso 39 | Clases: 9 y 9 (Misma) | Fidelidad: 0.9889 | Loss: 0.0111
Paso 40 | Clases: 2 y 4 (Diferente) | Fidelidad: 0.8672 | Loss: 0.8672
Paso 41 | Clases: 6 y 6 (Misma) | Fidelidad: 0.8838 | Loss: 0.1162
Paso 42 | Clases: 2 y 0 (Diferente) | Fidelidad: 0.8768 | Loss: 0.8768
Paso 43 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9273 | Loss: 0.0727
Paso 44 | Clases: 2 y 1 (Diferente) | Fidelidad: 0.9256 | Loss: 0.9256
Paso 45 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9038 | Loss: 0.0962
Paso 46 | Clases: 2 y 5 (Diferente) | Fidelidad: 0.8753 | Loss: 0.8753
Paso 47 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9724 | Loss: 0.0276
Paso 48 | Clases: 1 y 4 (Diferente) | Fidelidad: 0.9117 | Loss: 0.9117
Paso 49 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9546 | Loss: 0.0454
Paso 50 | Clases: 7 y 6 (Diferente) | Fidelidad: 0.9477 | Loss: 0.9477
Paso 51 | Clases: 9 y 9 (Misma) | Fidelidad: 1.0000 | Loss: 0.0000
Paso 52 | Clases: 6 y 2 (Diferente) | Fidelidad: 0.9864 | Loss: 0.9864
Paso 53 | Clases: 4 y 4 (Misma) | Fidelidad: 0.7803 | Loss: 0.2197
Paso 54 | Clases: 7 y 8 (Diferente) | Fidelidad: 0.8105 | Loss: 0.8105
Paso 55 | Clases: 2 y 2 (Misma) | Fidelidad: 0.9247 | Loss: 0.0753
Paso 56 | Clases: 6 y 0 (Diferente) | Fidelidad: 0.9116 | Loss: 0.9116
Paso 57 | Clases: 2 y 2 (Misma) | Fidelidad: 0.9428 | Loss: 0.0572
Paso 58 | Clases: 2 y 3 (Diferente) | Fidelidad: 0.8473 | Loss: 0.8473
Paso 59 | Clases: 9 y 9 (Misma) | Fidelidad: 0.9648 | Loss: 0.0352
Paso 60 | Clases: 9 y 2 (Diferente) | Fidelidad: 0.9367 | Loss: 0.9367
Paso 61 | Clases: 8 y 8 (Misma) | Fidelidad: 0.9539 | Loss: 0.0461
Paso 62 | Clases: 2 y 1 (Diferente) | Fidelidad: 0.9119 | Loss: 0.9119
Paso 63 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9176 | Loss: 0.0824
Paso 64 | Clases: 1 y 2 (Diferente) | Fidelidad: 0.8811 | Loss: 0.8811
Paso 65 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9423 | Loss: 0.0577
Paso 66 | Clases: 2 y 5 (Diferente) | Fidelidad: 0.9307 | Loss: 0.9307
Paso 67 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9865 | Loss: 0.0135
Paso 68 | Clases: 7 y 4 (Diferente) | Fidelidad: 0.9095 | Loss: 0.9095
Paso 69 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9857 | Loss: 0.0143
Paso 70 | Clases: 2 y 6 (Diferente) | Fidelidad: 0.8953 | Loss: 0.8953
Paso 71 | Clases: 9 y 9 (Misma) | Fidelidad: 0.8718 | Loss: 0.1282
Paso 72 | Clases: 0 y 7 (Diferente) | Fidelidad: 0.9774 | Loss: 0.9774
Paso 73 | Clases: 8 y 8 (Misma) | Fidelidad: 0.9278 | Loss: 0.0722
Paso 74 | Clases: 7 y 0 (Diferente) | Fidelidad: 0.9207 | Loss: 0.9207
Paso 75 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9441 | Loss: 0.0559
Paso 76 | Clases: 3 y 6 (Diferente) | Fidelidad: 0.9448 | Loss: 0.9448
Paso 77 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9274 | Loss: 0.0726
Paso 78 | Clases: 5 y 2 (Diferente) | Fidelidad: 0.9593 | Loss: 0.9593
Paso 79 | Clases: 4 y 4 (Misma) | Fidelidad: 0.9365 | Loss: 0.0635
Paso 80 | Clases: 4 y 9 (Diferente) | Fidelidad: 0.9823 | Loss: 0.9823
Paso 81 | Clases: 8 y 8 (Misma) | Fidelidad: 0.9044 | Loss: 0.0956
Paso 82 | Clases: 0 y 8 (Diferente) | Fidelidad: 0.9324 | Loss: 0.9324
Paso 83 | Clases: 6 y 6 (Misma) | Fidelidad: 0.9793 | Loss: 0.0207
Paso 84 | Clases: 9 y 4 (Diferente) | Fidelidad: 0.9870 | Loss: 0.9870
Paso 85 | Clases: 8 y 8 (Misma) | Fidelidad: 0.9728 | Loss: 0.0272
Paso 86 | Clases: 1 y 3 (Diferente) | Fidelidad: 0.8836 | Loss: 0.8836
Paso 87 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9582 | Loss: 0.0418
Paso 88 | Clases: 2 y 7 (Diferente) | Fidelidad: 0.7848 | Loss: 0.7848
Paso 89 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9252 | Loss: 0.0748
Paso 90 | Clases: 3 y 4 (Diferente) | Fidelidad: 0.9761 | Loss: 0.9761
Paso 91 | Clases: 9 y 9 (Misma) | Fidelidad: 0.9748 | Loss: 0.0252
Paso 92 | Clases: 7 y 0 (Diferente) | Fidelidad: 0.9093 | Loss: 0.9093
Paso 93 | Clases: 0 y 0 (Misma) | Fidelidad: 0.9806 | Loss: 0.0194
Paso 94 | Clases: 6 y 0 (Diferente) | Fidelidad: 0.8225 | Loss: 0.8225
Paso 95 | Clases: 2 y 2 (Misma) | Fidelidad: 0.9476 | Loss: 0.0524
Paso 96 | Clases: 8 y 6 (Diferente) | Fidelidad: 0.8342 | Loss: 0.8342
Paso 97 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9899 | Loss: 0.0101
Paso 98 | Clases: 1 y 8 (Diferente) | Fidelidad: 0.8067 | Loss: 0.8067
Paso 99 | Clases: 4 y 4 (Misma) | Fidelidad: 0.9696 | Loss: 0.0304
Paso 100 | Clases: 6 y 0 (Diferente) | Fidelidad: 0.9382 | Loss: 0.9382
Paso 101 | Clases: 8 y 8 (Misma) | Fidelidad: 0.9817 | Loss: 0.0183
Paso 102 | Clases: 7 y 8 (Diferente) | Fidelidad: 0.9686 | Loss: 0.9686
Paso 103 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9002 | Loss: 0.0998
Paso 104 | Clases: 6 y 9 (Diferente) | Fidelidad: 0.8584 | Loss: 0.8584
Paso 105 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9394 | Loss: 0.0606
Paso 106 | Clases: 2 y 6 (Diferente) | Fidelidad: 0.9186 | Loss: 0.9186
Paso 107 | Clases: 0 y 0 (Misma) | Fidelidad: 0.9585 | Loss: 0.0415
Paso 108 | Clases: 8 y 7 (Diferente) | Fidelidad: 0.9427 | Loss: 0.9427
Paso 109 | Clases: 0 y 0 (Misma) | Fidelidad: 0.9885 | Loss: 0.0115
Paso 110 | Clases: 7 y 5 (Diferente) | Fidelidad: 0.9249 | Loss: 0.9249
Paso 111 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9877 | Loss: 0.0123
Paso 112 | Clases: 3 y 2 (Diferente) | Fidelidad: 0.8816 | Loss: 0.8816
Paso 113 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9874 | Loss: 0.0126
Paso 114 | Clases: 5 y 1 (Diferente) | Fidelidad: 0.9297 | Loss: 0.9297
Paso 115 | Clases: 5 y 5 (Misma) | Fidelidad: 0.8902 | Loss: 0.1098
Paso 116 | Clases: 8 y 0 (Diferente) | Fidelidad: 0.8901 | Loss: 0.8901
Paso 117 | Clases: 4 y 4 (Misma) | Fidelidad: 0.9709 | Loss: 0.0291
Paso 118 | Clases: 7 y 1 (Diferente) | Fidelidad: 0.9783 | Loss: 0.9783
Paso 119 | Clases: 4 y 4 (Misma) | Fidelidad: 0.9575 | Loss: 0.0425
Paso 120 | Clases: 6 y 1 (Diferente) | Fidelidad: 0.9787 | Loss: 0.9787
Paso 121 | Clases: 3 y 3 (Misma) | Fidelidad: 0.8656 | Loss: 0.1344
Paso 122 | Clases: 2 y 6 (Diferente) | Fidelidad: 0.9003 | Loss: 0.9003
Paso 123 | Clases: 6 y 6 (Misma) | Fidelidad: 0.9402 | Loss: 0.0598
Paso 124 | Clases: 9 y 6 (Diferente) | Fidelidad: 0.9349 | Loss: 0.9349
Paso 125 | Clases: 8 y 8 (Misma) | Fidelidad: 0.9150 | Loss: 0.0850
Paso 126 | Clases: 1 y 3 (Diferente) | Fidelidad: 0.9243 | Loss: 0.9243
Paso 127 | Clases: 9 y 9 (Misma) | Fidelidad: 0.9527 | Loss: 0.0473
Paso 128 | Clases: 9 y 6 (Diferente) | Fidelidad: 0.9304 | Loss: 0.9304
Paso 129 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9856 | Loss: 0.0144
Paso 130 | Clases: 4 y 2 (Diferente) | Fidelidad: 0.8634 | Loss: 0.8634
Paso 131 | Clases: 4 y 4 (Misma) | Fidelidad: 0.9847 | Loss: 0.0153
Paso 132 | Clases: 4 y 7 (Diferente) | Fidelidad: 0.8623 | Loss: 0.8623
Paso 133 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9275 | Loss: 0.0725
Paso 134 | Clases: 3 y 0 (Diferente) | Fidelidad: 0.9597 | Loss: 0.9597
Paso 135 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9848 | Loss: 0.0152
Paso 136 | Clases: 6 y 9 (Diferente) | Fidelidad: 0.9034 | Loss: 0.9034
Paso 137 | Clases: 9 y 9 (Misma) | Fidelidad: 0.9662 | Loss: 0.0338
Paso 138 | Clases: 7 y 2 (Diferente) | Fidelidad: 0.9710 | Loss: 0.9710
Paso 139 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9898 | Loss: 0.0102
Paso 140 | Clases: 9 y 0 (Diferente) | Fidelidad: 0.8512 | Loss: 0.8512
Paso 141 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9084 | Loss: 0.0916
Paso 142 | Clases: 2 y 7 (Diferente) | Fidelidad: 0.9374 | Loss: 0.9374
Paso 143 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9929 | Loss: 0.0071
Paso 144 | Clases: 8 y 6 (Diferente) | Fidelidad: 0.9543 | Loss: 0.9543
Paso 145 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9384 | Loss: 0.0616
Paso 146 | Clases: 3 y 6 (Diferente) | Fidelidad: 0.8246 | Loss: 0.8246
Paso 147 | Clases: 8 y 8 (Misma) | Fidelidad: 0.9484 | Loss: 0.0516
Paso 148 | Clases: 0 y 3 (Diferente) | Fidelidad: 0.9458 | Loss: 0.9458
Paso 149 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9224 | Loss: 0.0776
Paso 150 | Clases: 6 y 3 (Diferente) | Fidelidad: 0.8008 | Loss: 0.8008
Paso 151 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9753 | Loss: 0.0247
Paso 152 | Clases: 4 y 0 (Diferente) | Fidelidad: 0.9462 | Loss: 0.9462
Paso 153 | Clases: 5 y 5 (Misma) | Fidelidad: 0.8793 | Loss: 0.1207
Paso 154 | Clases: 2 y 8 (Diferente) | Fidelidad: 0.9408 | Loss: 0.9408
Paso 155 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9208 | Loss: 0.0792
Paso 156 | Clases: 8 y 2 (Diferente) | Fidelidad: 0.9465 | Loss: 0.9465
Paso 157 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9691 | Loss: 0.0309
Paso 158 | Clases: 3 y 4 (Diferente) | Fidelidad: 0.9182 | Loss: 0.9182
Paso 159 | Clases: 0 y 0 (Misma) | Fidelidad: 0.8904 | Loss: 0.1096
Paso 160 | Clases: 4 y 7 (Diferente) | Fidelidad: 0.9788 | Loss: 0.9788
Paso 161 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9654 | Loss: 0.0346
Paso 162 | Clases: 1 y 7 (Diferente) | Fidelidad: 0.9279 | Loss: 0.9279
Paso 163 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9567 | Loss: 0.0433
Paso 164 | Clases: 0 y 3 (Diferente) | Fidelidad: 0.9261 | Loss: 0.9261
Paso 165 | Clases: 5 y 5 (Misma) | Fidelidad: 0.9460 | Loss: 0.0540
Paso 166 | Clases: 1 y 3 (Diferente) | Fidelidad: 0.9265 | Loss: 0.9265
Paso 167 | Clases: 5 y 5 (Misma) | Fidelidad: 0.8854 | Loss: 0.1146
Paso 168 | Clases: 5 y 7 (Diferente) | Fidelidad: 0.8285 | Loss: 0.8285
Paso 169 | Clases: 0 y 0 (Misma) | Fidelidad: 0.9840 | Loss: 0.0160
Paso 170 | Clases: 4 y 8 (Diferente) | Fidelidad: 0.8948 | Loss: 0.8948
Paso 171 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9477 | Loss: 0.0523
Paso 172 | Clases: 1 y 2 (Diferente) | Fidelidad: 0.8048 | Loss: 0.8048
Paso 173 | Clases: 6 y 6 (Misma) | Fidelidad: 0.9275 | Loss: 0.0725
Paso 174 | Clases: 4 y 5 (Diferente) | Fidelidad: 0.9180 | Loss: 0.9180
Paso 175 | Clases: 7 y 7 (Misma) | Fidelidad: 0.9671 | Loss: 0.0329
Paso 176 | Clases: 1 y 5 (Diferente) | Fidelidad: 0.8286 | Loss: 0.8286
Paso 177 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9894 | Loss: 0.0106
Paso 178 | Clases: 0 y 6 (Diferente) | Fidelidad: 0.9351 | Loss: 0.9351
Paso 179 | Clases: 1 y 1 (Misma) | Fidelidad: 0.9783 | Loss: 0.0217
Paso 180 | Clases: 6 y 0 (Diferente) | Fidelidad: 0.8144 | Loss: 0.8144
Paso 181 | Clases: 0 y 0 (Misma) | Fidelidad: 0.9750 | Loss: 0.0250
Paso 182 | Clases: 2 y 0 (Diferente) | Fidelidad: 0.9354 | Loss: 0.9354
Paso 183 | Clases: 2 y 2 (Misma) | Fidelidad: 0.8768 | Loss: 0.1232
Paso 184 | Clases: 1 y 2 (Diferente) | Fidelidad: 0.8605 | Loss: 0.8605
Paso 185 | Clases: 6 y 6 (Misma) | Fidelidad: 0.9688 | Loss: 0.0312
Paso 186 | Clases: 9 y 1 (Diferente) | Fidelidad: 0.9495 | Loss: 0.9495
Paso 187 | Clases: 3 y 3 (Misma) | Fidelidad: 0.8830 | Loss: 0.1170
Paso 188 | Clases: 0 y 7 (Diferente) | Fidelidad: 0.9184 | Loss: 0.9184
Paso 189 | Clases: 0 y 0 (Misma) | Fidelidad: 0.9736 | Loss: 0.0264
Paso 190 | Clases: 7 y 9 (Diferente) | Fidelidad: 0.8920 | Loss: 0.8920
Paso 191 | Clases: 0 y 0 (Misma) | Fidelidad: 0.9587 | Loss: 0.0413
Paso 192 | Clases: 8 y 2 (Diferente) | Fidelidad: 0.9836 | Loss: 0.9836
Paso 193 | Clases: 3 y 3 (Misma) | Fidelidad: 0.9000 | Loss: 0.1000
Paso 194 | Clases: 4 y 1 (Diferente) | Fidelidad: 0.9028 | Loss: 0.9028
Paso 195 | Clases: 9 y 9 (Misma) | Fidelidad: 0.8771 | Loss: 0.1229
Paso 196 | Clases: 5 y 3 (Diferente) | Fidelidad: 0.9757 | Loss: 0.9757
Paso 197 | Clases: 9 y 9 (Misma) | Fidelidad: 0.9715 | Loss: 0.0285
Paso 198 | Clases: 9 y 7 (Diferente) | Fidelidad: 0.9835 | Loss: 0.9835
Paso 199 | Clases: 9 y 9 (Misma) | Fidelidad: 0.9570 | Loss: 0.0430
Paso 200 | Clases: 7 y 9 (Diferente) | Fidelidad: 0.9349 | Loss: 0.9349
Los valores finales de theta son: [ 0.71000546  3.3669798  -0.0877981  -0.0448373   0.22448504 -0.33262438]
```


The I made the evaluation and validation phase of the model. To do this, a function that executes the circuit using the optimized parameters is defined, deactivating the gradient calculation.

Finally, the code carries out two empirical validation tests and generates a visual representation of the results. Extract a couple of images from the same class and another from different classes, processing them through the circuit to calculate their final quantum fidelity



```Python
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transformar)

def evaluar_y_graficar(img_a, img_b, label_a, label_b, theta):

    with torch.no_grad():
        probabilidades = quantum_comparator(img_a, img_b, theta)
        fidelidad = torch.abs(2.0 * probabilidades[0] - 1.0)
        fidelidad = torch.clamp(fidelidad, 0.0, 1.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    axes[0].imshow(img_a.view(8, 8).numpy(), cmap=plt.cm.binary)
    axes[0].set_title(f"Clase: {label_a}")
    axes[0].axis('off')
    
    axes[1].imshow(img_b.view(8, 8).numpy(), cmap=plt.cm.binary)
    axes[1].set_title(f"Clase: {label_b}")
    axes[1].axis('off')
    
    plt.suptitle(f"Fidelidad Cuántica: {fidelidad.item():.4f}\n(1.0 = Idénticas | 0.0 = Diferentes)", fontsize=12)
    plt.tight_layout()
    plt.show()

print("Prueba 1: Esperamos una fidelidad ALTA")
img1_misma, img2_misma, l1_misma, l2_misma = obtener_par_imagenes(test_dataset, misma_clase=True)
evaluar_y_graficar(img1_misma, img2_misma, l1_misma, l2_misma, theta)

print("Prueba 2: Esperamos una fidelidad BAJA ")
img1_dif, img2_dif, l1_dif, l2_dif = obtener_par_imagenes(test_dataset, misma_clase=False)
evaluar_y_graficar(img1_dif, img2_dif, l1_dif, l2_dif, theta)
```
This first image demonstrates high fidelity and is correctly represented.

<img width="515" height="299" alt="image" src="https://github.com/user-attachments/assets/e7e30298-1bd2-4051-8041-6488b938726e" />

The second image reflects high fidelity but I expected low, indicating that the model failed to converge correctly. Increasing the input pixel density might improve the network's performance.

<img width="515" height="299" alt="image" src="https://github.com/user-attachments/assets/bdd9a6e8-4d7f-4435-946e-d3a1451e6fff" />


This last block of code implements a quantitative and statistically more robust evaluation of the quantum model. The function I made extracts random pairs from the test data set, evaluates their similarity by means of the already trained circuit and stores the individual results to finally average them.

```Python

def evaluar_promedio(dataset, theta, misma_clase=True, n=20):
    fidelidades = []
    
    for _ in range(n):
        img_a, img_b, _, _ = obtener_par_imagenes(dataset, misma_clase)
        
        with torch.no_grad():
            probs = quantum_comparator(img_a, img_b, theta)
            fid = torch.clamp(2.0 * probs[0] - 1.0, 0.0, 1.0)
            fidelidades.append(fid.item())
    
    return sum(fidelidades) / len(fidelidades)


fid_misma = evaluar_promedio(test_dataset, theta, True)
fid_dif   = evaluar_promedio(test_dataset, theta, False)

print("Fidelidad promedio (misma clase):", fid_misma)
print("Fidelidad promedio (distinta clase):", fid_dif)


```

So, I obtained the following results:

```Text
Fidelidad promedio (misma clase): 0.9385314471773739
Fidelidad promedio (distinta clase): 0.9099716268307695
```






