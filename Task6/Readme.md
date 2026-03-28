### Task VI: Quantum representation learning


In this section I will show you my procedure tu solve Task 6






```Python

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math
import pennylane as qml
import tensorflow as tf
import random
import torch.optim as optim


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



img_a_prueba = torch.rand(16)
img_b_prueba = torch.rand(16)
theta_prueba = torch.rand(16)

fig, ax = qml.draw_mpl(quantum_comparator)(img_a_prueba, img_b_prueba, theta_prueba)

plt.show()


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

print("Prueba 1: Esperamos una fidelidad ALTA (cercana a 1.0)")
img1_misma, img2_misma, l1_misma, l2_misma = obtener_par_imagenes(test_dataset, misma_clase=True)
evaluar_y_graficar(img1_misma, img2_misma, l1_misma, l2_misma, theta)

print("Prueba 2: Esperamos una fidelidad BAJA (más cercana a 0.0)")
img1_dif, img2_dif, l1_dif, l2_dif = obtener_par_imagenes(test_dataset, misma_clase=False)
evaluar_y_graficar(img1_dif, img2_dif, l1_dif, l2_dif, theta)






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







