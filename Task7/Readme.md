# Task VII Equivariant quantum neural networks

In this section I will show you my procedure tu solve Task 7.

Task 7 ask the following

In this task you are supposed to get started with equivariant quantum neural networks by implementing a Z_2 × Z_2 equivariant quantum neural network. Z_2 is a symmetry group an as an example we will generate a simple classical dataset which is respects the Z_2 x Z_2 symmetry.

This example is explained in the paper https://arxiv.org/abs/2205.06217 and additional background can be found in https://arxiv.org/abs/2210.08566. 

- Generate a classification dataset with two classes and two features x_1 and x_2 which respects the Z_2 x Z_2 symmetry (this corresponds to mirroring along y=x). An example can be found in the first reference paper.
- Train a QNN to solve the classification problem
- Train an Z_2 x Z_2 equivariant QNN to solve the classification problem and compare the results.

### Equivariant quantum neural networks

Equivariant quantum neural networks (EQNNs) are quantum artificial intelligence models designed from scratch to respect and preserve the natural symmetries of the data they process.

Here equivariant means that if I apply a transformation to the input, the output transforms in an identical and predictable way.

Equivariant Quantum Neural Networks offer several important advantages, especially in physics-driven applications. By explicitly incorporating symmetries into the model, they require significantly less training data, since the network does not need to relearn equivalent patterns under different transformations. This leads to improved generalization, as the model can better handle unseen data that follows the same underlying symmetries. Additionally, these networks naturally respect fundamental physical laws, making them particularly suitable for domains where symmetry principles are essential. As a result, EQNNs are especially powerful in fields such as high-energy physics, quantum chemistry, and any system where symmetry plays a central role.

### Code

First, I imported all the necessary packages

```Python
import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

I generated a set of 3000 random points in a 2D plane and assigned them binary labels based on their geographical location: if the point fell within one of the two mathematically defined circular radii, I assigned it a 1, and if not, a 0. Later, I sparted this data to set aside 20% for testing and finally converted all NumPy arrangements into PyTorch tensors.

```Python
def generate_dataset(N=2000):
    X = np.random.uniform(-1, 1, (N, 2))
    
    def label(x1, x2):
        r1 = (x1 + 0.7)**2 + (x2 - 0.7)**2
        r2 = (x1 - 0.7)**2 + (x2 + 0.7)**2
        return 1 if (r1 < 0.5 or r2 < 0.5) else 0
    
    y = np.array([label(x1, x2) for x1, x2 in X])
    return X, y

X, y = generate_dataset(2000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float64)
X_test  = torch.tensor(X_test, dtype=torch.float64)
y_train = torch.tensor(y_train, dtype=torch.float64)
y_test  = torch.tensor(y_test, dtype=torch.float64)

```
Then I visualized the spatial distribution of the data I generated previously.

```Python
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=5)
plt.title("Dataset simétrico Z2 x Z2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
```
Getting the following graph

<img width="587" height="455" alt="image" src="https://github.com/user-attachments/assets/4342c5d5-d201-4f08-b855-11a0dcec7ee2" /> <br>


Then I designed a quantum circuit of a qubit. In it, I encode the coordinates of data by rotating the Y and Z axes, followed by parameterized rotations that the model will adjust during learning. Finally, I measure the state of the qubit (PauliZ) and, by using the Torch interface, I get the circuit to train and optimize automatically by backpropagation, functioning smoothly like another layer of a classic neural network.


```Python
n_qubits = 1
dev = qml.device("default.qubit", wires=n_qubits)

def circuit(x, weights):
    for i in range(len(weights)):
        qml.RY(x[0], wires=0)
        qml.RZ(x[1], wires=0)
        qml.RY(weights[i], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch")
def qnode(x, weights):
    return circuit(x, weights)
```
In the initialization of the model, I defined a set of four randomly initialized weights, which will be the parameters that the network will adjust during training. Then, I programmed the forward to take a full batch of input data, process each element individually through the quantum node, and finally group all the individual predictions into a single output tensor.


```Python
class QNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(4))
        
    def forward(self, x):
        return torch.stack([qnode(xi, self.weights) for xi in x])
```

Then I wrote a function that takes the two coordinates of a point and returns a list with four geometric versions of it: the original point, the point with the axes exchanged, and those same two versions but with the mathematical signs inverted.

```Python
def group_transforms(x):
    x1, x2 = x
    return [
        [x1, x2],
        [x2, x1],
        [-x2, -x1],
        [-x1, -x2]
    ]
```

I built an improved version of the quantum neural network (EQNN). By initializing it, I kept the four random trainable weights, but now, for each point that the network receives, I generate its four geometric variations. Then, I evaluate each one in the quantum circuit and calculate the average of their predictions. By returning these averages, I mathematically force the model to deliver the same result regardless of the reflection or rotation of the original data, creating an inherently symmetrical.


```Python
class EQNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(4))
        
    def forward(self, x):
        outputs = []
        
        for xi in x:
            transformado = group_transforms(xi)
            
            preds = []
            for t in transformado:
                t = torch.tensor(t, dtype=torch.float32)
                preds.append(qnode(t, self.weights))
            
            outputs.append(torch.mean(torch.stack(preds)))
        
        return torch.stack(outputs)
```

I programmed the training cycle using the Adam optimizer. In each iteration, I take the predictions of the quantum model and mathematically scale them to a range of 0 to 1 to be able to calculate the error using the binary cross entropy function. Then, I apply backpropagation to update the weights of the model and record the value of the loss to monitor how he learns season after season.

```Python 
def train(model, X, y, epochs=40, lr=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        opt.zero_grad()
        
        y_pred = (model(X) + 1) / 2 
        loss = torch.nn.BCELoss()(y_pred, y)
        
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        
        print(f"Epoch {epoch}, Loss= {loss.item():.4f}")
    
    return losses
```

I wrote a function to measure the accuracy of the model. And I compare these predictions with the real labels of my data and calculate the mathematical average to obtain the exact percentage of successes of the model.


```Python

def accuracy(model, X, y):
    with torch.no_grad():
        y_pred = (model(X) + 1) / 2
        y_pred = (y_pred > 0.5).float()
        return (y_pred == y).float().mean().item()
```

I tried the above code by instantiating the standard QNN and training it with the separate data set. After adjusting the weights, I used the evaluation function to measure their accuracy with the test data and showed the result to check the classification of the points.

```Python
qnn_model = QNN()
loss_qnn = train(qnn_model, X_train, y_train)

acc_qnn = accuracy(qnn_model, X_test, y_test)
print("QNN Accuracy=", acc_qnn)

```
So I obtained the following results

```Text

Epoch 0, Loss= 1.0286
Epoch 1, Loss= 1.0070
Epoch 2, Loss= 1.0112
Epoch 3, Loss= 1.0104
Epoch 4, Loss= 1.0040
Epoch 5, Loss= 0.9960
Epoch 6, Loss= 0.9942
Epoch 7, Loss= 0.9939
Epoch 8, Loss= 0.9917
Epoch 9, Loss= 0.9904
Epoch 10, Loss= 0.9890
Epoch 11, Loss= 0.9875
Epoch 12, Loss= 0.9868
Epoch 13, Loss= 0.9866
Epoch 14, Loss= 0.9858
Epoch 15, Loss= 0.9870
Epoch 16, Loss= 0.9873
Epoch 17, Loss= 0.9862
Epoch 18, Loss= 0.9862
Epoch 19, Loss= 0.9863
Epoch 20, Loss= 0.9858
Epoch 21, Loss= 0.9857
Epoch 22, Loss= 0.9862
Epoch 23, Loss= 0.9858
Epoch 24, Loss= 0.9853
Epoch 25, Loss= 0.9851
Epoch 26, Loss= 0.9847
Epoch 27, Loss= 0.9841
Epoch 28, Loss= 0.9840
Epoch 29, Loss= 0.9840
Epoch 30, Loss= 0.9837
Epoch 31, Loss= 0.9836
Epoch 32, Loss= 0.9835
Epoch 33, Loss= 0.9832
Epoch 34, Loss= 0.9832
Epoch 35, Loss= 0.9833
Epoch 36, Loss= 0.9832
Epoch 37, Loss= 0.9833
Epoch 38, Loss= 0.9834
Epoch 39, Loss= 0.9833
QNN Accuracy= 0.49166667461395264
```


And then train the EQNN network

```Python
eqnn_model = EQNN()
loss_eqnn = train(eqnn_model, X_train, y_train)

acc_eqnn = accuracy(eqnn_model, X_test, y_test)
print("EQNN Accuracy=", acc_eqnn)
```
That gave the following results

```Text
Epoch 0, Loss= 0.7709
Epoch 1, Loss= 0.7358
Epoch 2, Loss= 0.7111
Epoch 3, Loss= 0.6956
Epoch 4, Loss= 0.6865
Epoch 5, Loss= 0.6811
Epoch 6, Loss= 0.6778
Epoch 7, Loss= 0.6756
Epoch 8, Loss= 0.6741
Epoch 9, Loss= 0.6730
Epoch 10, Loss= 0.6722
Epoch 11, Loss= 0.6716
Epoch 12, Loss= 0.6711
Epoch 13, Loss= 0.6707
Epoch 14, Loss= 0.6704
Epoch 15, Loss= 0.6702
Epoch 16, Loss= 0.6701
Epoch 17, Loss= 0.6700
Epoch 18, Loss= 0.6700
Epoch 19, Loss= 0.6700
Epoch 20, Loss= 0.6701
Epoch 21, Loss= 0.6702
Epoch 22, Loss= 0.6703
Epoch 23, Loss= 0.6704
Epoch 24, Loss= 0.6705
Epoch 25, Loss= 0.6706
Epoch 26, Loss= 0.6707
Epoch 27, Loss= 0.6707
Epoch 28, Loss= 0.6708
Epoch 29, Loss= 0.6708
Epoch 30, Loss= 0.6708
Epoch 31, Loss= 0.6708
Epoch 32, Loss= 0.6708
Epoch 33, Loss= 0.6707
Epoch 34, Loss= 0.6707
Epoch 35, Loss= 0.6706
Epoch 36, Loss= 0.6706
Epoch 37, Loss= 0.6705
Epoch 38, Loss= 0.6704
Epoch 39, Loss= 0.6704
EQNN Accuracy= 0.5533333420753479

```
Finally I generated a graph to visually compare the learning process of my two models.

```Python
plt.plot(loss_qnn, label="QNN")
plt.plot(loss_eqnn, label="EQNN")
plt.legend()
plt.title("Función de Pérdida")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

```
And I obtained the following graph of the loss against the epochs

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/2e917ab8-53b4-4088-8086-165ecf93b1ef" />






