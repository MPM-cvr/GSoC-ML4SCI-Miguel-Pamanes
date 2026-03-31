# Task VII Equivariant quantum neural networks


```Python
import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




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





plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=5)
plt.title("Dataset simétrico Z2 x Z2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()




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





class QNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(4))
        
    def forward(self, x):
        return torch.stack([qnode(xi, self.weights) for xi in x])





def group_transforms(x):
    x1, x2 = x
    return [
        [x1, x2],
        [x2, x1],
        [-x2, -x1],
        [-x1, -x2]
    ]






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







def accuracy(model, X, y):
    with torch.no_grad():
        y_pred = (model(X) + 1) / 2
        y_pred = (y_pred > 0.5).float()
        return (y_pred == y).float().mean().item()





qnn_model = QNN()
loss_qnn = train(qnn_model, X_train, y_train)

acc_qnn = accuracy(qnn_model, X_test, y_test)
print("QNN Accuracy=", acc_qnn)





eqnn_model = EQNN()
loss_eqnn = train(eqnn_model, X_train, y_train)

acc_eqnn = accuracy(eqnn_model, X_test, y_test)
print("EQNN Accuracy=", acc_eqnn)




plt.plot(loss_qnn, label="QNN")
plt.plot(loss_eqnn, label="EQNN")
plt.legend()
plt.title("Función de Pérdida")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()



```






