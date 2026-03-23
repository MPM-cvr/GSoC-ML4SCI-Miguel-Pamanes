### Task II: Classical Graph Neural Network (GNN) 

In this section I will show you my procedure tu solve Task 2

For Task II, you will use ParticleNet’s data for Quark/Gluon jet classification available here with its corresponding description. 
- Choose 2 Graph-based architectures of your choice to classify jets as being quarks or gluons. Provide a description on what considerations you have taken to project this point-cloud dataset to a set of interconnected nodes and edges.
- Discuss the resulting performance of the 2 chosen architectures. 

### Graph Neural Networks

Graph Neural Networks (GNNs) are neural networks designed to work with data that have a graph structure. A graph represents the relationships, which are called "edges", between a collection of entities, which are called "nodes". 

<img width="875" height="659" alt="image" src="https://github.com/user-attachments/assets/01efbcca-3fd0-44b0-9d7b-79a2c83498f1" /> <br>
(Source: Anay Dongre, “A Comprehensive Introduction to Graph Neural Networks,” Towards AI, 2023) <br>

In a GNN, the information of a node depends not only on its own characteristics, but also on the characteristics of its neighbors and how they connect with each other.

GNNs work through an iterative process called Neural Message Passing.

- Message: Each node sends its information (its feature vector) to all its connected neighbors.
- Aggregation: The receiving node takes all the messages from its neighbors and summarizes them in a single vector.
- Update: The node combines the aggregated information of its neighbors with its own current information to generate a new state.

For this task, I chose these architectures:
- Graph Convolutional Networks
- Interaction Networks

### Graph Convolutional Networks (GCN)

The GCN is the direct adaptation of convolutional image networks (CNN) to graphs.

In a GCN, a node mixes its information with all the neighbors to which it is connected.

A GCN takes two matrices as input:
- Matrix of Characteristics (X): Describe what each node is. Dimension: N×F (where N is the number of nodes and F is the number of characteristics per node).
- Adjacency Matrix (A): Describes whether the nodes are connected. It is a N×N matrix full of 0s and 1s. If there is a 1 in the position (i,j), it means that node i and node j are connected.


