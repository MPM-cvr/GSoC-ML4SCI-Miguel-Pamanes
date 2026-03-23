### Task II: Classical Graph Neural Network (GNN) 

In this section I will show you my procedure tu solve Task 2

For Task II, you will use ParticleNet’s data for Quark/Gluon jet classification available here with its corresponding description. 
- Choose 2 Graph-based architectures of your choice to classify jets as being quarks or gluons. Provide a description on what considerations you have taken to project this point-cloud dataset to a set of interconnected nodes and edges.
- Discuss the resulting performance of the 2 chosen architectures. 

### Graph Neural Networks

Graph Neural Networks (GNNs) are neural networks designed to work with data that have a graph structure. A graph represents the relationships, which are called "edges", between a collection of entities, which are called "nodes". 

<img width="875" height="659" alt="image" src="https://github.com/user-attachments/assets/01efbcca-3fd0-44b0-9d7b-79a2c83498f1" />
(Source: Anay Dongre, “A Comprehensive Introduction to Graph Neural Networks,” Towards AI, 2023) <br>



In a GNN, the information of a node depends not only on its own characteristics, but also on the characteristics of its neighbors and how they connect with each other.

GNNs work through an iterative process called Neural Message Passing.

- Message: Each node sends its information (its feature vector) to all its connected neighbors.
- Aggregation: The receiving node takes all the messages from its neighbors and summarizes them in a single vector.
- Update: The node combines the aggregated information of its neighbors with its own current information to generate a new state.

For this task, I chose these architectures:
- Graph Convolutional Networks
- Interaction Networks


