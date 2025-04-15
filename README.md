# Hot Vectors

## Overview
Hot Vectors is a vector database designed with native hierarchical clustering and parallelism in mind. The project aims to provide fast, deterministic, and stable vector operations while optimizing for performance and scalability.

## Current Goals
- **Implement essential atomic operations**: Develop fundamental operations for managing vector partitions and clusters efficiently:
  - Merge partitions
  - Add vectors to partitions
  - Create cluster sets
  - Query operations (KNN, get vectors, get clusters, etc.)
  - Merge partitions
- **Implement vector insertion**: Develop a robust method for inserting vectors efficiently into the database while maintaining clustering properties.
- **Optimize for speed**: Ensure that insertion and retrieval operations are highly performant, leveraging parallel processing where possible.
- **Ensure stability to prevent crashes**: Implement rigorous error handling and stress testing to make the system resilient to edge cases.
- **Maintain determinism in operations**: Ensure that operations produce consistent and predictable results, which is crucial for reliability in clustering and search tasks.
- **GPU Optimization**: Leverage GPU acceleration to speed up computations, particularly for large-scale vector operations, nearest neighbor searches, and clustering algorithms.
- **Expand Clustering Algorithms**: Implement additional clustering techniques such as HDBSCAN and other advanced methods to improve flexibility and accuracy in hierarchical clustering.

## Future Goals
- **Add multiple connection methods**: Implement various ways to connect to the database, allowing for flexibility in integration:
  - Support different connection protocols
  - Implement efficient communication channels
  - Improve the REST API for better interaction
  - Enable connection types as configurable features that can be toggled at compile time.
- **Support transactions**: Introduce a transactional system to allow atomic operations on multiple vectors, ensuring data integrity.
- **Improve batch insertions for efficiency**: Enhance the system's ability to process large batches of vectors efficiently, reducing overhead and improving throughput.
- **Enable deletion of vectors**: Implement a method for safely removing vectors from the database while preserving clustering structures and minimizing reprocessing costs.

## Demonstration
### Vectors being inserted into a database
![Vectors being inserted into a database](.\demo.gif)

## Future Optimization
### Updates of Minimum Spanning Trees
- [Dynamic Euclidean MST](https://link.springer.com/article/10.1007/BF01228509): Explores efficient methods for maintaining a minimum spanning tree dynamically as the dataset evolves.
- [Fully Dynamic MST](https://link.springer.com/article/10.1007/BF01944354): Discusses fully dynamic solutions for updating spanning trees, which can be useful in clustering applications.

### Better Neighbor Detection
- [Efficient Delaunay Triangulation](https://dl.acm.org/doi/pdf/10.1145/220279.220286): Discusses sequential algorithms for constructing Delaunay triangulations, comparing divide-and-conquer, sweepline, and incremental approaches.Covers techniques for improving the accuracy and speed of nearest neighbor searches, a core component of vector databases.
- [Approximate UMAP for Real-Time Projections](https://arxiv.org/pdf/2404.04001): Introduces a novel variant of UMAP (aUMAP) designed for rapid, real-time data projections, significantly reducing computational costs while maintaining accuracy.A research paper detailing the latest methodologies in neighbor detection, which can enhance clustering and search performance.

