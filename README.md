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
- **Native UMAP Integration**: Implement UMAP directly into the database engine as a built-in dimensionality reduction method. This enables the database to support high-dimensional (n-dim) data while maintaining efficient and meaningful representations for search, clustering, and visualization.

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
- [Efficient Delaunay Triangulation](https://dl.acm.org/doi/pdf/10.1145/220279.220286): Discusses sequential algorithms for constructing Delaunay triangulations, comparing divide-and-conquer, sweepline, and incremental approaches. Covers techniques for improving the accuracy and speed of nearest neighbor searches, a core component of vector databases.
- [Approximate UMAP for Real-Time Projections](https://arxiv.org/pdf/2404.04001): Introduces a novel variant of UMAP (aUMAP) designed for rapid, real-time data projections, significantly reducing computational costs while maintaining accuracy.
- [Parametric UMAP embeddings for representation and semi-supervised learning](https://arxiv.org/pdf/2009.12981): Extends UMAP to learn a parametric mapping via neural networks, enabling fast embeddings of new data and improved regularization in autoencoders. Useful for online vector insertion and semi-supervised learning through structural embedding of unlabeled data.

## TODO

### Currently Tasks

1. **(Medium)** Finalize and test KNN query implementation  
   _Skills_: Rust, distance metrics, unit testing, API/interface design

2. **(Hard)** Implement vector update operation (delete + reinsert)  
   _Skills_: Rust, clustering logic, memory safety

3. **(Hard)** Implement vector deletion  
   _Skills_: Rust, graph maintenance, indexing

4. **(Hard)** GPU-accelerated distance computation  
   _Skills_: CUDA/wgpu, SIMD, parallel Rust

5. **(Medium)** Implement robust error handling and edge-case testing  
   _Skills_: Defensive Rust, concurrency testing

6. **(Medium)** Code cleanup and reorganization  
   _Skills_: Refactoring, idiomatic Rust, file structure cleanup

7. **(Hard)** Improve UMAP: support incremental updates and split into vector vs centroid UMAPs  
   _Skills_: Rust, dimensionality reduction, online learning

8. **(Hard)** Implement chaining of atomic operations  
   _Skills_: Command pattern, operation composition

9. **(Hard)** Optimizer for chained operations  
   _Skills_: Query optimization, DAG rewrite logic

10. **(Medium)** Link vectors to external data (e.g., strings or structured metadata)  
    _Skills_: Serialization, type-safe associations, schema design

11. **(Medium)** Improve REST API performance  
    _Skills_: Rust async, actix/tokio, request profiling

12. **(Medium)** Update and improve connection pool management  
    _Skills_: Resource pooling, concurrency management

13. **(Medium)** Create language bindings (Python/JS/etc.) using connection pool APIs  
    _Skills_: FFI, language interop, API client development

14. **(Medium)** Add filtering to vector selection/query logic  
    _Skills_: Query syntax parsing, filter application, indexing

### Future Tasks

15. **(Medium)** Expand REST API capabilities  
16. **(Easy)** Compile-time toggles for protocol support  
17. **(Hard)** Add transaction support for atomic multi-vector ops  
18. **(Medium)** Improve efficiency of batch vector insertions  
19. **(Hard)** Implement HDBSCAN and other clustering techniques