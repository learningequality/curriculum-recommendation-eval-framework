# curriculum-recommendation-eval-framework
Shared codebase for evaluation framework for curriculum alignment recommendation algorithms

## Two input files to evalution.py
**nodes-path:** a csv containing all nodes' metadata, including both topic nodes and content nodes. Columns must include "node_id", "content_id", "parent_id", "kind", "level", "language", "condition", "channel_id".  
**embeddings-path:** a csv containing all nodes' embeddings. The last column must be named "weight" and should store the weight for each embedding. Topic nodes can have multiple embeddings, but content nodes can only have one embedding. The "weight" column for content nodes will be ignored.
