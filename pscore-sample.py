from pscore import ReputationScorer

# This is the example from section 4.2 of the paper.
#
# Group 1 published:
# - 3 papers in venue v1 (graph node id = 0)
# - 2 papers in venue v2 (graph node id = 1)
# - 1 paper  in venue v3 (graph node id = 2)
#
# The number of publications of Group 1 is 6.
#
# Venue v1 receives:
# - 3 papers of group g1 (graph node id = 0)
# - 2 papers of group g2 (graph node id = 1)

# Initialize the ranker
rs = ReputationScorer()

# Initialize graph structure
rs.source_target_edge(0, 0, 3) # source=0 target=0 weight=3
rs.source_target_edge(0, 1, 2) # source=0 target=1 weight=2
rs.source_target_edge(0, 2, 1) # source=0 target=2 weight=1
# source = 1
rs.source_target_edge(1, 0, 2) # source=1 target=0 weight=2
rs.source_target_edge(1, 1, 4) # source=1 target=1 weight=4
rs.source_target_edge(1, 2, 2) # source=1 target=2 weight=2

# Solve markov chain and rank target nodes (all parameters are optional)
target_scores = rs.rank(n_steps=10000, epilson=1e-5)
print(target_scores)
# target_scores is a np.array where each position is the score
# of each target node:
# Output: [ 0.36  0.43  0.21]
