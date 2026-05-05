# REMES — Relatedness-based Multi-Entity Summarization

Python re-implementation of

> Gunaratna, Yazdavar, Thirunarayan, Sheth, Cheng.
> *Relatedness-based Multi-Entity Summarization.* IJCAI 2017.
> [paper](https://www.ijcai.org/proceedings/2017/0147.pdf)

The paper ships no public source. This module reproduces the formulation
end-to-end on top of NumPy + (optionally) NetworkX.

## What's in here

| File | What it is |
|------|------------|
| `remes.py`        | Core library: `KnowledgeGraph`, `HybridRelatedness`, `REMES`. |
| `example_demo.py` | Toy KG (Steve Jobs / Apple Inc.) reproducing Figure 1 of the paper. |
| `test_remes.py`   | Sanity check: REMES must beat random selection on its own QMKP objective. |

## Mapping to the paper

| Paper construct | Code |
|---|---|
| Feature `f = (p, v)`                              | `Feature(prop, value)` |
| Feature set `FS(e)`                               | `KnowledgeGraph.feature_set(e)` |
| Informativeness `Inf(f)` (Eq 7)                   | `KnowledgeGraph.informativeness(f)` |
| Value popularity `Po(v)` (Eq 8)                   | `KnowledgeGraph.value_popularity(v)` |
| Importance `rank(f)` (Eq 9)                       | `KnowledgeGraph.rank(f)` |
| Property similarity `SemRel_p` (Eq 10, WordNet)   | `wordnet_property_similarity` |
| Value similarity `GraphRel_v` (Eq 11, RDF2Vec)    | `make_embedding_value_similarity` |
| Pairwise relatedness `r(f_i, f_j)` (Eq 12)        | `HybridRelatedness.relatedness` |
| Profit matrix `p_{i,a,j,b}` (Eq 6)                | built inside `REMES.summarize` |
| QMKP with per-entity budgets `k_i` (Eq 4)         | `REMES.summarize(entities, k_list)` |
| Greedy ranking `Gr(S, f)` (Eq 5)                  | `REMES._greedy_score` |
| η-threshold RCL filter (Sect 4.3)                 | `REMES._intra_entity_relatedness` |
| Memory-based GRASP (Sect 4.2)                     | `_construct` + `_local_search` |

Default hyper-parameters match Section 5.1 of the paper:
α=2, β=1, γ=1.5, τ=1, φ=0.5, η=0.45, unit feature weights.

## Quick start

```python
import networkx as nx
from remes import KnowledgeGraph, REMES

# 1. Build / load a KG
g = nx.MultiDiGraph()
for s, p, o in triples:
    g.add_edge(s, o, predicate=p)
kg = KnowledgeGraph(nx_graph=g)

# 2. Summarise a multi-entity collection
remes = REMES(kg, random_seed=0)
summaries = remes.summarize(
    entities=["Steve_Jobs", "Apple_Inc"],   # >= 2 entities
    k=[4, 4],                                # one budget per entity
)

for entity, features in summaries.items():
    print(entity)
    for f in features:
        print(" ", f.prop, "->", f.value)
```

## Plugging in real similarity backends

The two relatedness components are pluggable:

```python
from remes import (
    HybridRelatedness, REMES,
    make_embedding_value_similarity, wordnet_property_similarity,
)

# Real RDF2Vec / TransE / ComplEx vectors keyed by entity IRI
embeddings = load_my_pretrained_embeddings()      # dict[str, np.ndarray]

rel = HybridRelatedness(
    prop_sim  = wordnet_property_similarity,                # WordNet hypernyms
    value_sim = make_embedding_value_similarity(embeddings) # cosine on vectors
)
remes = REMES(kg, relatedness=rel, random_seed=0)
```

If NLTK / WordNet aren't installed, `wordnet_property_similarity` silently
falls back to a token-Jaccard. If you don't pass embeddings,
`HybridRelatedness()` defaults to a token-Jaccard for value similarity too.
Both fallbacks let the code run with zero extra dependencies; for paper-
faithful behaviour install NLTK and load real RDF2Vec vectors.

## Run the demo

```
python example_demo.py
python test_remes.py
```

## Notes / deviations

* Eq 8's `log|...|` can be 0 when the value count is 1; we use `log(1 + count)`
  to keep the rank score well-behaved.
* The eta-filter in the paper is described as a constraint on "any already
  selected feature." Since cross-entity profits are positive (γ·r), filtering
  on the *full* selected set would also kick out features that are highly
  inter-entity-related — which contradicts the stated objective of *promoting*
  inter-entity relatedness. We therefore apply the η filter only against
  features already selected for the *same* entity, which matches the paper's
  stated goal: "introduce better diversity in the results for each entity".
* GRASP is non-deterministic; pass `random_seed` for reproducibility.
* Computational complexity is O(n²) for the profit matrix and roughly
  O(iters · k · n²) for GRASP, where n is the total number of features
  across all input entities. Fine for entity-summarization-scale n (10²–10³).
