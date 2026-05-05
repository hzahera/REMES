"""
Demo: REMES on a small knowledge graph.
Reproduces (in spirit) Figure 1 of the paper -- summarising
"Steve Jobs" together with "Apple Inc.".

Run:    python example_demo.py
"""

import numpy as np
import networkx as nx

from remes import (
    KnowledgeGraph,
    REMES,
    HybridRelatedness,
    make_embedding_value_similarity,
    wordnet_property_similarity,
)


# --------------------------------------------------------------- toy KG
TRIPLES = [
    # ----- Steve_Jobs
    ("Steve_Jobs", "founder",       "Apple_Inc"),
    ("Steve_Jobs", "founder",       "NeXT"),
    ("Steve_Jobs", "founder",       "Pixar"),
    ("Steve_Jobs", "birthPlace",    "California"),
    ("Steve_Jobs", "deathPlace",    "California"),
    ("Steve_Jobs", "knownFor",      "Microcomputer_revolution"),
    ("Steve_Jobs", "knownFor",      "Apple_Inc"),
    ("Steve_Jobs", "occupation",    "Entrepreneur"),
    ("Steve_Jobs", "spouse",        "Laurene_Powell_Jobs"),
    ("Steve_Jobs", "almaMater",     "Reed_College"),
    ("Steve_Jobs", "boardMember",   "Walt_Disney_Company"),
    ("Steve_Jobs", "title",         "CEO_of_Apple"),

    # ----- Apple_Inc
    ("Apple_Inc",  "founder",       "Steve_Jobs"),
    ("Apple_Inc",  "founder",       "Steve_Wozniak"),
    ("Apple_Inc",  "founder",       "Ronald_Wayne"),
    ("Apple_Inc",  "product",       "iPod"),
    ("Apple_Inc",  "product",       "iPhone"),
    ("Apple_Inc",  "product",       "Macintosh"),
    ("Apple_Inc",  "location",      "California"),
    ("Apple_Inc",  "industry",      "Consumer_electronics"),
    ("Apple_Inc",  "successor",     "Tim_Cook"),
    ("Apple_Inc",  "headquarters",  "Cupertino"),
    ("Apple_Inc",  "foundedYear",   "1976"),

    # ----- background entities (so IDF / popularity statistics are non-trivial)
    ("Microsoft",  "founder",       "Bill_Gates"),
    ("Microsoft",  "founder",       "Paul_Allen"),
    ("Microsoft",  "industry",      "Software"),
    ("Microsoft",  "headquarters",  "Redmond"),
    ("Microsoft",  "product",       "Windows"),
    ("Microsoft",  "product",       "Office"),

    ("Google",     "founder",       "Larry_Page"),
    ("Google",     "founder",       "Sergey_Brin"),
    ("Google",     "industry",      "Internet"),
    ("Google",     "headquarters",  "Mountain_View"),
    ("Google",     "successor",     "Sundar_Pichai"),

    ("Pixar",      "industry",      "Animation"),
    ("Pixar",      "headquarters",  "Emeryville"),
    ("Pixar",      "founder",       "Ed_Catmull"),

    ("NeXT",       "industry",      "Computer"),
    ("NeXT",       "founder",       "Steve_Jobs"),
    ("NeXT",       "headquarters",  "Redwood_City"),
]


# -------------------------------------------- (optional) toy embeddings
# Stand-in for RDF2Vec.  In production, plug in real pretrained vectors.
def make_toy_embeddings(triples, dim=24, seed=7):
    rng = np.random.default_rng(seed)
    vocab = set()
    for s, _, o in triples:
        vocab.add(s)
        vocab.add(o)
    base = {v: rng.normal(size=dim) for v in vocab}

    # Inject light structure: entities sharing a relation neighbour are pulled
    # together.  This is a *toy* surrogate -- it merely makes the demo show
    # the right qualitative behaviour.
    for s, _, o in triples:
        base[s] = 0.85 * base[s] + 0.15 * base[o]
        base[o] = 0.85 * base[o] + 0.15 * base[s]
    return base


# ---------------------------------------------------------------- main
def build_kg():
    g = nx.MultiDiGraph()
    for s, p, o in TRIPLES:
        g.add_edge(s, o, predicate=p)
    return KnowledgeGraph(nx_graph=g)


def render(summary, title):
    print(f"\n=== Summary: {title} ===")
    for f in summary:
        print(f"  {f.prop:14s}  ->  {f.value}")


def run(use_embeddings=True):
    kg = build_kg()

    if use_embeddings:
        emb = make_toy_embeddings(TRIPLES)
        rel = HybridRelatedness(
            prop_sim=wordnet_property_similarity,            # WordNet if available
            value_sim=make_embedding_value_similarity(emb),  # RDF2Vec slot
        )
    else:
        rel = HybridRelatedness()  # both fallbacks

    summarizer = REMES(
        kg,
        relatedness=rel,
        alpha=2.0, beta=1.0, gamma=1.5,
        tau=1.0, phi=0.5, eta=0.45,
        grasp_iters=30,
        rcl_size=5,
        random_seed=42,
    )

    src, tgt = "Steve_Jobs", "Apple_Inc"
    k_src, k_tgt = 4, 4

    print(f"|FS({src})| = {len(kg.feature_set(src))}")
    print(f"|FS({tgt})| = {len(kg.feature_set(tgt))}")

    summaries = summarizer.summarize([src, tgt], [k_src, k_tgt])
    for e, fs in summaries.items():
        render(fs, e)


if __name__ == "__main__":
    run(use_embeddings=True)
