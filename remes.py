"""
REMES: Relatedness-based Multi-Entity Summarization
====================================================
Unofficial Python re-implementation of:

    Gunaratna, Yazdavar, Thirunarayan, Sheth, Cheng.
    "Relatedness-based Multi-Entity Summarization."
    IJCAI 2017.   https://www.ijcai.org/proceedings/2017/0147.pdf

The paper has no public source release. This module reproduces the formulation
faithfully:

  * Feature importance  rank(f) = Inf(f) * Po(Val(f))                   (Eqs 7-9)
  * Pairwise relatedness r(f_i, f_j) = (SemRel_p + GraphRel_v) / 2      (Eqs 10-12)
  * Profit matrix:
        P[a,a]  =  alpha * rank(f_a)                                    (diagonal)
        P[a,b]  = -beta  * r(f_a, f_b)   if same entity                 (intra)
        P[a,b]  = +gamma * r(f_a, f_b)   if different entities          (inter)   (Eq 6)
  * QMKP objective with one knapsack-budget k_i per entity              (Eq 4)
  * Memory-based GRASP construction + local search                      (Sect 4.2/4.3)
        - augmented greedy score with tau / phi terms                   (Eq 5)
        - eta-threshold RCL filter for diversity                        (Sect 4.3)

External dependencies are NumPy (required) and NetworkX (only if you want to
load a graph from a `nx.MultiDiGraph`). NLTK / WordNet is *optional*: if
installed, property similarity follows the paper exactly (jaccard over property
labels expanded with WordNet hypernyms); otherwise it falls back to a token
Jaccard. RDF2Vec embeddings are pluggable -- pass any dict[str, np.ndarray] to
`make_embedding_value_similarity`.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# =============================================================================
#  Data structures
# =============================================================================

@dataclass(frozen=True)
class Feature:
    """A property-value pair (p, v) describing some entity in the KG."""
    prop: str
    value: str

    def __repr__(self) -> str:
        return f"({self.prop} -> {self.value})"


# =============================================================================
#  Knowledge graph wrapper
# =============================================================================

class KnowledgeGraph:
    """Minimal RDF-style KG used by REMES.

    Build from a list of (s, p, o) triples or from a `networkx.MultiDiGraph`
    in which each edge carries a `predicate` (or `label` / `relation`)
    attribute.

    The wrapper maintains the indices needed by Eqs. 7-9 (informativeness
    and value popularity).
    """

    def __init__(
        self,
        triples: Optional[Iterable[Tuple[str, str, str]]] = None,
        nx_graph=None,
    ):
        self._fs: Dict[str, List[Feature]] = defaultdict(list)
        self._all_entities: Set[str] = set()
        # f -> set of entities described by f  (for IDF / Eq 7)
        self._feature_to_entities: Dict[Feature, Set[str]] = defaultdict(set)
        # value -> count of triples whose object is this value (Eq 8)
        self._value_count: Dict[str, int] = defaultdict(int)

        if triples is not None:
            for s, p, o in triples:
                self.add(s, p, o)
        if nx_graph is not None:
            self._load_from_networkx(nx_graph)

    # ---------- construction ----------

    def add(self, s: str, p: str, o: str) -> None:
        s, p, o = str(s), str(p), str(o)
        f = Feature(p, o)
        self._fs[s].append(f)
        self._all_entities.add(s)
        self._feature_to_entities[f].add(s)
        self._value_count[o] += 1

    def _load_from_networkx(self, g) -> None:
        for u, v, data in g.edges(data=True):
            p = data.get("predicate") or data.get("label") or data.get("relation")
            if p is None:
                continue
            self.add(u, p, v)

    # ---------- queries ----------

    def feature_set(self, e: str) -> List[Feature]:
        """FS(e): the feature set of entity e (deduplicated, order preserved)."""
        seen: Set[Feature] = set()
        out: List[Feature] = []
        for f in self._fs.get(e, []):
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def num_entities(self) -> int:
        return len(self._all_entities)

    # ---------- importance (Eqs 7-9) ----------

    def informativeness(self, f: Feature) -> float:
        """Inf(f) = log(N / |{e | f ∈ FS(e)}|).  +1 smoothing on the
        denominator to keep singleton features informative without
        producing log(N/0)."""
        N = self.num_entities()
        df = len(self._feature_to_entities.get(f, ()))
        if df == 0 or N == 0:
            return 0.0
        return math.log((N + 1.0) / df)

    def value_popularity(self, value: str) -> float:
        """Po(v): log of the number of triples whose object is v."""
        c = self._value_count.get(value, 0)
        return math.log(1.0 + c)

    def rank(self, f: Feature) -> float:
        """rank(f) = Inf(f) · Po(Val(f))."""
        return self.informativeness(f) * self.value_popularity(f.value)


# =============================================================================
#  Relatedness measures (Eqs 10-12)
# =============================================================================

_STOP = {
    "the", "a", "an", "of", "is", "are", "was", "were", "for",
    "in", "on", "at", "by", "to", "and", "or", "with",
}


def _tokenize_label(name: str) -> List[str]:
    """Tokenize an RDF predicate / entity label.  Splits on URI boundary,
    snake_case and camelCase; lowercases; drops a small stopword list."""
    name = re.sub(r".*[#/]", "", name)            # strip URI prefix
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    toks = [t.lower() for t in re.split(r"\W+", name) if t]
    return [t for t in toks if t and t not in _STOP]


def _wordnet_hypernyms(token: str) -> Set[str]:
    """Return {token} ∪ hypernym-lemmas if NLTK/WordNet is available;
    otherwise just {token}.  This is the lexical expansion described in
    Section 4.3 ("we also pre-process them and combine with hypernyms")."""
    out: Set[str] = {token}
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
        synsets = wn.synsets(token, pos=wn.NOUN)[:3]
        for syn in synsets:
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    out.add(lemma.name().lower().replace("_", " "))
    except Exception:
        pass
    return out


def wordnet_property_similarity(p1: str, p2: str) -> float:
    """SemRel_p(f_i, f_j) -- jaccard of property labels expanded with
    WordNet hypernyms (Eq 10)."""
    s1: Set[str] = set()
    for t in _tokenize_label(p1):
        s1 |= _wordnet_hypernyms(t)
    s2: Set[str] = set()
    for t in _tokenize_label(p2):
        s2 |= _wordnet_hypernyms(t)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def jaccard_token_value_similarity(v1: str, v2: str) -> float:
    """Fallback for GraphRel_v when no embedding model is available:
    token-jaccard between the two value labels."""
    s1, s2 = set(_tokenize_label(v1)), set(_tokenize_label(v2))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def make_embedding_value_similarity(
    embeddings: Dict[str, np.ndarray],
) -> Callable[[str, str], float]:
    """Build GraphRel_v from a dict {entity_iri: embedding_vector}.

    Plug in pre-trained RDF2Vec, TransE, ComplEx, ... -- any vector model
    over your KG vocabulary.  Returns cosine similarity clipped to [0, 1].
    Unseen entities yield 0."""

    def sim(v1: str, v2: str) -> float:
        e1 = embeddings.get(v1)
        e2 = embeddings.get(v2)
        if e1 is None or e2 is None:
            return 0.0
        n1 = float(np.linalg.norm(e1))
        n2 = float(np.linalg.norm(e2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        c = float(np.dot(e1, e2) / (n1 * n2))
        return max(0.0, min(1.0, 0.5 * (c + 1.0)))  # rescale [-1,1] -> [0,1]
    return sim


class HybridRelatedness:
    """r(f_i, f_j) = (SemRel_p(p_i, p_j) + GraphRel_v(v_i, v_j)) / 2  (Eq 12).

    `prop_sim` and `value_sim` are user-supplied functions; sensible defaults
    are applied if you don't pass them."""

    def __init__(
        self,
        prop_sim: Callable[[str, str], float] = wordnet_property_similarity,
        value_sim: Callable[[str, str], float] = jaccard_token_value_similarity,
    ):
        self.prop_sim = prop_sim
        self.value_sim = value_sim
        self._cache: Dict[Tuple[Feature, Feature], float] = {}

    def relatedness(self, f1: Feature, f2: Feature) -> float:
        key = (f1, f2) if (f1.prop, f1.value) <= (f2.prop, f2.value) else (f2, f1)
        if key in self._cache:
            return self._cache[key]
        v = 0.5 * (self.prop_sim(f1.prop, f2.prop) + self.value_sim(f1.value, f2.value))
        self._cache[key] = v
        return v


# =============================================================================
#  REMES summarizer (QMKP + memory-based GRASP)
# =============================================================================

class REMES:
    """Generate REMES summaries for a collection of entities.

    Parameters mirror the paper (Section 5.1 default values shown below).

    Profit-matrix weights
        alpha   importance reward (diagonal)            (default 2.0)
        beta    intra-entity diversity penalty          (default 1.0)
        gamma   inter-entity relatedness reward         (default 1.5)

    Greedy-score (Eq 5)
        tau     weight on profits with already selected (default 1.0)
        phi     weight on profits with unselected       (default 0.5)

    RCL filter
        eta     diversity threshold for candidate list  (default 0.45)

    GRASP loop
        grasp_iters  number of restarts                 (default 30)
        rcl_size     restricted candidate-list size     (default 5)
        local_search_max_swaps   local-search budget    (default 50)
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        relatedness: Optional[HybridRelatedness] = None,
        alpha: float = 2.0,
        beta: float = 1.0,
        gamma: float = 1.5,
        tau: float = 1.0,
        phi: float = 0.5,
        eta: float = 0.45,
        grasp_iters: int = 30,
        rcl_size: int = 5,
        local_search_max_swaps: int = 50,
        random_seed: Optional[int] = None,
    ):
        self.kg = kg
        self.relatedness = relatedness or HybridRelatedness()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.eta = eta
        self.grasp_iters = grasp_iters
        self.rcl_size = rcl_size
        self.local_search_max_swaps = local_search_max_swaps
        self.rng = random.Random(random_seed)

    # ----------------------------------------------------------------- API
    def summarize(
        self,
        entities: Sequence[str],
        k: Sequence[int],
    ) -> Dict[str, List[Feature]]:
        """Return {entity -> ranked summary features} of size <= k_i.

        For 2-entity input (source / target), pass `[src, tgt]` and `[k1, k2]`.
        Generalises directly to n entities.
        """
        if len(entities) != len(k):
            raise ValueError("`k` must have one budget per entity")
        if len(entities) < 2:
            raise ValueError("REMES is multi-entity; supply at least 2 entities")

        # 1. Collect feature sets per entity --------------------------------
        feature_sets = [self.kg.feature_set(e) for e in entities]
        if any(len(fs) == 0 for fs in feature_sets):
            missing = [e for e, fs in zip(entities, feature_sets) if not fs]
            raise ValueError(f"No features in KG for: {missing}")

        # Flatten into items[a] = (entity_idx, Feature)
        items: List[Tuple[int, Feature]] = []
        items_by_entity: List[List[int]] = []
        for ei, fs in enumerate(feature_sets):
            idxs: List[int] = []
            for f in fs:
                idxs.append(len(items))
                items.append((ei, f))
            items_by_entity.append(idxs)

        # 2. Build profit matrix P (Eq 6) -----------------------------------
        n = len(items)
        P = np.zeros((n, n), dtype=np.float64)

        ranks = np.array([self.kg.rank(items[a][1]) for a in range(n)])
        rmax = ranks.max() if ranks.size else 1.0
        ranks_norm = ranks / rmax if rmax > 0 else ranks

        for a in range(n):
            P[a, a] = self.alpha * ranks_norm[a]
            ea, fa = items[a]
            for b in range(a + 1, n):
                eb, fb = items[b]
                r = self.relatedness.relatedness(fa, fb)
                if ea == eb:
                    P[a, b] = -self.beta * r
                else:
                    P[a, b] = +self.gamma * r
                P[b, a] = P[a, b]

        # Normalise globally (paper, Sect 5.1: "divide by max profit")
        amax = float(np.max(np.abs(P)))
        if amax > 0:
            P = P / amax

        # 3. Modified GRASP -------------------------------------------------
        best_sol: Optional[List[List[int]]] = None
        best_profit = -math.inf

        for _ in range(self.grasp_iters):
            sol = self._construct(P, items, items_by_entity, k)
            sol = self._local_search(sol, P, items, items_by_entity, k)
            prof = self._total_profit(sol, P)
            if prof > best_profit:
                best_profit = prof
                best_sol = [list(s) for s in sol]

        assert best_sol is not None

        # 4. Map back to entity-keyed result, sorted by per-feature score ---
        out: Dict[str, List[Feature]] = {}
        for ei, e in enumerate(entities):
            sel = best_sol[ei]
            # Order features by their contribution: P[a,a] + sum_{b in S} P[a,b]
            S_all = set(i for s in best_sol for i in s)

            def contrib(a: int) -> float:
                return P[a, a] + sum(P[a, b] for b in S_all if b != a)

            sel_sorted = sorted(sel, key=contrib, reverse=True)
            out[e] = [items[a][1] for a in sel_sorted]
        return out

    # ----------------------------------------------------------- GRASP bits
    def _greedy_score(
        self,
        S: Set[int],
        f: int,
        P: np.ndarray,
        all_idx: Sequence[int],
    ) -> float:
        """Equation 5, with unit weights (paper, Sect 5.1)."""
        if S:
            S_list = list(S)
            pair_S = sum(P[i, j] for i in S_list for j in S_list if j <= i)
            tau_term = self.tau * sum(P[x, f] for x in S_list)
        else:
            pair_S, tau_term = 0.0, 0.0
        phi_term = self.phi * sum(
            P[x, f] for x in all_idx if x != f and x not in S
        )
        diag = P[f, f]
        denom = float(len(S) + 1)            # unit weights
        return (pair_S + tau_term + phi_term + diag) / denom

    def _intra_entity_relatedness(
        self,
        f_idx: int,
        sel_in_entity: Sequence[int],
        items: List[Tuple[int, Feature]],
    ) -> float:
        """max relatedness of candidate f with already-selected features in
        the SAME entity -- used by the eta filter (Section 4.3)."""
        if not sel_in_entity:
            return 0.0
        fa = items[f_idx][1]
        return max(
            self.relatedness.relatedness(fa, items[j][1]) for j in sel_in_entity
        )

    def _construct(
        self,
        P: np.ndarray,
        items: List[Tuple[int, Feature]],
        items_by_entity: List[List[int]],
        k: Sequence[int],
    ) -> List[List[int]]:
        """Construction phase (memory-based GRASP, Sect 4.2).

        At every step we pool eta-feasible candidates from *all* entities
        whose knapsacks aren't full, score them with Gr(S, f), keep the
        top `rcl_size` as the Restricted Candidate List, then sample one
        uniformly.  This is the standard GRASP randomised-greedy schema."""
        sol: List[List[int]] = [[] for _ in items_by_entity]
        sol_set: Set[int] = set()
        # Map item index -> its entity
        owner = {i: ei for ei, idxs in enumerate(items_by_entity) for i in idxs}
        all_idx = list(owner.keys())

        while True:
            # Pool candidates across entities that still have capacity
            cands: List[int] = []
            for ei, idxs in enumerate(items_by_entity):
                if len(sol[ei]) >= k[ei]:
                    continue
                for f_idx in idxs:
                    if f_idx in sol_set:
                        continue
                    if (
                        self._intra_entity_relatedness(f_idx, sol[ei], items)
                        > self.eta
                    ):
                        continue
                    cands.append(f_idx)
            if not cands:
                break

            scored = [(self._greedy_score(sol_set, f, P, all_idx), f)
                      for f in cands]
            scored.sort(reverse=True)
            rcl = scored[: max(1, self.rcl_size)]
            _, pick_f = self.rng.choice(rcl)
            sol[owner[pick_f]].append(pick_f)
            sol_set.add(pick_f)

        return sol

    def _local_search(
        self,
        sol: List[List[int]],
        P: np.ndarray,
        items: List[Tuple[int, Feature]],
        items_by_entity: List[List[int]],
        k: Sequence[int],
    ) -> List[List[int]]:
        """Best-improvement swap-based local search.  At each step we look
        for the single (entity, selected, candidate) swap that yields the
        largest strict improvement in total profit and apply it; we stop
        when no improving swap exists or we hit the budget."""
        sol = [list(s) for s in sol]
        cur_profit = self._total_profit(sol, P)

        for _ in range(self.local_search_max_swaps):
            best_swap: Optional[Tuple[int, int, int, float]] = None

            for ei, idxs in enumerate(items_by_entity):
                in_sel = set(sol[ei])
                unsel = [x for x in idxs if x not in in_sel]
                for sel in sol[ei]:
                    others = [x for x in sol[ei] if x != sel]
                    for cand in unsel:
                        if (
                            self._intra_entity_relatedness(cand, others, items)
                            > self.eta
                        ):
                            continue
                        new_sol = [list(s) for s in sol]
                        new_sol[ei] = [cand if x == sel else x for x in sol[ei]]
                        p = self._total_profit(new_sol, P)
                        if p > cur_profit + 1e-9 and (
                            best_swap is None or p > best_swap[3]
                        ):
                            best_swap = (ei, sel, cand, p)

            if best_swap is None:
                break
            ei, sel, cand, p = best_swap
            sol[ei] = [cand if x == sel else x for x in sol[ei]]
            cur_profit = p
        return sol

    @staticmethod
    def _total_profit(sol: List[List[int]], P: np.ndarray) -> float:
        flat = sorted(i for s in sol for i in s)
        if not flat:
            return 0.0
        s = 0.0
        for ai, i in enumerate(flat):
            for j in flat[ai:]:
                s += P[i, j]
        return s
