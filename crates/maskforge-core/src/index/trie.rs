//! Stores vocabulary tries in a compressed sparse-row layout.

use super::cache::VocabFingerprint;
use crate::automaton::RefEngine;
use crate::error::{CompileError, ErrorCode, Stage};
use crate::primitives::{ByteClassId, TrieNodeId};
use crate::vocab::{PreparedVocabulary, Vocabulary};

/// Bounds on trie construction so a hostile vocabulary cannot drive unbounded work or heap. Each
/// limit is checked before the allocation it guards; the defaults cover any real tokenizer.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TrieBuildLimits {
    /// Maximum vocabulary entries (checked before the `pairs` reference vector is allocated).
    pub max_vocab_entries: usize,
    /// Maximum trie nodes (checked before each node allocation).
    pub max_nodes: usize,
    /// Maximum total token bytes across the vocabulary (checked before any byte is touched).
    pub max_total_token_bytes: u64,
    /// Maximum total leaf token ids (checked as leaves are filled).
    pub max_leaf_token_ids: usize,
}

impl Default for TrieBuildLimits {
    fn default() -> Self {
        Self {
            max_vocab_entries: 1 << 24,
            max_nodes: 1 << 23,
            max_total_token_bytes: 1 << 28,
            max_leaf_token_ids: 1 << 25,
        }
    }
}

/// Which alphabet a trie's edge keys are drawn from.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum TrieKind {
    /// Edge keys are raw bytes; the walk maps each through the engine's class table.
    Byte,
    /// Edge keys are byte-class ids; the walk steps the DFA on them directly.
    Class,
}

/// A trie over the vocabulary's token byte-strings (or their class signatures), CSR-packed.
///
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabTrie {
    first_child: Vec<u32>,
    token_head: Vec<u32>,
    edge_key: Vec<u16>,
    child_node: Vec<TrieNodeId>,
    leaf_tokens: Vec<u32>,
    kind: TrieKind,
    vocab_fingerprint: VocabFingerprint,
    partition: Option<[u8; 32]>,
    max_dfs_frontier: usize,
}

/// The exact maximum simultaneous DFS-stack size a walk over this topology can reach: push the
/// root, then repeatedly pop one node and push all its children (the same discipline
fn compute_max_dfs_frontier(first_child: &[u32], child_node: &[TrieNodeId]) -> usize {
    if first_child.len() <= 1 {
        return 0; // no root node
    }
    let mut stack: Vec<u32> = vec![0];
    let mut maximum = stack.len();
    while let Some(n) = stack.pop() {
        let n = n as usize;
        let start = first_child[n] as usize;
        let end = first_child[n + 1] as usize;
        stack.extend(child_node[start..end].iter().map(|c| c.get()));
        maximum = maximum.max(stack.len());
    }
    maximum
}

/// Borrowed `(token bytes, token ids)` references into a vocabulary, the trie build's input.
type BorrowedPairs<'a> = Vec<(&'a [u8], &'a [u32])>;

/// A mutable trie node during construction; children stay sorted ascending by key.
#[derive(Default)]
struct BuildNode {
    children: Vec<(u16, usize)>,
    tokens: Vec<u32>,
}

impl VocabTrie {
    /// Builds the cacheable byte-trie: one edge per distinct token byte. Schema-independent.
    pub fn build_byte(vocab: &Vocabulary) -> Result<Self, CompileError> {
        build_byte_limited(vocab, TrieBuildLimits::default())
    }

    /// Builds the byte-trie stamping `fingerprint` (the caller's already-computed
    /// `VocabularyHandle` fingerprint) instead of rehashing the whole vocabulary. Crate-internal so
    pub(crate) fn build_byte_with_fingerprint(
        vocab: &Vocabulary,
        fingerprint: VocabFingerprint,
    ) -> Result<Self, CompileError> {
        build_byte_limited_fp(vocab, fingerprint, TrieBuildLimits::default())
    }

    /// Builds the byte-trie from an ALREADY-canonical `PreparedVocabulary`, so a caller that shares
    /// one sorted form with fingerprinting (`VocabularyHandle`) never sorts the token map twice.
    pub(crate) fn build_byte_from_prepared(
        prepared: &PreparedVocabulary,
        fingerprint: VocabFingerprint,
    ) -> Result<Self, CompileError> {
        build_byte_limited_fp_prepared(prepared, fingerprint, TrieBuildLimits::default())
    }

    pub(crate) fn build_byte_filtered_from_prepared(
        prepared: &PreparedVocabulary,
        fingerprint: VocabFingerprint,
        keep: impl Fn(&[u8]) -> bool,
    ) -> Result<Self, CompileError> {
        let limits = TrieBuildLimits::default();
        let mut pairs = BorrowedPairs::new();
        pairs
            .try_reserve_exact(prepared.len())
            .map_err(|_| limit_err("filtered trie record allocation failed"))?;
        pairs.extend(prepared.iter().filter(|(bytes, _)| keep(bytes)));
        check_total_bytes(&pairs, limits)?;
        build_byte_csr(&pairs, fingerprint, limits)
    }

    /// Builds the per-compile class-trie by structural grouping: each token's byte string is mapped
    /// to its class signature, so distinct bytes in one class share an edge and equal signatures
    pub fn build_class(vocab: &Vocabulary, engine: &RefEngine) -> Result<Self, CompileError> {
        build_class_limited(vocab, engine, TrieBuildLimits::default())
    }

    /// The fingerprint of the vocabulary this trie was built from.
    #[must_use]
    pub fn vocab_fingerprint(&self) -> VocabFingerprint {
        self.vocab_fingerprint
    }

    /// The class-partition fingerprint a class-trie was built under (`None` for a byte-trie).
    pub(crate) fn partition(&self) -> Option<[u8; 32]> {
        self.partition
    }

    /// An approximate retained-heap-byte estimate of the packed arrays (by allocated `capacity`, not
    /// `len`), for the cache byte budget. Small fixed struct overhead is not counted.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.first_child.capacity() * 4
            + self.token_head.capacity() * 4
            + self.edge_key.capacity() * 2
            + self.child_node.capacity() * 4
            + self.leaf_tokens.capacity() * 4
    }

    /// The number of trie nodes (the root plus every distinct prefix).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.first_child.len() - 1
    }

    /// The exact maximum DFS-stack size a joint walk over this trie can ever reach - the value
    /// every bind path reserves its walk stack to, instead of the looser `node_count + 1` bound.
    #[must_use]
    pub fn max_dfs_frontier(&self) -> usize {
        self.max_dfs_frontier
    }

    pub(crate) fn kind(&self) -> TrieKind {
        self.kind
    }

    /// The edge keys and child nodes of `n`, an out-of-range node id being a structured error.
    pub(crate) fn children(&self, n: TrieNodeId) -> Result<(&[u16], &[TrieNodeId]), CompileError> {
        let (start, end) = self.range(&self.first_child, n)?;
        Ok((&self.edge_key[start..end], &self.child_node[start..end]))
    }

    /// The token ids ending at `n` (empty when `n` is not terminal); out-of-range is an error.
    pub(crate) fn leaf_tokens_of(&self, n: TrieNodeId) -> Result<&[u32], CompileError> {
        let (start, end) = self.range(&self.token_head, n)?;
        Ok(&self.leaf_tokens[start..end])
    }

    /// Maps an edge key to its DFA byte class: a byte through the class table, a class id directly.
    pub(crate) fn class_of_edge(&self, key: u16, engine: &RefEngine) -> ByteClassId {
        match self.kind {
            TrieKind::Byte => engine.class_of_byte(u8::try_from(key).expect("byte-trie key < 256")),
            TrieKind::Class => ByteClassId(key),
        }
    }

    fn range(&self, offsets: &[u32], n: TrieNodeId) -> Result<(usize, usize), CompileError> {
        let i = n.get() as usize;
        let start = *offsets.get(i).ok_or_else(bad_node)? as usize;
        let end = *offsets.get(i + 1).ok_or_else(bad_node)? as usize;
        Ok((start, end))
    }
}

/// The byte-trie build, bounded. Collects BORROWED token-byte references (fat pointers, not copies)
/// and rejects an oversized vocabulary before touching any byte; the walk maps each byte to its u16
pub(crate) fn build_byte_limited(
    vocab: &Vocabulary,
    limits: TrieBuildLimits,
) -> Result<VocabTrie, CompileError> {
    build_byte_limited_fp(vocab, VocabFingerprint::of(vocab)?, limits)
}

/// The byte-trie build with the fingerprint supplied (not recomputed). Prepares (sorts) the tokens
/// once via `PreparedVocabulary`, the same canonical form fingerprinting reads, then builds via a
pub(crate) fn build_byte_limited_fp(
    vocab: &Vocabulary,
    fingerprint: VocabFingerprint,
    limits: TrieBuildLimits,
) -> Result<VocabTrie, CompileError> {
    if vocab.tokens().len() > limits.max_vocab_entries {
        return Err(limit_err(
            "vocabulary entry count exceeds the trie build limit",
        ));
    }
    let prepared = PreparedVocabulary::build(vocab)?;
    build_byte_limited_fp_prepared(&prepared, fingerprint, limits)
}

/// The byte-trie build from an already-sorted `PreparedVocabulary` - no independent sort here.
fn build_byte_limited_fp_prepared(
    prepared: &PreparedVocabulary,
    fingerprint: VocabFingerprint,
    limits: TrieBuildLimits,
) -> Result<VocabTrie, CompileError> {
    if prepared.len() > limits.max_vocab_entries {
        return Err(limit_err(
            "vocabulary entry count exceeds the trie build limit",
        ));
    }
    let pairs: BorrowedPairs<'_> = prepared.iter().collect();
    check_total_bytes(&pairs, limits)?;
    build_byte_csr(&pairs, fingerprint, limits)
}

/// Collects borrowed `(bytes, ids)` references, rejecting a vocabulary with too many entries before
/// the reference vector is allocated.
fn borrow_pairs(
    vocab: &Vocabulary,
    limits: TrieBuildLimits,
) -> Result<BorrowedPairs<'_>, CompileError> {
    if vocab.tokens().len() > limits.max_vocab_entries {
        return Err(limit_err(
            "vocabulary entry count exceeds the trie build limit",
        ));
    }
    Ok(vocab
        .tokens()
        .iter()
        .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice()))
        .collect())
}

/// The pre-optimization byte-trie builder (per-byte `binary_search` + `Vec::insert`), retained as
/// the differential reference the append-only builder must match byte-for-byte.
#[cfg(test)]
pub(crate) fn build_byte_reference(vocab: &Vocabulary) -> Result<VocabTrie, CompileError> {
    let limits = TrieBuildLimits::default();
    let pairs = borrow_pairs(vocab, limits)?;
    build_from_refs(
        pairs,
        TrieKind::Byte,
        None,
        VocabFingerprint::of(vocab)?,
        limits,
    )
}

/// Rejects a vocabulary whose total token bytes exceed the build limit, before any byte is walked.
fn check_total_bytes(
    pairs: &BorrowedPairs<'_>,
    limits: TrieBuildLimits,
) -> Result<(), CompileError> {
    let mut total_bytes: u64 = 0;
    for (bytes, _) in pairs {
        total_bytes = total_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| limit_err("token byte total overflow"))?;
        if total_bytes > limits.max_total_token_bytes {
            return Err(limit_err("total token bytes exceed the trie build limit"));
        }
    }
    Ok(())
}

/// Length of the shared leading run of two byte slices.
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// True when `ids` is already sorted and duplicate-free (a `PreparedVocabulary` record's invariant).
/// Skips a redundant scratch copy on that path; measured effect on total build time is within noise.
fn is_strictly_ascending(ids: &[u32]) -> bool {
    ids.windows(2).all(|w| w[0] < w[1])
}

/// Builds the byte-trie CSR arrays from lexicographically SORTED pairs via a longest-common-prefix
/// stack, no per-node `Vec`. Byte-identical to the reference builder (see the differential tests).
fn build_byte_csr(
    pairs: &BorrowedPairs<'_>,
    fingerprint: VocabFingerprint,
    limits: TrieBuildLimits,
) -> Result<VocabTrie, CompileError> {
    let mut edge_parent: Vec<u32> = Vec::new();
    let mut edge_key_raw: Vec<u16> = Vec::new();
    let mut edge_child_raw: Vec<u32> = Vec::new();
    let mut leaf_nodes: Vec<u32> = Vec::with_capacity(pairs.len()); // node id per token, ascending
    let mut leaf_slices: Vec<&[u32]> = Vec::with_capacity(pairs.len());
    let mut path: Vec<u32> = vec![0]; // path[d] = node reached after d bytes of the previous token
    let mut prev: &[u8] = &[];
    let mut node_count: u32 = 1;
    let mut leaf_id_total = 0usize;
    for &(bytes, ids) in pairs {
        let lcp = common_prefix_len(prev, bytes);
        path.truncate(lcp + 1);
        let mut cur = path[lcp];
        for &b in &bytes[lcp..] {
            if node_count as usize >= limits.max_nodes {
                return Err(limit_err("trie node count exceeds the build fuel"));
            }
            let new = node_count;
            node_count = node_count
                .checked_add(1)
                .ok_or_else(|| limit_err("trie node count overflow"))?;
            edge_parent.push(cur);
            edge_key_raw.push(u16::from(b));
            edge_child_raw.push(new);
            path.push(new);
            cur = new;
        }
        if !ids.is_empty() {
            leaf_id_total = leaf_id_total
                .checked_add(ids.len())
                .ok_or_else(|| limit_err("leaf token id total overflow"))?;
            if leaf_id_total > limits.max_leaf_token_ids {
                return Err(limit_err(
                    "total leaf token ids exceed the trie build limit",
                ));
            }
            leaf_nodes.push(cur);
            leaf_slices.push(ids);
        }
        prev = bytes;
    }

    let nc = node_count as usize;
    let edges = edge_parent.len();
    let mut first_child = vec![0u32; nc + 1];
    for &p in &edge_parent {
        first_child[p as usize + 1] += 1;
    }
    for i in 0..nc {
        first_child[i + 1] = first_child[i]
            .checked_add(first_child[i + 1])
            .ok_or_else(|| limit_err("trie edge offset overflow"))?;
    }
    let mut cursor: Vec<u32> = first_child[..nc].to_vec();
    let mut edge_key = vec![0u16; edges];
    let mut child_node = vec![TrieNodeId(0); edges];
    for i in 0..edges {
        let slot = &mut cursor[edge_parent[i] as usize];
        let pos = *slot as usize;
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| limit_err("trie edge cursor overflow"))?;
        edge_key[pos] = edge_key_raw[i];
        child_node[pos] = TrieNodeId(edge_child_raw[i]);
    }
    let mut token_head = vec![0u32; nc + 1];
    let mut leaf_tokens: Vec<u32> = Vec::with_capacity(leaf_id_total);
    let mut scratch: Vec<u32> = Vec::new();
    let mut li = 0usize;
    for (node, head) in token_head.iter_mut().enumerate().take(nc) {
        *head = checked_u32(leaf_tokens.len())?;
        if li < leaf_nodes.len() && leaf_nodes[li] as usize == node {
            let slice = leaf_slices[li];
            if is_strictly_ascending(slice) {
                leaf_tokens.extend_from_slice(slice);
            } else {
                scratch.clear();
                scratch.extend_from_slice(slice);
                if sort_enabled() {
                    scratch.sort_unstable();
                }
                scratch.dedup();
                leaf_tokens.extend_from_slice(&scratch);
            }
            li += 1;
        }
    }
    token_head[nc] = checked_u32(leaf_tokens.len())?;
    if li != leaf_nodes.len() {
        return Err(CompileError::new(
            ErrorCode::ArtifactOutOfBounds,
            Stage::L4Bind,
            "trie build left a terminal node unplaced",
        ));
    }

    let max_dfs_frontier = compute_max_dfs_frontier(&first_child, &child_node);
    Ok(VocabTrie {
        first_child,
        token_head,
        edge_key,
        child_node,
        leaf_tokens,
        kind: TrieKind::Byte,
        vocab_fingerprint: fingerprint,
        partition: None,
        max_dfs_frontier,
    })
}

/// The class-trie build, bounded. Same borrowed collection; the byte-to-class mapping is applied
/// lazily during descent (no per-token signature `Vec` is materialized up front).
pub(crate) fn build_class_limited(
    vocab: &Vocabulary,
    engine: &RefEngine,
    limits: TrieBuildLimits,
) -> Result<VocabTrie, CompileError> {
    let pairs = borrow_pairs(vocab, limits)?;
    build_from_refs(
        pairs,
        TrieKind::Class,
        Some(engine),
        VocabFingerprint::of(vocab)?,
        limits,
    )
}

/// Core build from borrowed `(bytes, ids)` refs, sorting by bytes then id for a deterministic node
/// numbering (the final CSR is insertion-order-independent regardless). Every limit is checked
fn build_from_refs(
    mut pairs: BorrowedPairs<'_>,
    kind: TrieKind,
    engine: Option<&RefEngine>,
    vocab_fingerprint: VocabFingerprint,
    limits: TrieBuildLimits,
) -> Result<VocabTrie, CompileError> {
    check_total_bytes(&pairs, limits)?;
    if sort_enabled() {
        pairs.sort_unstable();
    }
    let mut nodes = vec![BuildNode::default()];
    let mut leaf_ids = 0usize;
    for (bytes, ids) in &pairs {
        let leaf = descend(&mut nodes, bytes, kind, engine, limits.max_nodes)?;
        leaf_ids = leaf_ids
            .checked_add(ids.len())
            .ok_or_else(|| limit_err("leaf token id total overflow"))?;
        if leaf_ids > limits.max_leaf_token_ids {
            return Err(limit_err(
                "total leaf token ids exceed the trie build limit",
            ));
        }
        nodes[leaf].tokens.extend_from_slice(ids);
    }
    let partition = engine.map(partition_fingerprint);
    flatten(&nodes, kind, vocab_fingerprint, partition)
}

/// Maps a raw byte to its edge key: itself for a byte-trie, its class id for a class-trie.
fn key_of(byte: u8, kind: TrieKind, engine: Option<&RefEngine>) -> u16 {
    match kind {
        TrieKind::Byte => u16::from(byte),
        TrieKind::Class => engine
            .expect("class-trie build needs an engine")
            .class_of_byte(byte)
            .get(),
    }
}

fn limit_err(msg: &'static str) -> CompileError {
    CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L4Bind, msg)
}

/// A content address of the engine's byte-class partition (the full `of_byte[256]` map, not just
/// the class count), so a class-trie built under one partition cannot be walked under another.
pub(crate) fn partition_fingerprint(engine: &RefEngine) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for b in 0u16..=255 {
        let byte = u8::try_from(b).expect("0..=255 fits u8");
        hasher.update(&engine.class_of_byte(byte).get().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Walks/creates the path for `bytes` (mapped to edge keys lazily) from the root, returning the
/// terminal node index. The node cap is checked BEFORE each allocation so one long token cannot
fn descend(
    nodes: &mut Vec<BuildNode>,
    bytes: &[u8],
    kind: TrieKind,
    engine: Option<&RefEngine>,
    max_nodes: usize,
) -> Result<usize, CompileError> {
    let mut cur = 0;
    for &b in bytes {
        let k = key_of(b, kind, engine);
        match nodes[cur]
            .children
            .binary_search_by_key(&k, |&(key, _)| key)
        {
            Ok(i) => cur = nodes[cur].children[i].1,
            Err(i) => {
                if nodes.len() >= max_nodes {
                    return Err(limit_err("trie node count exceeds the build fuel"));
                }
                let new = nodes.len();
                nodes.push(BuildNode::default());
                nodes[cur].children.insert(i, (k, new));
                cur = new;
            }
        }
    }
    Ok(cur)
}

/// Emits the CSR arrays in DFS-preorder (children ascending), so two builds are byte-identical.
fn flatten(
    nodes: &[BuildNode],
    kind: TrieKind,
    vocab_fingerprint: VocabFingerprint,
    partition: Option<[u8; 32]>,
) -> Result<VocabTrie, CompileError> {
    let mut order = Vec::with_capacity(nodes.len());
    let mut stack = vec![0usize];
    while let Some(b) = stack.pop() {
        order.push(b);
        stack.extend(nodes[b].children.iter().rev().map(|&(_, child)| child));
    }
    let mut final_of = vec![0u32; nodes.len()];
    for (fi, &b) in order.iter().enumerate() {
        final_of[b] = checked_u32(fi)?;
    }

    let mut first_child = Vec::with_capacity(nodes.len() + 1);
    let mut token_head = Vec::with_capacity(nodes.len() + 1);
    let mut edge_key = Vec::new();
    let mut child_node = Vec::new();
    let mut leaf_tokens = Vec::new();
    for &b in &order {
        first_child.push(checked_u32(edge_key.len())?);
        token_head.push(checked_u32(leaf_tokens.len())?);
        for &(k, child) in &nodes[b].children {
            edge_key.push(k);
            child_node.push(TrieNodeId(final_of[child]));
        }
        let mut toks = nodes[b].tokens.clone();
        if sort_enabled() {
            toks.sort_unstable();
        }
        toks.dedup();
        leaf_tokens.extend_from_slice(&toks);
    }
    first_child.push(checked_u32(edge_key.len())?);
    token_head.push(checked_u32(leaf_tokens.len())?);

    let max_dfs_frontier = compute_max_dfs_frontier(&first_child, &child_node);
    Ok(VocabTrie {
        first_child,
        token_head,
        edge_key,
        child_node,
        leaf_tokens,
        kind,
        vocab_fingerprint,
        partition,
        max_dfs_frontier,
    })
}

fn checked_u32(value: usize) -> Result<u32, CompileError> {
    u32::try_from(value).map_err(|_| {
        CompileError::new(
            ErrorCode::InternalLimitExceeded,
            Stage::L4Bind,
            "trie index exceeds the addressable range",
        )
    })
}

fn bad_node() -> CompileError {
    CompileError::new(
        ErrorCode::ArtifactOutOfBounds,
        Stage::L4Bind,
        "trie node id is out of range",
    )
}

#[cfg(test)]
static SORT_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
fn sort_enabled() -> bool {
    !SORT_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(not(test))]
fn sort_enabled() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::build_from_regex;
    use crate::vocab::build_vocabulary;
    use rustc_hash::FxHashMap;

    static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        GUARD.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn vocab(pairs: &[(&[u8], &[u32])]) -> Vocabulary {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        for &(bytes, ids) in pairs {
            map.insert(bytes.to_vec(), ids.to_vec());
        }
        build_vocabulary(9999, map).expect("vocab")
    }

    fn total_tokens(t: &VocabTrie) -> usize {
        (0..t.node_count())
            .map(|i| t.leaf_tokens_of(TrieNodeId(i as u32)).unwrap().len())
            .sum()
    }

    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn random_vocab(seed: u64) -> Vocabulary {
        let mut st = seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1);
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        let mut next_id = 0u32;
        let tokens = 8 + (next_rand(&mut st) % 40) as usize;
        for _ in 0..tokens {
            let len = 1 + (next_rand(&mut st) % 6) as usize;
            let bytes: Vec<u8> = (0..len)
                .map(|_| (next_rand(&mut st) & 0xff) as u8)
                .collect();
            let entry = map.entry(bytes).or_default();
            let ids = 1 + (next_rand(&mut st) % 3) as usize;
            for _ in 0..ids {
                entry.push(next_id); // globally unique ids (Vocabulary rejects a shared id)
                next_id += 1;
            }
        }
        build_vocabulary(9999, map).expect("random vocab")
    }

    /// A phase-timed replica of `build_byte_csr`, so a cold trie build's cost can be attributed to
    /// its actual sub-phases instead of guessed from reading the algorithm: (1) the LCP-walk that
    #[cfg(feature = "huggingface-hub")]
    fn build_byte_csr_phase_timed(
        pairs: &BorrowedPairs<'_>,
        _limits: TrieBuildLimits,
    ) -> (VocabTrie, [f64; 4]) {
        use std::time::Instant;

        let t0 = Instant::now();
        let mut edge_parent: Vec<u32> = Vec::new();
        let mut edge_key_raw: Vec<u16> = Vec::new();
        let mut edge_child_raw: Vec<u32> = Vec::new();
        let mut leaf_nodes: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut leaf_slices: Vec<&[u32]> = Vec::with_capacity(pairs.len());
        let mut path: Vec<u32> = vec![0];
        let mut prev: &[u8] = &[];
        let mut node_count: u32 = 1;
        let mut leaf_id_total = 0usize;
        for &(bytes, ids) in pairs {
            let lcp = common_prefix_len(prev, bytes);
            path.truncate(lcp + 1);
            let mut cur = path[lcp];
            for &b in &bytes[lcp..] {
                let new = node_count;
                node_count += 1;
                edge_parent.push(cur);
                edge_key_raw.push(u16::from(b));
                edge_child_raw.push(new);
                path.push(new);
                cur = new;
            }
            if !ids.is_empty() {
                leaf_id_total += ids.len();
                leaf_nodes.push(cur);
                leaf_slices.push(ids);
            }
            prev = bytes;
        }
        let phase1_walk_us = t0.elapsed().as_secs_f64() * 1e6;

        let t1 = Instant::now();
        let nc = node_count as usize;
        let edges = edge_parent.len();
        let mut first_child = vec![0u32; nc + 1];
        for &p in &edge_parent {
            first_child[p as usize + 1] += 1;
        }
        for i in 0..nc {
            first_child[i + 1] += first_child[i];
        }
        let phase2_count_prefix_sum_us = t1.elapsed().as_secs_f64() * 1e6;

        let t2 = Instant::now();
        let mut cursor: Vec<u32> = first_child[..nc].to_vec();
        let mut edge_key = vec![0u16; edges];
        let mut child_node = vec![TrieNodeId(0); edges];
        for i in 0..edges {
            let slot = &mut cursor[edge_parent[i] as usize];
            let pos = *slot as usize;
            *slot += 1;
            edge_key[pos] = edge_key_raw[i];
            child_node[pos] = TrieNodeId(edge_child_raw[i]);
        }
        let phase3_scatter_us = t2.elapsed().as_secs_f64() * 1e6;

        let t3 = Instant::now();
        let mut token_head = vec![0u32; nc + 1];
        let mut leaf_tokens: Vec<u32> = Vec::with_capacity(leaf_id_total);
        let mut scratch: Vec<u32> = Vec::new();
        let mut li = 0usize;
        for (node, head) in token_head.iter_mut().enumerate().take(nc) {
            *head = leaf_tokens.len() as u32;
            if li < leaf_nodes.len() && leaf_nodes[li] as usize == node {
                let slice = leaf_slices[li];
                if is_strictly_ascending(slice) {
                    leaf_tokens.extend_from_slice(slice);
                } else {
                    scratch.clear();
                    scratch.extend_from_slice(slice);
                    scratch.sort_unstable();
                    scratch.dedup();
                    leaf_tokens.extend_from_slice(&scratch);
                }
                li += 1;
            }
        }
        token_head[nc] = leaf_tokens.len() as u32;
        let phase4_leaf_dedup_us = t3.elapsed().as_secs_f64() * 1e6;

        let max_dfs_frontier = compute_max_dfs_frontier(&first_child, &child_node);
        let trie = VocabTrie {
            first_child,
            token_head,
            edge_key,
            child_node,
            leaf_tokens,
            kind: TrieKind::Byte,
            vocab_fingerprint: VocabFingerprint::from_bytes([0; 32]),
            partition: None,
            max_dfs_frontier,
        };
        (
            trie,
            [
                phase1_walk_us,
                phase2_count_prefix_sum_us,
                phase3_scatter_us,
                phase4_leaf_dedup_us,
            ],
        )
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    #[ignore = "timing; run with --release --features huggingface-hub -- --ignored --nocapture"]
    fn trie_build_phase_breakdown_on_real_tokenizers() {
        fn stats(mut xs: Vec<f64>) -> (f64, f64) {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (xs[xs.len() / 2], xs[xs.len() - 1])
        }
        const N: usize = 30;
        for repo in [
            "openai-community/gpt2",
            "Qwen/Qwen2.5-0.5B",
            "bigscience/bloom",
        ] {
            let vocab = Vocabulary::from_pretrained(repo, None).expect("vocab");
            let prepared = PreparedVocabulary::build(&vocab).expect("prepared");
            let pairs: BorrowedPairs<'_> = prepared.iter().collect();
            let limits = TrieBuildLimits::default();

            let mut phases: [Vec<f64>; 4] = Default::default();
            let mut node_count = 0usize;
            let mut edge_count = 0usize;
            let mut leaf_id_total = 0usize;
            for _ in 0..N {
                let (trie, times) = build_byte_csr_phase_timed(&pairs, limits);
                for (bucket, t) in phases.iter_mut().zip(times) {
                    bucket.push(t);
                }
                node_count = trie.node_count();
                edge_count = trie.edge_key.len();
                leaf_id_total = trie.leaf_tokens.len();
                std::hint::black_box(&trie);
            }
            println!(
                "=== {repo}: {} tokens, {node_count} nodes, {edge_count} edges, {leaf_id_total} leaf ids ===",
                vocab.len()
            );
            let labels = [
                "1_lcp_walk (build every node/edge/leaf)",
                "2_csr_count_prefix_sum",
                "3_csr_scatter",
                "4_leaf_dedup (sort+dedup per terminal)",
            ];
            let mut total_median = 0.0;
            for (label, bucket) in labels.iter().zip(phases.iter()) {
                let (median, max) = stats(bucket.clone());
                total_median += median;
                println!("    {label}: median={median:.1} max={max:.1} us");
            }
            println!("    sum of phase medians: {total_median:.1} us");
        }
    }

    #[test]
    fn append_only_builder_matches_the_reference_on_many_vocabularies() {
        let _g = guard();
        let all_bytes: Vec<(Vec<u8>, Vec<u32>)> =
            (0u32..256).map(|b| (vec![b as u8], vec![b])).collect();
        let all_bytes_refs: Vec<(&[u8], &[u32])> = all_bytes
            .iter()
            .map(|(b, i)| (b.as_slice(), i.as_slice()))
            .collect();
        let cases: Vec<Vocabulary> = vec![
            vocab(&[(b"a", &[0])]),                                            // one token
            vocab(&[(b"a", &[0]), (b"ab", &[1]), (b"abc", &[2])]), // prefix is also terminal
            vocab(&[(b"1", &[0]), (b"12", &[1]), (b"123", &[2])]), // prefix overlap
            vocab(&[(b"ab", &[5, 200])]),                          // one sequence, multiple ids
            vocab(&[(b"z", &[3, 1, 2, 2])]),                       // unsorted + duplicate ids
            vocab(&[(&[0xff], &[7]), (&[0x00], &[8]), (&[0x80, 0xff], &[9])]), // non-UTF-8 bytes
            vocab(&[(b"a", &[100]), (b"b", &[5000]), (b"c", &[9998])]), // sparse ids
            vocab(&all_bytes_refs),                                // all 256 single-byte tokens
            vocab(&[
                (b"aaaaaaaaaaaaaaaa", &[0]),
                (b"aaaaaaaaaaaaaaab", &[1]),
                (b"aaaaaaaaaaaaaaac", &[2]),
            ]), // long shared prefix, one divergent tail byte
        ];
        for v in &cases {
            assert_eq!(
                VocabTrie::build_byte(v).unwrap(),
                build_byte_reference(v).unwrap()
            );
        }
        if let Ok(empty) = build_vocabulary(9999, FxHashMap::default()) {
            assert_eq!(
                VocabTrie::build_byte(&empty).unwrap(),
                build_byte_reference(&empty).unwrap()
            );
        }
        for seed in 0..96u64 {
            let v = random_vocab(seed);
            assert_eq!(
                VocabTrie::build_byte(&v).unwrap(),
                build_byte_reference(&v).unwrap(),
                "append-only and reference tries diverge at seed {seed}"
            );
        }
    }

    #[test]
    fn reversed_insertion_order_builds_an_identical_trie() {
        let _g = guard();
        let fwd = vocab(&[(b"a", &[0]), (b"ab", &[1]), (b"abc", &[2]), (b"b", &[3])]);
        let rev = vocab(&[(b"b", &[3]), (b"abc", &[2]), (b"ab", &[1]), (b"a", &[0])]);
        assert_eq!(
            VocabTrie::build_byte(&fwd).unwrap(),
            VocabTrie::build_byte(&rev).unwrap()
        );
    }

    #[test]
    fn every_child_edge_points_at_a_real_node_with_an_ascending_key_run() {
        let _g = guard();
        let v = vocab(&[
            (b"cat", &[0]),
            (b"car", &[1]),
            (b"carbon", &[2]),
            (b"do", &[3]),
        ]);
        let t = VocabTrie::build_byte(&v).unwrap();
        for n in 0..t.node_count() {
            let (keys, kids) = t.children(TrieNodeId(n as u32)).unwrap();
            for w in keys.windows(2) {
                assert!(w[0] < w[1], "child keys must be strictly ascending");
            }
            for &c in kids {
                assert!((c.get() as usize) < t.node_count(), "child id in range");
                assert!(
                    c.get() as usize > n,
                    "a child id is always greater than its parent"
                );
            }
        }
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    #[ignore = "real tokenizers; run with --features huggingface-hub -- --ignored"]
    fn append_only_builder_matches_the_reference_on_real_tokenizers() {
        let _g = guard();
        for repo in ["openai-community/gpt2", "Qwen/Qwen2.5-0.5B"] {
            let v = Vocabulary::from_pretrained(repo, None).expect("real vocab");
            let new = VocabTrie::build_byte(&v).unwrap();
            let old = build_byte_reference(&v).unwrap();
            assert_eq!(new, old, "append-only vs reference diverge on {repo}");
        }
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    #[ignore = "timing; run with --release --features huggingface-hub -- --ignored --nocapture"]
    fn build_byte_old_vs_new_timing_on_real_tokenizers() {
        use std::time::Instant;
        let _g = guard();
        let median = |mut xs: Vec<u128>| {
            xs.sort_unstable();
            xs[xs.len() / 2]
        };
        for repo in ["openai-community/gpt2", "Qwen/Qwen2.5-0.5B"] {
            let v = Vocabulary::from_pretrained(repo, None).expect("real vocab");
            let iters = 11;
            let mut old_us = Vec::new();
            let mut new_us = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                let o = build_byte_reference(&v).unwrap();
                old_us.push(t.elapsed().as_micros());
                std::hint::black_box(&o);
                let t = Instant::now();
                let n = VocabTrie::build_byte(&v).unwrap();
                new_us.push(t.elapsed().as_micros());
                std::hint::black_box(&n);
            }
            let mut sort_us = Vec::new();
            for _ in 0..iters {
                let mut pairs: BorrowedPairs<'_> = v
                    .tokens()
                    .iter()
                    .map(|(b, i)| (b.as_slice(), i.as_slice()))
                    .collect();
                let t = Instant::now();
                pairs.sort_unstable();
                sort_us.push(t.elapsed().as_micros());
                std::hint::black_box(&pairs);
            }
            let fp = VocabFingerprint::of(&v).unwrap();
            let mut nofp_us = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                let n = VocabTrie::build_byte_with_fingerprint(&v, fp).unwrap();
                nofp_us.push(t.elapsed().as_micros());
                std::hint::black_box(&n);
            }
            let n = VocabTrie::build_byte(&v).unwrap();
            println!(
                "{repo}: old_us={} new_withhash_us={} new_prevalidated_us={} sort_us={} nodes={} edges={} leaves={} heap_bytes={}",
                median(old_us),
                median(new_us),
                median(nofp_us),
                median(sort_us),
                n.node_count(),
                n.edge_key.len(),
                n.leaf_tokens.len(),
                n.heap_bytes()
            );
        }
    }

    #[test]
    fn append_only_builder_stamps_the_supplied_fingerprint() {
        let _g = guard();
        let v = vocab(&[(b"cat", &[0]), (b"car", &[1])]);
        let fp = VocabFingerprint::of(&v).unwrap();
        let a = VocabTrie::build_byte_with_fingerprint(&v, fp).unwrap();
        let b = VocabTrie::build_byte(&v).unwrap();
        assert_eq!(a, b, "prevalidated and generic builds must be identical");
        assert_eq!(a.vocab_fingerprint(), fp);
    }

    #[test]
    fn csr_offsets_have_node_count_plus_one_entries() {
        let _g = guard();
        let t = VocabTrie::build_byte(&vocab(&[(b"cat", &[0]), (b"car", &[1])])).unwrap();
        assert_eq!(t.first_child.len(), t.node_count() + 1);
        assert_eq!(t.token_head.len(), t.node_count() + 1);
        assert_eq!(*t.first_child.last().unwrap() as usize, t.edge_key.len());
        assert_eq!(*t.token_head.last().unwrap() as usize, t.leaf_tokens.len());
    }

    #[test]
    fn shared_prefix_is_one_node_not_two() {
        let _g = guard();
        let t = VocabTrie::build_byte(&vocab(&[(b"cat", &[0]), (b"car", &[1])])).unwrap();
        assert_eq!(t.node_count(), 5);
        assert_eq!(total_tokens(&t), 2);
    }

    #[test]
    fn max_dfs_frontier_exact_on_known_shapes() {
        let _g = guard();
        let single = VocabTrie::build_byte(&vocab(&[(b"a", &[0])])).unwrap();
        assert_eq!(single.max_dfs_frontier(), 1);

        let branch = VocabTrie::build_byte(&vocab(&[(b"cat", &[0]), (b"car", &[1])])).unwrap();
        assert_eq!(branch.max_dfs_frontier(), 2);

        let wide_pairs: Vec<(Vec<u8>, Vec<u32>)> = (b'a'..=b'z')
            .map(|b| (vec![b], vec![u32::from(b)]))
            .collect();
        let wide_refs: Vec<(&[u8], &[u32])> = wide_pairs
            .iter()
            .map(|(b, i)| (b.as_slice(), i.as_slice()))
            .collect();
        let wide = VocabTrie::build_byte(&vocab(&wide_refs)).unwrap();
        assert_eq!(wide.max_dfs_frontier(), 26);
    }

    #[test]
    fn max_dfs_frontier_is_far_tighter_than_node_count_on_a_deep_chain() {
        let _g = guard();
        let long: Vec<u8> = vec![b'a'; 20_000];
        let t = VocabTrie::build_byte(&vocab(&[(&long, &[0])])).unwrap();
        assert_eq!(t.node_count(), 20_001); // root + one node per byte
        assert_eq!(
            t.max_dfs_frontier(),
            1,
            "an unbranching chain never has more than one pending node"
        );
    }

    #[test]
    fn leaf_tokens_are_ascending_and_multi_id_leaves_are_kept() {
        let _g = guard();
        let t = VocabTrie::build_byte(&vocab(&[(b"z", &[3, 1, 2])])).unwrap();
        let leaf = (0..t.node_count())
            .map(|i| TrieNodeId(i as u32))
            .find(|&n| !t.leaf_tokens_of(n).unwrap().is_empty())
            .unwrap();
        assert_eq!(t.leaf_tokens_of(leaf).unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn class_trie_collapses_same_class_bytes_into_one_edge() {
        let _g = guard();
        let engine = build_from_regex("[0-9]+").unwrap();
        let v = vocab(&[(b"1", &[0]), (b"2", &[1]), (b"12", &[2])]);
        let byte = VocabTrie::build_byte(&v).unwrap();
        let class = VocabTrie::build_class(&v, &engine).unwrap();
        assert!(
            class.node_count() < byte.node_count(),
            "class trie {} should have fewer nodes than byte trie {}",
            class.node_count(),
            byte.node_count()
        );
        assert_eq!(total_tokens(&class), 3);
    }

    #[test]
    fn out_of_range_node_is_a_structured_error_not_a_panic() {
        let _g = guard();
        let t = VocabTrie::build_byte(&vocab(&[(b"a", &[0])])).unwrap();
        let forged = TrieNodeId(u32::MAX);
        assert_eq!(
            t.children(forged).unwrap_err().code,
            ErrorCode::ArtifactOutOfBounds
        );
        assert_eq!(
            t.leaf_tokens_of(forged).unwrap_err().code,
            ErrorCode::ArtifactOutOfBounds
        );
    }

    const FP: VocabFingerprint = VocabFingerprint::from_bytes([0; 32]);

    fn refs(
        pairs: Vec<(&[u8], &[u32])>,
        kind: TrieKind,
        engine: Option<&RefEngine>,
        limits: TrieBuildLimits,
    ) -> Result<VocabTrie, CompileError> {
        build_from_refs(pairs, kind, engine, FP, limits)
    }

    #[test]
    fn build_is_deterministic_regardless_of_insertion_order() {
        let _g = guard();
        let (a, b, c) = (b"a".to_vec(), b"b".to_vec(), b"c".to_vec());
        let (i0, i1, i2) = ([0u32], [1u32], [2u32]);
        let limits = TrieBuildLimits::default();
        let forward = refs(
            vec![
                (c.as_slice(), i0.as_slice()),
                (a.as_slice(), i1.as_slice()),
                (b.as_slice(), i2.as_slice()),
            ],
            TrieKind::Byte,
            None,
            limits,
        )
        .unwrap();
        let reversed = refs(
            vec![
                (b.as_slice(), i2.as_slice()),
                (c.as_slice(), i0.as_slice()),
                (a.as_slice(), i1.as_slice()),
            ],
            TrieKind::Byte,
            None,
            limits,
        )
        .unwrap();
        assert_eq!(forward, reversed);
    }

    #[test]
    fn sorting_makes_a_class_merge_order_independent() {
        let _g = guard();
        use std::sync::atomic::Ordering;
        let engine = build_from_regex("[a-z]+").unwrap();
        let (a, b) = (b"a".to_vec(), b"b".to_vec());
        let (i7, i3) = ([7u32], [3u32]);
        let limits = TrieBuildLimits::default();
        let mk = |ab: bool| {
            let pairs = if ab {
                vec![(a.as_slice(), i7.as_slice()), (b.as_slice(), i3.as_slice())]
            } else {
                vec![(b.as_slice(), i3.as_slice()), (a.as_slice(), i7.as_slice())]
            };
            refs(pairs, TrieKind::Class, Some(&engine), limits).unwrap()
        };
        assert_eq!(mk(true), mk(false));

        SORT_DISABLED.store(true, Ordering::Relaxed);
        let (ua, ub) = (mk(true), mk(false));
        SORT_DISABLED.store(false, Ordering::Relaxed);
        assert_ne!(
            ua.leaf_tokens, ub.leaf_tokens,
            "without sorting, a merged class leaf is insertion-order dependent"
        );
    }

    #[test]
    fn build_fuel_trips_on_many_short_tokens_exceeding_the_node_cap() {
        let _g = guard();
        let bytes: Vec<Vec<u8>> = (0u8..40).map(|b| vec![b]).collect();
        let ids: Vec<Vec<u32>> = (0u32..40).map(|i| vec![i]).collect();
        let pairs: Vec<(&[u8], &[u32])> = bytes
            .iter()
            .zip(&ids)
            .map(|(b, i)| (b.as_slice(), i.as_slice()))
            .collect();
        let limits = TrieBuildLimits {
            max_nodes: 8,
            ..Default::default()
        };
        let err = refs(pairs, TrieKind::Byte, None, limits)
            .expect_err("a small node cap must trip the build fuel");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn build_fuel_trips_mid_token_on_one_long_token_before_allocating_it_all() {
        let _g = guard();
        let long = vec![0u8; 10_000];
        let ids = [0u32];
        let limits = TrieBuildLimits {
            max_nodes: 8,
            ..Default::default()
        };
        let err = refs(
            vec![(long.as_slice(), ids.as_slice())],
            TrieKind::Byte,
            None,
            limits,
        )
        .expect_err("a long token must trip the fuel mid-descent");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn build_rejects_a_vocabulary_exceeding_the_total_token_bytes_limit() {
        let _g = guard();
        let big = vec![0u8; 100];
        let ids = [0u32];
        let limits = TrieBuildLimits {
            max_total_token_bytes: 50,
            ..Default::default()
        };
        let err = refs(
            vec![(big.as_slice(), ids.as_slice())],
            TrieKind::Byte,
            None,
            limits,
        )
        .expect_err("total token bytes over the limit must be rejected early");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn build_rejects_a_vocabulary_exceeding_the_entry_limit() {
        let _g = guard();
        let v = vocab(&[(b"a", &[0]), (b"b", &[1]), (b"c", &[2])]);
        let limits = TrieBuildLimits {
            max_vocab_entries: 2,
            ..Default::default()
        };
        let err = build_byte_limited(&v, limits)
            .expect_err("more entries than the cap must be rejected before allocating pairs");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn build_rejects_a_vocabulary_exceeding_the_leaf_token_id_limit() {
        let _g = guard();
        let a = b"a".to_vec();
        let ids: Vec<u32> = (0..100).collect();
        let limits = TrieBuildLimits {
            max_leaf_token_ids: 10,
            ..Default::default()
        };
        let err = refs(
            vec![(a.as_slice(), ids.as_slice())],
            TrieKind::Byte,
            None,
            limits,
        )
        .expect_err("total leaf ids over the limit must be rejected");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn a_byte_trie_carries_its_vocabulary_fingerprint_and_no_partition() {
        let _g = guard();
        let v = vocab(&[(b"a", &[0]), (b"b", &[1])]);
        let t = VocabTrie::build_byte(&v).unwrap();
        assert_eq!(t.vocab_fingerprint(), VocabFingerprint::of(&v).unwrap());
        assert!(t.partition().is_none());
    }

    #[test]
    fn a_class_trie_carries_the_engine_partition_fingerprint() {
        let _g = guard();
        let engine = build_from_regex("[0-9]+").unwrap();
        let v = vocab(&[(b"1", &[0])]);
        let t = VocabTrie::build_class(&v, &engine).unwrap();
        assert_eq!(t.partition(), Some(partition_fingerprint(&engine)));
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    #[ignore = "network: downloads the real GPT-2 tokenizer"]
    fn a_tiny_node_cap_trips_on_the_real_gpt2_vocabulary() {
        let _g = guard();
        let v = Vocabulary::from_pretrained("openai-community/gpt2", None).expect("gpt2 vocab");
        let tiny = TrieBuildLimits {
            max_nodes: 100,
            ..Default::default()
        };
        let err =
            build_byte_limited(&v, tiny).expect_err("a 100-node cap must trip on 50k+ tokens");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert!(
            VocabTrie::build_byte(&v).is_ok(),
            "default limits still succeed"
        );
    }

    #[test]
    fn building_from_a_prepared_vocabulary_matches_the_direct_vocab_path() {
        let _g = guard();
        let all_bytes: Vec<(Vec<u8>, Vec<u32>)> =
            (0u32..256).map(|b| (vec![b as u8], vec![b])).collect();
        let all_bytes_refs: Vec<(&[u8], &[u32])> = all_bytes
            .iter()
            .map(|(b, i)| (b.as_slice(), i.as_slice()))
            .collect();
        let mut cases: Vec<Vocabulary> = vec![
            vocab(&[(b"a", &[0])]),
            vocab(&[(b"a", &[0]), (b"ab", &[1]), (b"abc", &[2])]),
            vocab(&[(b"1", &[0]), (b"12", &[1]), (b"123", &[2])]),
            vocab(&[(b"ab", &[5, 200])]),
            vocab(&[(b"z", &[3, 1, 2, 2])]), // unsorted + duplicate ids
            vocab(&[(&[0xff], &[7]), (&[0x00], &[8]), (&[0x80, 0xff], &[9])]),
            vocab(&[(b"a", &[100]), (b"b", &[5000]), (b"c", &[9998])]),
            vocab(&all_bytes_refs),
        ];
        for seed in 0u64..64 {
            cases.push(random_vocab(seed));
        }
        for v in &cases {
            let fp = VocabFingerprint::of(v).unwrap();
            let direct = VocabTrie::build_byte_with_fingerprint(v, fp).unwrap();
            let prepared = crate::vocab::PreparedVocabulary::build(v).unwrap();
            let from_prepared = VocabTrie::build_byte_from_prepared(&prepared, fp).unwrap();
            assert_eq!(direct, from_prepared, "trie mismatch for {v:?}");
        }
    }
}
