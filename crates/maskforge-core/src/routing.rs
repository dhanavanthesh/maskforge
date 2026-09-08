//! Shared, bounded execution-cost analysis for regular and structured compilation.

use crate::ir::{AdditionalPolicy, ContainsPolicy, ItemsPolicy, Node, RefSlice, SchemaIR};
use crate::primitives::NodeId;

/// Increment when a routing rule or cost model changes executable selection.
pub const ROUTE_POLICY_VERSION: u32 = 5;

const STATE_CAP: u64 = 1 << 16;
const CELL_CAP: u64 = 1 << 22;
const MEMORY_CAP: u64 = 256 << 20;

/// A bounded estimate that never wraps attacker-controlled arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bound {
    /// The value is known and no greater than the estimator's cap.
    Known(u64),
    /// The value overflowed or exceeded the estimator's cap.
    ExceedsLimit,
}

/// Adds two bounded values, short-circuiting on overflow or cap violation.
#[must_use]
pub fn add_bound(a: Bound, b: Bound, cap: u64) -> Bound {
    match (a, b) {
        (Bound::Known(a), Bound::Known(b)) => a
            .checked_add(b)
            .filter(|sum| *sum <= cap)
            .map_or(Bound::ExceedsLimit, Bound::Known),
        _ => Bound::ExceedsLimit,
    }
}

/// Multiplies two bounded values, short-circuiting on overflow or cap violation.
#[must_use]
pub fn mul_bound(a: Bound, b: Bound, cap: u64) -> Bound {
    match (a, b) {
        (Bound::Known(a), Bound::Known(b)) => a
            .checked_mul(b)
            .filter(|product| *product <= cap)
            .map_or(Bound::ExceedsLimit, Bound::Known),
        _ => Bound::ExceedsLimit,
    }
}

/// Computes `2^n` under a cap without an overflowing shift.
#[must_use]
pub fn pow2_bound(n: u32, cap: u64) -> Bound {
    1u64.checked_shl(n)
        .filter(|value| *value <= cap)
        .map_or(Bound::ExceedsLimit, Bound::Known)
}

/// Runtime shape selected for one IR node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionKind {
    /// Compile the complete subtree as one regular byte automaton.
    Regular,
    /// Execute the node and its descendants with structured semantics.
    Structured,
    /// Execute a structured node while retaining regular compiled descendants.
    Hybrid,
}

/// Stable primary reason for a node's selected route.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RouteReason {
    /// The bounded regular estimate is below all construction limits.
    CheapRegular,
    /// Static or dynamic reference scope requires structured execution.
    DynamicReference,
    /// Evaluation annotations must be preserved across applicators.
    EvaluationAnnotations,
    /// Open-object matching needs decoded-key structured semantics.
    OpenObject,
    /// A user pattern runs over decoded Unicode scalar values.
    UserPattern,
    /// The conservative shuffle estimate exceeds a regular construction budget.
    ShuffleCost,
    /// The conservative product estimate exceeds a regular construction budget.
    ProductCost,
    /// A repetition estimate exceeds a regular construction budget.
    RepetitionCost,
    /// Estimated retained or peak automaton memory exceeds its budget.
    AutomatonMemory,
    /// The keyword's semantics are implemented only by the structured engine.
    StructuredOnlySemantics,
}

/// Bounded construction and memory estimate for one execution strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostEstimate {
    /// Estimated reachable states or structured plan nodes.
    pub states: Bound,
    /// Estimated transition cells or structured graph edges.
    pub transition_cells: Bound,
    /// Estimated construction work units.
    pub work: Bound,
    /// Estimated retained bytes.
    pub retained_bytes: Bound,
    /// Estimated peak construction bytes.
    pub peak_bytes: Bound,
}

impl CostEstimate {
    fn regular(states: Bound, cells: Bound) -> Self {
        let retained = add_bound(
            mul_bound(states, Bound::Known(64), MEMORY_CAP),
            mul_bound(cells, Bound::Known(8), MEMORY_CAP),
            MEMORY_CAP,
        );
        let peak = add_bound(
            retained,
            mul_bound(states, Bound::Known(96), MEMORY_CAP),
            MEMORY_CAP,
        );
        Self {
            states,
            transition_cells: cells,
            work: cells,
            retained_bytes: retained,
            peak_bytes: peak,
        }
    }

    fn structured(child_count: usize) -> Self {
        let nodes = u64::try_from(child_count)
            .ok()
            .and_then(|count| count.checked_add(1))
            .map_or(Bound::ExceedsLimit, Bound::Known);
        Self {
            states: nodes,
            transition_cells: nodes,
            work: nodes,
            retained_bytes: mul_bound(nodes, Bound::Known(128), MEMORY_CAP),
            peak_bytes: mul_bound(nodes, Bound::Known(256), MEMORY_CAP),
        }
    }
}

/// One immutable route-table entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeRoute {
    /// Selected execution shape.
    pub kind: ExecutionKind,
    /// Stable primary route reason.
    pub reason: RouteReason,
    /// Conservative regular construction estimate.
    pub regular: CostEstimate,
    /// Conservative structured construction estimate.
    pub structured: CostEstimate,
}

/// The immutable per-node route analysis computed once for a frozen IR.
#[derive(Clone, Debug)]
pub struct RouteTable {
    routes: Box<[NodeRoute]>,
}

impl RouteTable {
    /// Route for `node`, if it belongs to this IR.
    #[must_use]
    pub fn get(&self, node: NodeId) -> Option<&NodeRoute> {
        self.routes.get(node.get() as usize)
    }

    /// All routes in arena order.
    #[must_use]
    pub fn routes(&self) -> &[NodeRoute] {
        &self.routes
    }
}

pub(crate) fn analyze(ir: &SchemaIR) -> RouteTable {
    let mut routes = vec![None; ir.node_count()];
    let mut visit = vec![0u8; ir.node_count()];
    let mut stack = Vec::with_capacity(ir.node_count().saturating_mul(2));
    for index in (0..ir.node_count()).rev() {
        stack.push((NodeId(u32::try_from(index).unwrap_or(u32::MAX)), false));
    }
    while let Some((id, done)) = stack.pop() {
        let index = id.get() as usize;
        if index >= visit.len() {
            continue;
        }
        if done {
            routes[index] = ir.node(id).map(|node| estimate_node(ir, node, &routes));
            visit[index] = 2;
            continue;
        }
        if visit[index] != 0 {
            continue;
        }
        visit[index] = 1;
        stack.push((id, true));
        for_each_child(ir, ir.node(id), |child| {
            let child_index = child.get() as usize;
            if visit.get(child_index) == Some(&0) {
                stack.push((child, false));
            }
        });
    }
    let fallback = NodeRoute {
        kind: ExecutionKind::Structured,
        reason: RouteReason::StructuredOnlySemantics,
        regular: CostEstimate::regular(Bound::ExceedsLimit, Bound::ExceedsLimit),
        structured: CostEstimate::structured(0),
    };
    RouteTable {
        routes: routes
            .into_iter()
            .map(|route| route.unwrap_or(fallback))
            .collect(),
    }
}

fn estimate_node(ir: &SchemaIR, node: &Node, routes: &[Option<NodeRoute>]) -> NodeRoute {
    let mut children = Vec::new();
    for_each_child(ir, Some(node), |child| children.push(child));
    let child_routes: Vec<NodeRoute> = children
        .iter()
        .filter_map(|child| routes.get(child.get() as usize).and_then(|route| *route))
        .collect();
    let unresolved_child = child_routes.len() != children.len();
    let (regular, cost_reason) = regular_cost(ir, node, &child_routes);
    let local_reason = local_structured_reason(ir, node, regular);
    let child_non_regular = child_routes
        .iter()
        .find(|route| route.kind != ExecutionKind::Regular);
    let (kind, reason) = if let Some(reason) = local_reason {
        let has_regular_child = child_routes
            .iter()
            .any(|route| route.kind == ExecutionKind::Regular);
        (
            if has_regular_child {
                ExecutionKind::Hybrid
            } else {
                ExecutionKind::Structured
            },
            reason,
        )
    } else if unresolved_child {
        (ExecutionKind::Hybrid, RouteReason::DynamicReference)
    } else if let Some(child) = child_non_regular {
        (ExecutionKind::Hybrid, child.reason)
    } else if cost_reason != RouteReason::CheapRegular {
        (ExecutionKind::Hybrid, cost_reason)
    } else {
        (ExecutionKind::Regular, RouteReason::CheapRegular)
    };
    NodeRoute {
        kind,
        reason,
        regular,
        structured: CostEstimate::structured(children.len()),
    }
}

fn local_structured_reason(
    ir: &SchemaIR,
    node: &Node,
    regular: CostEstimate,
) -> Option<RouteReason> {
    match node {
        Node::OpenObject { .. } => Some(RouteReason::OpenObject),
        Node::StringPattern { .. } if ir.is_user_pattern(node) => Some(RouteReason::UserPattern),
        Node::StringPattern { .. } if ir.is_unrolled_codepoint_string(node) => {
            Some(RouteReason::RepetitionCost)
        }
        Node::Object {
            fields,
            dependent,
            dependent_required,
            ..
        } => {
            if ir
                .props_at(*dependent)
                .is_some_and(|items| !items.is_empty())
                || ir
                    .dependent_required_at(*dependent_required)
                    .is_some_and(|items| !items.is_empty())
            {
                Some(RouteReason::EvaluationAnnotations)
            } else if ir
                .props_at(*fields)
                .is_some_and(|items| items.len() > crate::ir::MAX_SAFE_SHUFFLE_FIELDS)
                || regular.states == Bound::ExceedsLimit
                || regular.transition_cells == Bound::ExceedsLimit
            {
                Some(RouteReason::ShuffleCost)
            } else {
                None
            }
        }
        Node::Array {
            unique_items,
            items,
            contains,
            min_items,
            max_items,
            ..
        } => {
            if *unique_items || matches!(items, ItemsPolicy::AllowAny) || contains.is_some() {
                Some(RouteReason::StructuredOnlySemantics)
            } else if *min_items > crate::ir::MAX_UNROLLED_ARRAY_ITEMS
                || max_items.is_some_and(|max| max > crate::ir::MAX_UNROLLED_ARRAY_ITEMS)
                || regular.states == Bound::ExceedsLimit
            {
                Some(RouteReason::RepetitionCost)
            } else {
                None
            }
        }
        Node::Tuple {
            unique_items,
            contains,
            min_items,
            max_items,
            ..
        } => {
            if *unique_items || contains.is_some() {
                Some(RouteReason::StructuredOnlySemantics)
            } else if *min_items > crate::ir::MAX_UNROLLED_ARRAY_ITEMS
                || max_items.is_some_and(|max| max > crate::ir::MAX_UNROLLED_ARRAY_ITEMS)
                || regular.states == Bound::ExceedsLimit
            {
                Some(RouteReason::RepetitionCost)
            } else {
                None
            }
        }
        Node::Intersection { branches } | Node::ExactlyOne { branches }
            if combinator_has_expensive_branch(ir, *branches)
                || regular.states == Bound::ExceedsLimit
                || regular.transition_cells == Bound::ExceedsLimit =>
        {
            Some(RouteReason::ProductCost)
        }
        Node::Number {
            integer_only,
            minimum,
            maximum,
            multiple_of,
        } if *integer_only || minimum.is_some() || maximum.is_some() || multiple_of.is_some() => {
            Some(RouteReason::StructuredOnlySemantics)
        }
        Node::Not { .. } | Node::Unsupported { .. } => Some(RouteReason::StructuredOnlySemantics),
        Node::Ref { .. } | Node::DynamicRef { .. } => Some(RouteReason::DynamicReference),
        Node::Unevaluated { .. } => Some(RouteReason::EvaluationAnnotations),
        _ if regular.retained_bytes == Bound::ExceedsLimit
            || regular.peak_bytes == Bound::ExceedsLimit =>
        {
            Some(RouteReason::AutomatonMemory)
        }
        _ => None,
    }
}

fn regular_cost(ir: &SchemaIR, node: &Node, children: &[NodeRoute]) -> (CostEstimate, RouteReason) {
    if ir.is_large_string_enum(node) {
        let states = match node {
            Node::Enum { values } => ir
                .lits_at(*values)
                .and_then(|values| u64::try_from(values.len()).ok())
                .map_or(Bound::ExceedsLimit, |count| {
                    add_bound(Bound::Known(count), Bound::Known(1), STATE_CAP)
                }),
            _ => Bound::ExceedsLimit,
        };
        let cells = mul_bound(states, Bound::Known(1), CELL_CAP);
        return (
            CostEstimate::regular(states, cells),
            RouteReason::CheapRegular,
        );
    }
    let child_states = |index: usize| {
        children
            .get(index)
            .map_or(Bound::Known(1), |route| route.regular.states)
    };
    let states = match node {
        Node::Union { .. } => children.iter().fold(Bound::Known(1), |total, child| {
            add_bound(total, child.regular.states, STATE_CAP)
        }),
        Node::Intersection { .. } | Node::ExactlyOne { .. } => {
            children.iter().fold(Bound::Known(1), |total, child| {
                mul_bound(total, child.regular.states, STATE_CAP)
            })
        }
        Node::StringPattern {
            max_len: Some(max),
            charset: crate::ir::Charset::Utf8CountedCodepoints,
            ..
        } => add_bound(
            mul_bound(Bound::Known(u64::from(*max)), Bound::Known(8), STATE_CAP),
            Bound::Known(16),
            STATE_CAP,
        ),
        Node::StringPattern { max_len: None, .. } => Bound::Known(64),
        Node::LexicalNumber { regex } => ir
            .str_at(*regex)
            .and_then(|regex| u64::try_from(regex.len()).ok())
            .map_or(Bound::ExceedsLimit, |bytes| {
                add_bound(Bound::Known(bytes), Bound::Known(16), STATE_CAP)
            }),
        Node::Object { fields, .. } => {
            let count = ir.props_at(*fields).map_or(0usize, <[_]>::len);
            if count == 0 {
                return (
                    CostEstimate::regular(Bound::Known(1), Bound::Known(64)),
                    RouteReason::CheapRegular,
                );
            }
            let count_u32 = u32::try_from(count).unwrap_or(u32::MAX);
            let subsets = pow2_bound(count_u32, STATE_CAP);
            let half = count_u32
                .checked_sub(1)
                .map_or(Bound::Known(1), |n| pow2_bound(n, STATE_CAP));
            let sum = ir
                .props_at(*fields)
                .unwrap_or(&[])
                .iter()
                .fold(Bound::Known(0), |total, (_, child)| {
                    add_bound(total, shuffle_child_states(ir, *child), STATE_CAP)
                });
            add_bound(subsets, mul_bound(half, sum, STATE_CAP), STATE_CAP)
        }
        Node::Array { max_items, .. } | Node::Tuple { max_items, .. } => max_items.map_or_else(
            || add_bound(child_states(0), Bound::Known(16), STATE_CAP),
            |max| {
                add_bound(
                    Bound::Known(1),
                    mul_bound(Bound::Known(u64::from(max)), child_states(0), STATE_CAP),
                    STATE_CAP,
                )
            },
        ),
        Node::Enum { values } => ir.lits_at(*values).map_or(Bound::ExceedsLimit, |values| {
            values.iter().fold(Bound::Known(1), |total, value| {
                let bytes = match value {
                    crate::ir::ScalarLit::Str(value) => {
                        u64::try_from(value.len()).unwrap_or(u64::MAX)
                    }
                    _ => 16,
                };
                add_bound(total, Bound::Known(bytes.saturating_add(1)), STATE_CAP)
            })
        }),
        _ => Bound::Known(32),
    };
    let cells = if matches!(node, Node::Intersection { .. } | Node::ExactlyOne { .. }) {
        mul_bound(states, Bound::Known(256), 1 << 24)
    } else if matches!(node, Node::Object { fields, .. } if ir.props_at(*fields).is_some_and(|fields| !fields.is_empty()))
    {
        mul_bound(states, Bound::Known(256), 1 << 20)
    } else {
        mul_bound(states, Bound::Known(64), CELL_CAP)
    };
    let reason = if matches!(node, Node::Intersection { .. } | Node::ExactlyOne { .. })
        && (states == Bound::ExceedsLimit || cells == Bound::ExceedsLimit)
    {
        RouteReason::ProductCost
    } else if matches!(node, Node::Object { .. })
        && (states == Bound::ExceedsLimit || cells == Bound::ExceedsLimit)
    {
        RouteReason::ShuffleCost
    } else if states == Bound::ExceedsLimit || cells == Bound::ExceedsLimit {
        RouteReason::AutomatonMemory
    } else {
        RouteReason::CheapRegular
    };
    (CostEstimate::regular(states, cells), reason)
}

fn shuffle_child_states(ir: &SchemaIR, id: NodeId) -> Bound {
    match ir.node(id) {
        Some(Node::StringPattern {
            regex,
            max_len: Some(max),
            ..
        }) => add_bound(
            mul_bound(
                Bound::Known(u64::from(*max)),
                Bound::Known(if ir.str_at(*regex).is_some_and(|regex| regex.len() <= 8) {
                    8
                } else {
                    96
                }),
                STATE_CAP,
            ),
            Bound::Known(16),
            STATE_CAP,
        ),
        Some(Node::StringPattern { max_len: None, .. }) => Bound::Known(512),
        Some(Node::Union { branches }) => ir
            .refs_at(*branches)
            .unwrap_or(&[])
            .iter()
            .map(|child| shuffle_child_states(ir, *child))
            .max_by_key(|bound| match bound {
                Bound::Known(value) => *value,
                Bound::ExceedsLimit => u64::MAX,
            })
            .unwrap_or(Bound::Known(1)),
        Some(Node::Enum { values }) => ir.lits_at(*values).map_or(Bound::ExceedsLimit, |values| {
            let source_bytes = values.iter().fold(Bound::Known(1), |total, value| {
                let bytes = match value {
                    crate::ir::ScalarLit::Str(value) => {
                        u64::try_from(value.len()).unwrap_or(u64::MAX)
                    }
                    _ => 16,
                };
                add_bound(total, Bound::Known(bytes.saturating_add(1)), STATE_CAP)
            });
            mul_bound(source_bytes, Bound::Known(8), STATE_CAP)
        }),
        Some(Node::Boolean | Node::Null | Node::Never) => Bound::Known(1),
        Some(Node::Integer { .. } | Node::Number { .. }) => Bound::Known(512),
        Some(Node::Array {
            items: ItemsPolicy::Schema(child),
            ..
        }) => add_bound(
            shuffle_child_states(ir, *child),
            Bound::Known(32),
            STATE_CAP,
        ),
        Some(Node::Tuple { prefix, tail, .. }) => {
            let prefix_cost = ir
                .refs_at(*prefix)
                .unwrap_or(&[])
                .iter()
                .fold(Bound::Known(32), |total, child| {
                    add_bound(total, shuffle_child_states(ir, *child), STATE_CAP)
                });
            tail.map_or(prefix_cost, |child| {
                add_bound(prefix_cost, shuffle_child_states(ir, child), STATE_CAP)
            })
        }
        Some(Node::LexicalNumber { regex }) => ir
            .str_at(*regex)
            .and_then(|regex| u64::try_from(regex.len()).ok())
            .map_or(Bound::ExceedsLimit, Bound::Known),
        Some(_) => Bound::Known(2048),
        None => Bound::ExceedsLimit,
    }
}

fn combinator_has_expensive_branch(ir: &SchemaIR, branches: RefSlice) -> bool {
    ir.refs_at(branches).is_some_and(|branches| {
        branches.iter().any(|branch| {
            ir.node(*branch).is_some_and(|node| match node {
                Node::Integer {
                    multiple_of: Some(_),
                    ..
                }
                | Node::Number {
                    multiple_of: Some(_),
                    ..
                } => true,
                Node::Object { fields, .. } => {
                    ir.props_at(*fields).is_some_and(|fields| fields.len() >= 2)
                }
                Node::OpenObject { known, .. } => {
                    ir.props_at(*known).is_some_and(|fields| fields.len() >= 2)
                }
                Node::Array { max_items, .. } | Node::Tuple { max_items, .. } => {
                    max_items.is_none_or(|max| max > 5)
                }
                _ => ir.is_large_string_enum(node),
            })
        })
    })
}

fn for_each_child(ir: &SchemaIR, node: Option<&Node>, mut visit: impl FnMut(NodeId)) {
    match node {
        Some(Node::Object {
            fields, dependent, ..
        }) => {
            for &(_, child) in ir.props_at(*fields).unwrap_or(&[]) {
                visit(child);
            }
            for &(_, child) in ir.props_at(*dependent).unwrap_or(&[]) {
                visit(child);
            }
        }
        Some(Node::OpenObject {
            known,
            patterns,
            additional,
            property_names,
            dependent,
            ..
        }) => {
            for &(_, child) in ir.props_at(*known).unwrap_or(&[]) {
                visit(child);
            }
            for &(_, child) in ir.props_at(*patterns).unwrap_or(&[]) {
                visit(child);
            }
            if let AdditionalPolicy::Schema(child) = additional {
                visit(*child);
            }
            if let Some(child) = property_names {
                visit(*child);
            }
            for &(_, child) in ir.props_at(*dependent).unwrap_or(&[]) {
                visit(child);
            }
        }
        Some(Node::Array {
            items, contains, ..
        }) => {
            if let ItemsPolicy::Schema(child) = items {
                visit(*child);
            }
            if let Some(contains) = contains {
                if let ContainsPolicy::Schema(child) = contains.policy {
                    visit(child);
                }
            }
        }
        Some(Node::Tuple {
            prefix,
            tail,
            contains,
            ..
        }) => {
            for &child in ir.refs_at(*prefix).unwrap_or(&[]) {
                visit(child);
            }
            if let Some(child) = tail {
                visit(*child);
            }
            if let Some(contains) = contains {
                if let ContainsPolicy::Schema(child) = contains.policy {
                    visit(child);
                }
            }
        }
        Some(Node::Union { branches })
        | Some(Node::Intersection { branches })
        | Some(Node::ExactlyOne { branches }) => {
            for &child in ir.refs_at(*branches).unwrap_or(&[]) {
                visit(child);
            }
        }
        Some(Node::Not { inner }) => visit(*inner),
        Some(Node::Ref { def }) => {
            if let Some(child) = ir.def_target(*def) {
                visit(child);
            }
        }
        Some(Node::DynamicRef { initial_target, .. }) => visit(*initial_target),
        Some(Node::Unevaluated {
            scope, unevaluated, ..
        }) => {
            visit(*scope);
            visit(*unevaluated);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_arithmetic_closes_overflow_and_boundaries() {
        assert_eq!(
            add_bound(Bound::Known(4), Bound::Known(5), 9),
            Bound::Known(9)
        );
        assert_eq!(
            add_bound(Bound::Known(4), Bound::Known(6), 9),
            Bound::ExceedsLimit
        );
        assert_eq!(
            mul_bound(Bound::Known(3), Bound::Known(4), 12),
            Bound::Known(12)
        );
        assert_eq!(
            mul_bound(Bound::Known(u64::MAX), Bound::Known(2), u64::MAX),
            Bound::ExceedsLimit
        );
        assert_eq!(pow2_bound(16, 1 << 16), Bound::Known(1 << 16));
        assert_eq!(pow2_bound(17, 1 << 16), Bound::ExceedsLimit);
        assert_eq!(pow2_bound(64, u64::MAX), Bound::ExceedsLimit);
    }
}
