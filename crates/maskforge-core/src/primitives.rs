//! The newtype spine: zero-cost, transparent ids that never appear as a bare integer in new code.

use std::num::TryFromIntError;

use bincode::{Decode, Encode};

/// Builds a `#[repr(transparent)]` newtype over `$inner` with a checked `TryFrom<usize>`.
macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident($inner:ty)) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Encode, Decode)]
        pub struct $name(pub $inner);

        impl $name {
            /// Returns the wrapped integer.
            #[inline]
            #[must_use]
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl From<$name> for $inner {
            #[inline]
            fn from(id: $name) -> $inner {
                id.0
            }
        }

        impl TryFrom<usize> for $name {
            type Error = TryFromIntError;
            /// Fails (never truncates) when `value` exceeds the inner integer's range.
            #[inline]
            fn try_from(value: usize) -> Result<Self, Self::Error> {
                <$inner>::try_from(value).map(Self)
            }
        }
    };
}

id_newtype!(
    /// Index of a state in the reference engine (dense `0..n`; `StateId(0)` is canonical DEAD).
    StateId(u32)
);

impl StateId {
    /// Reinterprets caller-owned raw ids as `StateId` values without allocation or copying.
    /// Avoids rebuilding a `Vec<StateId>` for batch mask queries.
    #[inline]
    #[must_use]
    pub fn cast_slice(raw: &[u32]) -> &[StateId] {
        // SAFETY: `StateId` is `repr(transparent)` over `u32` with identical layout.
        // Every `u32` bit pattern is a valid `StateId`.
        unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<StateId>(), raw.len()) }
    }
}
id_newtype!(
    /// Identifier of a vocabulary token.
    TokenId(u32)
);
id_newtype!(
    /// Identifier of a minterm (a byte-equivalence class in a collapsed alphabet).
    MintermId(u32)
);
id_newtype!(
    /// Index of a byte class into `ClassTable`.
    ByteClassId(u16)
);
id_newtype!(
    /// Index of a node in the arena IR. `NodeId(0)` is a valid node; range is checked on decode.
    NodeId(u32)
);
id_newtype!(
    /// Index of a node in a vocabulary trie. `TrieNodeId(0)` is the root.
    TrieNodeId(u32)
);

#[cfg(test)]
mod tests {
    use std::mem::{align_of, size_of};

    use super::*;

    #[test]
    fn newtypes_are_zero_cost() {
        assert_eq!(size_of::<StateId>(), size_of::<u32>());
        assert_eq!(align_of::<StateId>(), align_of::<u32>());
        assert_eq!(size_of::<TokenId>(), size_of::<u32>());
        assert_eq!(size_of::<MintermId>(), size_of::<u32>());
        assert_eq!(size_of::<NodeId>(), size_of::<u32>());
        assert_eq!(align_of::<NodeId>(), align_of::<u32>());
        assert_eq!(size_of::<ByteClassId>(), size_of::<u16>());
        assert_eq!(align_of::<ByteClassId>(), align_of::<u16>());
    }

    #[test]
    fn cast_slice_matches_element_by_element_construction_with_no_reallocation() {
        let raw = [0u32, 7, 42, u32::MAX];
        let cast = StateId::cast_slice(&raw);
        let built: Vec<StateId> = raw.iter().copied().map(StateId).collect();
        assert_eq!(cast, &built[..]);
        assert_eq!(
            cast.as_ptr().cast::<u32>(),
            raw.as_ptr(),
            "cast_slice must reinterpret raw's own allocation, not build a new one"
        );
    }

    #[test]
    fn cast_slice_of_empty_is_empty() {
        let raw: [u32; 0] = [];
        assert!(StateId::cast_slice(&raw).is_empty());
    }

    #[test]
    fn checked_conversion_fails_out_of_range() {
        assert_eq!(TokenId::try_from(7usize).map(TokenId::get), Ok(7));
        if let Some(too_big) = usize::try_from(u32::MAX)
            .ok()
            .and_then(|m| m.checked_add(1))
        {
            assert!(TokenId::try_from(too_big).is_err());
        }
        assert!(ByteClassId::try_from(usize::from(u16::MAX) + 1).is_err());
        if let Some(too_big) = usize::try_from(u32::MAX)
            .ok()
            .and_then(|m| m.checked_add(1))
        {
            assert!(NodeId::try_from(too_big).is_err());
        }
    }
}
