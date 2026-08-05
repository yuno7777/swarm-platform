//! Stable hashing.
//!
//! FNV-1a: small, dependency-free, and — unlike `DefaultHasher` — **identical across
//! processes and releases**. The platform relies on that: idempotency keys, cache
//! keys, and mock provider responses all have to agree between a coordinator that
//! restarted and one that did not.

/// FNV-1a offset basis.
pub const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Fold `bytes` into `hash`. Chain calls to hash several fields.
#[must_use]
pub fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Hash of a single byte string.
#[must_use]
pub fn stable_hash(bytes: &[u8]) -> u64 {
    fnv1a(OFFSET, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_is_deterministic_and_sensitive_to_small_changes() {
        assert_eq!(stable_hash(b"swarm"), stable_hash(b"swarm"));
        assert_ne!(stable_hash(b"swarm"), stable_hash(b"swarn"));
        assert_ne!(stable_hash(b""), stable_hash(b"\0"));
    }

    #[test]
    fn chaining_distinguishes_field_boundaries() {
        let joined = fnv1a(stable_hash(b"ab"), b"c");
        let split = fnv1a(stable_hash(b"a"), b"bc");
        assert_eq!(joined, split, "FNV-1a over the same byte stream agrees");
        // Which is exactly why callers that need boundaries must insert separators.
        assert_ne!(stable_hash(b"a|bc"), stable_hash(b"ab|c"));
    }
}
