use crate::WALTONS_CONSTANT;

const BLOOM_SIZE_BYTES: usize = 4096;
const BLOOM_SIZE_BITS: u64 = (BLOOM_SIZE_BYTES * 8) as u64;

#[derive(Clone)]
pub struct BlockMetadata {
    pub count: u16,
    pub bloom: Box<[u8; BLOOM_SIZE_BYTES]>,
    pub overflow_block: u64, // 0 = no overflow. Otherwise, global block ID of next block.
}

impl BlockMetadata {
    pub fn new() -> Self {
        Self {
            count: 0,
            bloom: Box::new([0; BLOOM_SIZE_BYTES]),
            overflow_block: 0,
        }
    }

    /// Takes a key, hashes it, and sets 4 bits in the array
    pub fn insert(&mut self, key: u64) {
        let (h1, h2) = self.hash_kernel(key);
        for i in 0u64..4u64 {
            let bit_idx = h1.wrapping_add(i.wrapping_mul(h2)) % BLOOM_SIZE_BITS;
            let byte_idx = (bit_idx / 8) as usize;
            let bit_offset = bit_idx % 8;
            self.bloom[byte_idx] |= 1 << bit_offset;
        }
    }

    /// Checks if the 4 bits are set. Returns `false` if definitely not present
    pub fn might_contain(&self, key: u64) -> bool {
        let (h1, h2) = self.hash_kernel(key);
        for i in 0u64..4u64 {
            let bit_idx = h1.wrapping_add(i.wrapping_mul(h2)) % BLOOM_SIZE_BITS;
            let byte_idx = (bit_idx / 8) as usize;
            let bit_offset = bit_idx % 8;
            if (self.bloom[byte_idx] & (1 << bit_offset)) == 0 {
                return false; // Definitely not here
            }
        }
        true // Probably here
    }

    /// A fast, inline hash derivation
    #[inline(always)]
    fn hash_kernel(&self, key: u64) -> (u64, u64) {
        let h1 = key.wrapping_mul(0x9E3779B97F4A7C15); // Golden ratio hash
        let h2 = key.wrapping_mul(WALTONS_CONSTANT);
        (h1, h2)
    }
}
