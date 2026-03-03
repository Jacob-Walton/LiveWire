//! LiveWire
//!
//! The RUM Conjecture states that when designing a data structure,
//! you can optimize for two of three things, but never all three:
//! 1. *R*ead overhead
//! 2. *U*pdate/Write overhead
//! 3. *M*emory/Space overhead
//!
//! We can waste memory and space as modern computers have much more
//! RAM and storage than those around when things like B-trees were
//! thought of.
//!
//! What we're aiming to implement here is either of these:
//!
//! ### NVMe Exploitation
//!
//! NVMe SSDs behave completely differently to magnetic disks.
//! They write in 4KB pages but erase in massive blocks of 256KB.
//! They have dozens of internal queues and can process thousands
//! of requests in parallel. We can design a data structure that
//! completely abandons the concept of "trees" or "logs", and
//! instead is mathematically mapped to the exact architecture of
//! SSD flash controllers. We need to structure it to take
//! advantage of massive parallel I/O queues.
//!
//! ### Maths Tricks
//!
//! If we wrote the numbers 1 through 1,000,000 to a disk in order,
//! we don't need an index or a tree to find the number 54,321. We
//! can use pure maths to calculate exactly which byte on the disk
//! holds that number, and jump straight to it. O(1) read, O(1) write.
//!
//! This only works if the data is completely predictable.
//!
//! We could design a data structure that uses:
//! a. Machine learning
//! b. Linear regression
//! c. Mathematical hashing
//!
//! In order to predict exactly where a record should physically live
//! on a disk, bypassing the need to read an index entirely.
//!
//! ## Issues
//!
//! ### Cache
//!
//! Whilst RAM and SSDs have dramatically increased in capacity since
//! the 70s, the CPU cache is still absolutely tiny. However, some
//! modern processors have larger caches such as the 7950X3D with
//! - L1: 1MB
//! - L2: 16MB
//! - L3: 128MB
//!
//! However, even with a massive 128MB L3 cache, the CPU doesn't exactly
//! move data around how we want it to. CPUs don't fetch individual bytes.
//! Instead, they fetch memory in chunks known as *cache lines*, which are
//! usually 64 bytes.
//!
//! Since we're sacrificing on space, we have to make sure we don't destroy
//! our spatial localility otherwise we'd risk wasting most of our cache.
//!
//! ### NVMes and OSes
//!
//! To implement the NVMe trick we'd have to fight the OS the entire time.
//!
//! Standard OS commands like `read()` and `write()` go through the kernel's
//! "Page Cache". The OS tries to "help" us by batching and predicting our
//! reads/writes, which would completely destroy any kind of custom SSD
//! architecture we try to build.
//!
//! Instead, we would have to bypass the OS entirely. Such as on Linux,
//! we'd have to build it using `O_DIRECT` (Direct I/O) and an async
//! interface such as `io_uring`.
//!
//! ### ML
//!
//! This is currently a brand new field in CS called *Learned Indexes*.
//!
//! Let's say we use a linear regression model to predict where a record
//! lives on disk. If computing that maths equation takes our CPU 150
//! nanoseconds, but traversing a standard B-Tree only takes 50 nanoseconds,
//! our "maths trick" loses. The maths must be incredibly lightweight.

use std::{
    alloc::{Layout, alloc, dealloc},
    fs::File,
    os::windows::fs::FileExt,
};

const CACHE_LINE_SIZE: usize = 64;
const NVME_PAGE_SIZE: usize = 4096; // 4KB
const ERASE_BLOCK_SIZE: usize = 262_144; // 256KB
const TOTAL_BLOCKS: u64 = 512; // 128MB

/// Represents a single, predictable 64-byte slot.
/// Fits perfectly into one L1/L2/L3 cache line.
#[repr(C, align(64))]
pub struct WireSlot {
    pub key: u64,           // 8 bytes
    pub is_tombstone: bool, // 1 byte (very wasteful, but fast)
    pub payload: [u8; 55],  // 55 bytes of raw, wasted data to fill our 64 bytes
}

/// Represents a 256KB chnk mapped directly to the SSD's erase block.
/// By aligning to 4096, we guarantee `O_DIRECT` will accept it.
#[repr(C, align(4096))]
pub struct WireBlock {
    /// 256KB divided by 64 bytes = exactly 4,096 contiguous slots.
    pub slots: [WireSlot; 4096],
}

impl WireBlock {
    pub fn find_slot(&self, key: u64) -> Option<usize> {
        // Find the starting point
        let (_, start_slot) = LiveWire::predict_location(key);

        for i in 0..4096 {
            // Frog Jump
            let jump = i * i;
            let current_slot = (start_slot + jump) % 4096;

            let slot = &self.slots[current_slot];

            // Success if we found key or empty spot
            if slot.key == key || slot.key == 0 {
                return Some(current_slot);
            }
        }
        None // The 256KB block is completely full
    }

    fn as_slice_u8(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn as_mut_slice_u8(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self as *mut Self as *mut u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// A smart pointer that ensures memory is aligned to a 4KB page boundary.
pub struct AlignedBlock {
    ptr: *mut WireBlock,
}

impl AlignedBlock {
    pub fn new() -> Self {
        let layout = Layout::from_size_align(std::mem::size_of::<WireBlock>(), 4096).unwrap();

        let ptr = unsafe { alloc(layout) as *mut WireBlock };
        if ptr.is_null() {
            panic!("Failed to allocate aligned memory");
        }

        // Initialize with zeros
        unsafe {
            std::ptr::write_bytes(ptr, 0, 1);
        }

        Self { ptr }
    }
}

impl std::ops::Deref for AlignedBlock {
    type Target = WireBlock;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl std::ops::DerefMut for AlignedBlock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}

impl Drop for AlignedBlock {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(std::mem::size_of::<WireBlock>(), 4096).unwrap();
        unsafe {
            dealloc(self.ptr as *mut u8, layout);
        }
    }
}

pub struct DirtyBlock {
    pub data: AlignedBlock,
    pub is_dirty: bool,
}

pub struct LiveWire {
    pub handle: File,
    // TODO: `io_uring` instance
}

/// Walton's Constant.
/// Derived from the quantum radioactive decay of isotopes
/// measured in the North West of England.
const WALTONS_CONSTANT: u64 = 0xc47589d5cc327637;

impl LiveWire {
    /// Create a new LiveWire instance.
    pub fn new(handle: File) -> Self {
        Self { handle }
    }

    /// Where the magic happens. Given a key, it predicts
    /// exactly which 256KB block, and which 64-byte slot
    /// inside that block, the data lives in.
    ///
    /// ## Casino (Hashing/Scattering)
    ///
    /// Imagine our keys aren't nice, sequential numbers.
    /// Imagine they are so wildly unpredictable we couldn't
    /// even begin to image how unpredictable they truly are.
    ///
    /// We can't predict where it goes with this. Instead,
    /// we have to force it. How can we take a messy input
    /// and smash, twist, and chop it up until it spits out
    /// a perfectly clean number between 0 and 4095?
    ///
    /// However, if we smash data up randomly, eventually
    /// two different keys will spit out the exact same
    /// slot number (collision). We'll have to handle
    /// this edge case.
    pub fn predict_location(key: u64) -> (u64, usize) {
        let scrambled = key.wrapping_mul(WALTONS_CONSTANT);
        let folded = scrambled ^ (scrambled >> 32);

        let slot = (folded & 4095) as usize;
        let block = (folded >> 12) % TOTAL_BLOCKS;

        (block, slot)
    }

    pub fn get(&self, key: u64) -> Option<[u8; 55]> {
        // Where is it?
        let (block_id, _) = Self::predict_location(key);
        let offset = block_id * (ERASE_BLOCK_SIZE as u64);

        // Prepare blank block in memory
        let mut block = AlignedBlock::new();

        // Read from the NVMe
        match self.handle.seek_read(block.as_mut_slice_u8(), offset) {
            Ok(_) => block.find_slot(key).and_then(|idx| {
                let slot = &block.slots[idx];
                if slot.key == key {
                    Some(slot.payload)
                } else {
                    None
                }
            }),
            Err(_) => None,
        }
    }

    /// # Put
    ///
    /// TODO: Implement *shadow paging*:
    ///
    /// - Write new version to a new location on disk
    /// - Update master pointer once the write is confirmed
    /// - Makes us virtually immune to data corruption from
    ///   crashes
    pub fn put(&self, key: u64, data: [u8; 55]) -> std::io::Result<()> {
        // Find where key should live
        let (block_id, _) = Self::predict_location(key);
        let offset = block_id * (ERASE_BLOCK_SIZE as u64);

        // Pull the block into the Heap (avoiding stack overflow)
        let mut block = AlignedBlock::new();

        // Read the existing block from disk
        let read_result = self.handle.seek_read(block.as_mut_slice_u8(), offset);

        // Find a seat using Frog Jump
        if let Some(slot_idx) = block.find_slot(key) {
            let slot = &mut block.slots[slot_idx];
            slot.key = key;
            slot.payload = data;
            slot.is_tombstone = false;

            // Slam it back to the NVMe
            self.handle.seek_write(block.as_slice_u8(), offset)?;
            self.handle.sync_all()?; // SLAM
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "LiveWire Block Overflow!",
            ))
        }
    }
}
