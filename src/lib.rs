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
    collections::{HashMap, VecDeque},
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    os::windows::fs::{FileExt, OpenOptionsExt},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, SyncSender},
    },
    thread,
};

pub use serde_json;

mod bloom;
mod config;
mod wal;

// Re-export
use crate::{bloom::BlockMetadata, wal::WalEntry};
pub use config::{Durability, LiveWireConfig, WalConfig};

const FILE_FLAG_NO_BUFFERING: u32 = 0x20000000;

const ERASE_BLOCK_SIZE: usize = 262_144; // 256KB

/// Represents a single, predictable 64-byte slot.
/// Fits perfectly into one L1/L2/L3 cache line.
#[repr(C, align(64))]
pub struct WireSlot {
    pub key: u64,           // 8 bytes
    pub is_tombstone: bool, // 1 byte (very wasteful, but fast)
    pub payload: [u8; 55],  // 55 bytes of raw, wasted data to fill our 64 bytes
}

/// Represents a 256KB chunk mapped directly to the SSD's erase block.
/// By aligning to 4096, we guarantee `O_DIRECT` will accept it.
#[repr(C, align(4096))]
pub struct WireBlock {
    pub count: u16, // Track how many slots are occupied
    pub slots: [WireSlot; 4095],
}

impl WireBlock {
    pub fn find_slot(&self, key: u64, start_slot: usize) -> Option<usize> {
        if self.count == 0 && key != 0 {
            return Some(start_slot);
        }

        for i in 0..4095 {
            let jump = i * (i + 1) / 2;
            let current_slot = (start_slot + jump) % 4095;
            let slot = &self.slots[current_slot];

            if slot.key == key || slot.key == 0 {
                return Some(current_slot);
            }
        }
        None
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

impl Clone for AlignedBlock {
    fn clone(&self) -> Self {
        let new_block = AlignedBlock::new();
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, new_block.ptr, 1);
        }
        new_block
    }
}

// Let Rust know it's safe to send this to another thread
unsafe impl Send for AlignedBlock {}

pub struct DirtyBlock {
    pub data: AlignedBlock,
    pub is_dirty: bool,
}

pub struct LiveWire {
    pub handle: File,
    pub pool: HashMap<u64, DirtyBlock>,
    pub lru_order: VecDeque<u64>,
    pub config: LiveWireConfig,
    pub wal_config: WalConfig,
    pub flusher_tx: Option<mpsc::SyncSender<(u64, AlignedBlock)>>,
    pub flusher_thread: Option<thread::JoinHandle<()>>,
    /// The channel to send entires to the background WAL thread
    pub wal_tx: Option<SyncSender<(u64, WalEntry)>>,
    /// The WAL background thread
    pub wal_thread: Option<thread::JoinHandle<()>>,
    /// The global counter of every operation ever requested
    pub global_sequence: Arc<AtomicU64>,
    /// The counter updated only by the background thread
    /// when data hits the metal
    pub synced_sequence: Arc<AtomicU64>,
    /// Tracks how many puts since the last background sync
    pub unflushed_puts: u64,
    /// Tracks the count of every block without touching SSD
    pub block_counts: Vec<u16>,
    /// The global RAM directory for every block on the SSD
    pub directory: Vec<BlockMetadata>,
    // TODO: `io_uring` instance
}

/// Walton's Constant.
/// Derived from the quantum radioactive decay of isotopes
/// measured in the North West of England.
const WALTONS_CONSTANT: u64 = 0xc47589d5cc327637;

impl LiveWire {
    /// Create a new LiveWire instance.
    pub fn new(
        handle: File,
        config: LiveWireConfig,
        wal_config: WalConfig,
    ) -> std::io::Result<Self> {
        // Check size of handle, compare to config, adjust accordingly
        let expected_size =
            config.region_count * config.blocks_per_region * (ERASE_BLOCK_SIZE as u64);
        let current_size = handle
            .metadata()
            .expect("Failed to read file metadata")
            .len();
        if current_size < expected_size {
            handle
                .set_len(expected_size)
                .expect("Failed to pre-allocate disk space! Check disk capacity.");

            // Force the OS to update
            handle.sync_all()?;
        }

        let (tx, rx) = mpsc::sync_channel::<(u64, AlignedBlock)>(64);

        let bg_handle = handle.try_clone().expect("Failed to clone file handle");

        // Spawn the background flusher
        let flusher_thread_handle = thread::spawn(move || {
            while let Ok((block_id, block)) = rx.recv() {
                let offset = block_id * (ERASE_BLOCK_SIZE as u64);
                // Slow NVMe write happens away from main thread
                let _ = bg_handle.seek_write(block.as_slice_u8(), offset);
                // NOTE: We don't `sync_all()` here to keep throughput high.
                // The OS should flush the hardware buffer.
            }
        });

        // Initialize WAL atomics
        let global_sequence = Arc::new(AtomicU64::new(0));
        let synced_sequence = Arc::new(AtomicU64::new(0));

        // Create the Channel (bounded to 65,536 pending operations)
        let (wal_tx, wal_rx) = mpsc::sync_channel::<(u64, WalEntry)>(65_536);

        // Clone the Atomic for the background thread
        let bg_synced_sequence = Arc::clone(&synced_sequence);

        let wal_handle = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING)
            .open("main.wal")
            .unwrap();

        if wal_handle.metadata().unwrap().len() < 16_777_216 {
            wal_handle.set_len(16_777_216).unwrap();
        }

        let total_blocks = (config.region_count * config.blocks_per_region) as usize;
        let mut block_counts = vec![0u16; total_blocks];
        let mut directory = vec![BlockMetadata::new(); total_blocks];

        if let Ok(meta_file) = File::open("main.meta") {
            let mut reader = BufReader::new(meta_file);
            let mut count_buf = [0u8; 2];

            for i in 0..total_blocks {
                if reader.read_exact(&mut count_buf).is_err() {
                    break;
                }
                directory[i].count = u16::from_le_bytes(count_buf);

                // Read the bloom filter into the box
                if reader.read_exact(&mut *directory[i].bloom).is_err() {
                    break;
                }
            }
        }

        // Sweep disk to read block metadata
        let mut count_buffer = [0u8; 2];
        for i in 0..total_blocks {
            let offset = (i * ERASE_BLOCK_SIZE) as u64;
            // We only need to read the first 2 bytes
            if let Ok(_) = handle.seek_read(&mut count_buffer, offset) {
                block_counts[i] = u16::from_le_bytes(count_buffer);
            }
        }

        let mut engine = Self {
            handle,
            pool: HashMap::new(),
            lru_order: VecDeque::new(),
            config,
            wal_config,
            flusher_tx: Some(tx),
            flusher_thread: Some(flusher_thread_handle),
            wal_tx: Some(wal_tx),
            wal_thread: None, // We will assign this in a second
            global_sequence,
            synced_sequence,
            unflushed_puts: 0,
            block_counts: Vec::new(),
            directory,
        };

        let mut wal_handle_mut = wal_handle.try_clone().unwrap();
        engine.recover_from_wal(&mut wal_handle_mut);

        let thread_batch_size = engine.wal_config.max_batch_size;

        // Spawn WAL thread
        let wal_thread_handle = thread::spawn(move || {
            // Batching buffer
            let mut batch = Vec::with_capacity(thread_batch_size);
            let mut current_wal_offset = 0u64;

            let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
            let io_buffer = unsafe { std::alloc::alloc_zeroed(layout) as *mut WalEntry };

            // Blocks until first `put` happens
            while let Ok((seq_id, entry)) = wal_rx.recv() {
                batch.push((seq_id, entry));

                // Greedy Grab loop
                // Instantly drain any other pending items up to our 4KB limit
                while batch.len() < thread_batch_size {
                    match wal_rx.try_recv() {
                        Ok((next_seq, next_entry)) => batch.push((next_seq, next_entry)),
                        Err(mpsc::TryRecvError::Empty) => break, // Queue is empty
                        Err(mpsc::TryRecvError::Disconnected) => {
                            if batch.is_empty() {
                                return;
                            }
                            break;
                        } // Shutting down
                    }
                }

                unsafe {
                    // Fast pad
                    std::ptr::write_bytes(io_buffer as *mut u8, 0, 4096);

                    // Flat copy
                    for (i, (_seq, entry)) in batch.iter().enumerate() {
                        std::ptr::write(io_buffer.add(i), *entry);
                    }

                    // Slice it for I/O
                    let write_slice = std::slice::from_raw_parts(io_buffer as *const u8, 4096);

                    // Slam it
                    wal_handle
                        .seek_write(write_slice, current_wal_offset)
                        .expect("Fatal Error: WAL write failed!");
                }

                // Move forward 1 NVMe page (4KB)
                current_wal_offset += 4096;
                current_wal_offset %= 16_777_216;

                // Update the Atomic counter to release any spin-waiting main threads
                let highest_seq_in_batch = batch.last().unwrap().0;
                bg_synced_sequence.store(highest_seq_in_batch, Ordering::Release);

                // Clear the buffer for next burst
                batch.clear();
            }
        });

        engine.wal_thread = Some(wal_thread_handle);

        Ok(engine)
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
    pub fn predict_location(&self, key: u64) -> (u64, usize) {
        let scrambled = key.wrapping_mul(WALTONS_CONSTANT);
        let folded = scrambled ^ (scrambled >> 32);

        let slot = (folded & 4095) as usize;
        let block = (folded >> 12) % self.config.blocks_per_region;

        (block, slot)
    }

    pub fn get_or_fetch_block(&mut self, block_id: u64) -> std::io::Result<&mut DirtyBlock> {
        // Cache Hit
        if self.pool.contains_key(&block_id) {
            // We skip the O(n) retain and just push a duplicate.
            // Duplicates are handled lazily during eviction.
            self.lru_order.push_back(block_id);
            return Ok(self.pool.get_mut(&block_id).unwrap());
        }

        // Cache Miss
        // If we're at capacity, kick someone out to make room
        while self.pool.len() >= self.config.pool_capacity {
            if let Some(victim_id) = self.lru_order.pop_front() {
                // Skip stale duplicates
                if !self.pool.contains_key(&victim_id) {
                    continue;
                }
                if let Some(entry) = self.pool.remove(&victim_id) {
                    if entry.is_dirty {
                        let _ = self
                            .flusher_tx
                            .as_ref()
                            .unwrap()
                            .send((victim_id, entry.data));
                    }
                }
            } else {
                break;
            }
        }

        // Fetch the new block
        let mut block = AlignedBlock::new();
        let offset = block_id * (ERASE_BLOCK_SIZE as u64);
        let _ = self.handle.seek_read(block.as_mut_slice_u8(), offset);

        self.pool.insert(
            block_id,
            DirtyBlock {
                data: block,
                is_dirty: false,
            },
        );
        self.lru_order.push_back(block_id);

        Ok(self.pool.get_mut(&block_id).unwrap())
    }

    pub fn get(&mut self, key: u64) -> Option<[u8; 55]> {
        for region in 0..self.config.region_count {
            let effective_key = key ^ (region * WALTONS_CONSTANT);
            let (raw_block_id, start_slot) = self.predict_location(effective_key);
            let block_id = (region * 512) + (raw_block_id % 512);

            if let Ok(entry) = self.get_or_fetch_block(block_id) {
                if let Some(idx) = entry.data.find_slot(key, start_slot) {
                    let slot = &entry.data.slots[idx];

                    // Found it
                    if slot.key == key {
                        return Some(slot.payload);
                    }

                    // Exit early if:
                    // We found an empty slot AND the block isn't "Full".
                    // If the block is "Full" (> 3500), the key might be in the NEXT region.
                    if slot.key == 0 && entry.data.count <= 3500 {
                        return None;
                    }
                }
            }
        }
        None
    }

    /// # Put
    ///
    /// TODO: Implement *shadow paging*:
    ///
    /// - Write new version to a new location on disk
    /// - Update master pointer once the write is confirmed
    /// - Makes us virtually immune to data corruption from
    ///   crashes
    pub fn put(&mut self, key: u64, data: [u8; 55]) -> std::io::Result<()> {
        // Get a unique ID for this operation
        let seq_id = self.global_sequence.fetch_add(1, Ordering::Relaxed);

        let entry = WalEntry {
            key,
            is_tombstone: false,
            payload: data,
        };

        // Send to background thread's queue
        let _ = self.wal_tx.as_ref().unwrap().send((seq_id, entry));

        // Durability check
        if self.wal_config.mode == Durability::Strict {
            // Spin-wait. The CPU burns cycles checking the atomic
            // variable. It's incredibly fast because the variable stays
            // in the L3 cache until the background thread modifies it.
            while self.synced_sequence.load(Ordering::Acquire) < seq_id {
                std::hint::spin_loop(); // Tells the CPU we are waiting, saves power
            }
        }

        // We try every region until we find a block with an empty slot
        for region in 0..self.config.region_count {
            // Modify the key slightly for each region to ensure a different block
            let effective_key = key ^ (region * WALTONS_CONSTANT);
            let (raw_block_id, start_slot) = self.predict_location(effective_key);
            let block_id = (region * 512) + (raw_block_id % 512);
            let global_idx = block_id as usize;

            // If the block is full and the bloom filter guarantees our key isn't in there,
            // skip the disk entirely.
            let should_skip = {
                let meta = &self.directory[global_idx];
                meta.count > 3500 && !meta.might_contain(key)
            };

            if should_skip {
                continue;
            }

            // If we get here, either the block has room, or the key is probably updating.
            let entry = self.get_or_fetch_block(block_id)?;

            if let Some(slot_idx) = entry.data.find_slot(key, start_slot) {
                let is_empty_slot = entry.data.slots[slot_idx].key == 0;

                if is_empty_slot && entry.data.count > 3500 {
                    continue;
                }

                if is_empty_slot {
                    entry.data.count += 1;
                }

                entry.data.slots[slot_idx].key = key;
                entry.data.slots[slot_idx].payload = data;
                entry.is_dirty = true;

                if is_empty_slot {
                    self.directory[global_idx].count += 1;
                }
                self.directory[global_idx].insert(key);

                return Ok(());
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Total Database Overflow!",
        ))
    }

    pub fn sync(&mut self) -> std::io::Result<()> {
        for (&block_id, entry) in self.pool.iter_mut() {
            if entry.is_dirty {
                let offset = block_id * (ERASE_BLOCK_SIZE as u64);
                self.handle.seek_write(entry.data.as_slice_u8(), offset)?;
                entry.is_dirty = false;
            }
        }
        self.handle.sync_all()?;
        Ok(())
    }

    pub fn background_sync(&mut self) {
        for (&block_id, entry) in self.pool.iter_mut() {
            if entry.is_dirty {
                // Take memory snapshot
                let snapshot = entry.data.clone();
                // Mark the RAM block as clean
                entry.is_dirty = false;
                // Toss it to the background thread
                let _ = self.flusher_tx.as_ref().unwrap().send((block_id, snapshot));
            }
        }
    }

    fn recover_from_wal(&mut self, wal_handle: &mut File) {
        let mut recovered_keys = 0;
        let mut read_buffer = [0u8; 4096];

        for page in 0..4096 {
            let offset = page * 4096;
            let bytes_read = wal_handle.seek_read(&mut read_buffer, offset).unwrap();
            if bytes_read < 4096 {
                continue;
            }

            let entries: &[WalEntry; 64] = unsafe { std::mem::transmute(&read_buffer) };

            for entry in entries.iter() {
                if entry.key != 0 {
                    // We found data that survived a crash
                    recovered_keys += 1;

                    // Inject directly into engine
                    for region in 0..self.config.region_count {
                        let effective_key = entry.key ^ (region * WALTONS_CONSTANT);
                        let (raw_block_id, start_slot) = self.predict_location(effective_key);
                        let block_id = (region * 512) + (raw_block_id % 512);
                        let global_idx = block_id as usize;

                        if let Ok(block_entry) = self.get_or_fetch_block(block_id) {
                            if block_entry.data.count > 3500 {
                                continue;
                            }

                            if let Some(slot_idx) =
                                block_entry.data.find_slot(entry.key, start_slot)
                            {
                                let is_empty_slot = block_entry.data.slots[slot_idx].key == 0;
                                block_entry.data.slots[slot_idx].key = entry.key;
                                block_entry.data.slots[slot_idx].payload = entry.payload;
                                block_entry.is_dirty = true;

                                if is_empty_slot {
                                    block_entry.data.count += 1;
                                    self.directory[global_idx].count += 1;
                                }

                                self.directory[global_idx].insert(entry.key);
                                break;
                            }
                        }
                    }
                }
            }
        }

        if recovered_keys > 0 {
            // Force a sync to move recovered data into main file
            let _ = self.sync();
            // Wipe the WAL clean so we don't recover it twice
            wal_handle.set_len(0).unwrap();
            wal_handle.set_len(16_777_216).unwrap();
        }
    }
}

impl Drop for LiveWire {
    fn drop(&mut self) {
        // Flush all dirty blocks to main.lw
        for (&block_id, entry) in self.pool.iter_mut() {
            if entry.is_dirty {
                let offset = block_id * (ERASE_BLOCK_SIZE as u64);
                let _ = self.handle.seek_write(entry.data.as_slice_u8(), offset);
            }
        }
        let _ = self.handle.sync_all();

        // Kill flusher channel
        drop(self.flusher_tx.take());
        if let Some(handle) = self.flusher_thread.take() {
            let _ = handle.join();
        }

        // Kill the WAL channel
        drop(self.wal_tx.take());

        // Wait for the WAL thread to finish
        if let Some(handle) = self.wal_thread.take() {
            let _ = handle.join();
        }

        // Wipe the WAL file clean
        let wal_handle = OpenOptions::new().write(true).open("main.wal").unwrap();
        wal_handle.set_len(0).unwrap();
        wal_handle.set_len(16_777_216).unwrap();

        // Save metadata directory
        let meta_file = File::create("main.meta").expect("Failed to create main.meta");
        let mut writer = BufWriter::new(meta_file);

        for meta in &self.directory {
            writer.write_all(&meta.count.to_le_bytes()).unwrap();
            writer.write_all(&*meta.bloom).unwrap();
        }
        writer.flush().unwrap();
    }
}

// JSON Document Store

const INLINE_DATA_SIZE: usize = 50;
const OVERFLOW_CHUNK_SIZE: usize = 55;
const FLAG_INLINE: u8 = 0x00;
const FLAG_OVERFLOW: u8 = 0x01;

/// FNV-1a hash for string keys.
fn fnv1a(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // Keep bit 63 clear, overflow chunks use that bit
    hash & !(1u64 << 63)
}

/// Derives a key for overflow chunk `index` belonging to `base_key`.
fn overflow_key(base_key: u64, index: u64) -> u64 {
    (1u64 << 63) | base_key.wrapping_add(index.wrapping_mul(0x9e3779b97f4a7c15))
}

pub struct JsonStore {
    pub inner: LiveWire,
}

impl JsonStore {
    pub fn new(inner: LiveWire) -> Self {
        Self { inner }
    }

    pub fn put(&mut self, key: &str, value: &serde_json::Value) -> std::io::Result<()> {
        let hash = fnv1a(key);
        let serialized = serde_json::to_vec(value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let total_len = serialized.len() as u32;

        // Build the head slot payload
        let mut payload = [0u8; 55];
        let inline_len = serialized.len().min(INLINE_DATA_SIZE);
        payload[0] = if serialized.len() > INLINE_DATA_SIZE {
            FLAG_OVERFLOW
        } else {
            FLAG_INLINE
        };
        payload[1..5].copy_from_slice(&total_len.to_le_bytes());
        payload[5..5 + inline_len].copy_from_slice(&serialized[..inline_len]);

        self.inner.put(hash, payload)?;

        // Write overflow chunks if needed
        if serialized.len() > INLINE_DATA_SIZE {
            let remaining = &serialized[INLINE_DATA_SIZE..];
            for (i, chunk) in remaining.chunks(OVERFLOW_CHUNK_SIZE).enumerate() {
                let mut chunk_payload = [0u8; 55];
                chunk_payload[..chunk.len()].copy_from_slice(chunk);
                self.inner
                    .put(overflow_key(hash, i as u64), chunk_payload)?;
            }
        }

        Ok(())
    }

    pub fn get(&mut self, key: &str) -> Option<serde_json::Value> {
        let hash = fnv1a(key);
        let payload = self.inner.get(hash)?;

        let flags = payload[0];
        let total_len =
            u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]) as usize;

        if total_len == 0 {
            return None;
        }

        let inline_len = total_len.min(INLINE_DATA_SIZE);
        let mut data = Vec::with_capacity(total_len);
        data.extend_from_slice(&payload[5..5 + inline_len]);

        if flags == FLAG_OVERFLOW {
            let remaining = total_len - INLINE_DATA_SIZE;
            let num_chunks = (remaining + OVERFLOW_CHUNK_SIZE - 1) / OVERFLOW_CHUNK_SIZE;

            for i in 0..num_chunks {
                let chunk_payload = self.inner.get(overflow_key(hash, i as u64))?;
                let bytes_left = remaining - (i * OVERFLOW_CHUNK_SIZE);
                let chunk_len = bytes_left.min(OVERFLOW_CHUNK_SIZE);
                data.extend_from_slice(&chunk_payload[..chunk_len]);
            }
        }

        serde_json::from_slice(&data).ok()
    }

    pub fn delete(&mut self, key: &str) -> std::io::Result<()> {
        let hash = fnv1a(key);

        // Read the head to find overflow chunks
        if let Some(payload) = self.inner.get(hash) {
            let flags = payload[0];
            let total_len =
                u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]) as usize;

            // Tombstone overflow chunks
            if flags == FLAG_OVERFLOW && total_len > INLINE_DATA_SIZE {
                let remaining = total_len - INLINE_DATA_SIZE;
                let num_chunks = (remaining + OVERFLOW_CHUNK_SIZE - 1) / OVERFLOW_CHUNK_SIZE;
                let empty = [0u8; 55];
                for i in 0..num_chunks {
                    self.inner.put(overflow_key(hash, i as u64), empty)?;
                }
            }

            // Tombstone the head
            self.inner.put(hash, [0u8; 55])?;
        }

        Ok(())
    }

    pub fn sync(&mut self) -> std::io::Result<()> {
        self.inner.sync()
    }
}
