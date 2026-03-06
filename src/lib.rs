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
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;

#[cfg(target_os = "linux")]
use std::{os::unix::fs::OpenOptionsExt, thread::JoinHandle};

pub use crate::file::*;

pub use serde_json;

mod bloom;
mod config;
mod file;
mod wal;

// Re-export
use crate::{
    bloom::BlockMetadata,
    file::{seek_read, seek_write},
    wal::WalEntry,
};
pub use config::{Durability, LiveWireConfig, WalConfig};

const ERASE_BLOCK_SIZE: usize = 262_144; // 256KB

/// Single WAL file shared across all shards.
const WAL_FILE_SIZE: u64 = 67_108_864;

pub const TOMBSTONE_KEY: u64 = 0xFFFFFFFFFFFFFFFF;

/// Represents a single, predictable 64-byte slot.
/// Fits perfectly into one L1/L2/L3 cache line.
#[repr(C, align(64))]
pub struct WireSlot {
    pub key: u64,           // 8 bytes
    pub is_tombstone: bool, // 1 byte (very wasteful, but fast)
    pub payload: [u8; 55],  // 55 bytes of raw data to fill our 64 bytes
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
            return Some(start_slot % 4095);
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

impl Default for AlignedBlock {
    fn default() -> Self {
        Self::new()
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

// Tells Rust it's safe for other threads to see this pointer
unsafe impl Sync for AlignedBlock {}

pub struct DirtyBlock {
    pub data: AlignedBlock,
    pub is_dirty: bool,
    pub last_access: u64,
}

pub struct Shard {
    pub pool: HashMap<u64, DirtyBlock>,
    pub directory: Vec<BlockMetadata>,
    pub overflow_meta: HashMap<u64, BlockMetadata>,
    pub access_counter: AtomicU64,
}

pub struct LiveWire {
    pub handle: Arc<File>,
    pub shards: Vec<Arc<parking_lot::RwLock<Shard>>>,
    pub config: LiveWireConfig,
    pub wal_config: WalConfig,
    pub flusher_thread: Option<thread::JoinHandle<()>>,
    pub flusher_txs: Vec<crossbeam_channel::Sender<(u64, AlignedBlock)>>,
    pub wal_thread: Option<thread::JoinHandle<()>>,
    pub wal_tx: crossbeam_channel::Sender<WalEntry>,
    pub wal_epoch: Arc<AtomicU64>,
    pub next_free_block: AtomicU64,
}

/// Walton's Constant.
/// Derived from the quantum radioactive decay of isotopes
/// measured in the North West of England.
const WALTONS_CONSTANT: u64 = 0xc47589d5cc327637;

impl LiveWire {
    /// Create a new LiveWire instance.
    pub fn new(config: LiveWireConfig, wal_config: WalConfig) -> std::io::Result<Self> {
        if !config.data_dir.exists() {
            std::fs::create_dir_all(&config.data_dir).expect("failed to create data dir");
        }

        let lw_path = config.data_dir.join("main.lw");
        #[cfg(windows)]
        let handle = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .custom_flags(crate::FILE_FLAG_NO_BUFFERING)
            .open(lw_path)
            .unwrap();

        #[cfg(unix)]
        let handle = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(false)
            .open(lw_path)
            .unwrap();

        #[cfg(unix)]
        enable_direct_io(&handle).unwrap();

        let expected_size =
            config.region_count * config.blocks_per_region * (ERASE_BLOCK_SIZE as u64);
        if handle.metadata()?.len() < expected_size {
            handle.set_len(expected_size)?;
            handle.sync_all()?;
        }

        #[allow(unused)]
        let bg_handle = handle.try_clone().expect("Failed to clone file handle");

        let total_blocks = (config.region_count * config.blocks_per_region) as usize;
        let num_shards = config.num_shards.max(1);
        let blocks_per_shard = total_blocks / num_shards;

        let (wal_tx, wal_rx) = crossbeam_channel::bounded::<WalEntry>(131_072);
        let wal_epoch = Arc::new(AtomicU64::new(0));

        let wal_path = config.data_dir.join("main.wal");
        {
            let tmp = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(&wal_path)?;
            tmp.set_len(WAL_FILE_SIZE)?;
        }

        #[cfg(windows)]
        let wal_handle = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(false)
            .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH)
            .open(&wal_path)?;
        #[cfg(target_os = "linux")]
        let wal_handle = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(false)
            .custom_flags(libc::O_DSYNC)
            .open(&wal_path)?;

        #[cfg(target_os = "linux")]
        enable_direct_io(&wal_handle)?;

        let bg_epoch = Arc::clone(&wal_epoch);
        let bg_wal_config = wal_config;
        let wal_thread = thread::spawn(move || {
            let entry_size = std::mem::size_of::<WalEntry>();
            let max_batch = bg_wal_config.max_batch_size;
            let max_bytes = max_batch * entry_size;

            // 4KB-aligned buffer
            let alloc_size = (max_bytes + 4095) & !4095;
            let layout = std::alloc::Layout::from_size_align(alloc_size, 4096).unwrap();
            let io_buffer = unsafe { std::alloc::alloc_zeroed(layout) as *mut WalEntry };

            let mut batch: Vec<WalEntry> = Vec::with_capacity(max_batch);
            let mut current_offset = 0u64;

            while let Ok(entry) = wal_rx.recv() {
                batch.push(entry);
                while batch.len() < max_batch {
                    match wal_rx.try_recv() {
                        Ok(e) => batch.push(e),
                        _ => break,
                    }
                }

                let write_bytes = batch.len() * entry_size;
                let write_pages = (write_bytes + 4095) & !4095;

                unsafe {
                    std::ptr::write_bytes(io_buffer as *mut u8, 0, write_pages);
                    for (idx, e) in batch.iter().enumerate() {
                        std::ptr::write(io_buffer.add(idx), *e);
                    }
                    let slice = std::slice::from_raw_parts(io_buffer as *const u8, write_pages);
                    seek_write(&wal_handle, slice, current_offset)
                        .expect("WAL write failed");
                }

                // On Windows, FILE_FLAG_WRITE_THROUGH alone doesn't
                // guarantee FUA. Explicit sync_data() forces the 
                // drive to commit.
                #[cfg(windows)]
                if bg_wal_config.mode == Durability::Strict {
                    wal_handle.sync_data().expect("WAL sync failed");
                }

                current_offset = (current_offset + write_pages as u64) % WAL_FILE_SIZE;

                // Advance the epoch
                bg_epoch.fetch_add(1, Ordering::Release);
                batch.clear();
            }

            unsafe { std::alloc::dealloc(io_buffer as *mut u8, layout); }
        });

        let mut engine = Self {
            handle: Arc::new(handle),
            shards: Vec::with_capacity(num_shards),
            config: config.clone(),
            wal_config,
            flusher_thread: None,
            flusher_txs: Vec::with_capacity(num_shards),
            wal_thread: Some(wal_thread),
            wal_tx,
            wal_epoch,
            next_free_block: AtomicU64::new(total_blocks as u64),
        };

        let mut shard_receivers = Vec::with_capacity(num_shards);

        for _ in 0..num_shards {
            let (f_tx, f_rx) = crossbeam_channel::bounded::<(u64, AlignedBlock)>(128);
            shard_receivers.push(f_rx);
            engine.flusher_txs.push(f_tx);
            engine.shards.push(Arc::new(parking_lot::RwLock::new(Shard {
                pool: HashMap::default(),
                directory: vec![BlockMetadata::new(); blocks_per_shard],
                access_counter: AtomicU64::new(0),
                overflow_meta: HashMap::default(),
            })));
        }

        // Start global flusher
        #[cfg(target_os = "linux")]
        {
            engine.flusher_thread = Some(Self::start_linux_flusher(bg_handle, shard_receivers));
        }

        // Load metadata into shards
        let meta_path = config.data_dir.join("main.meta");
        if let Ok(meta_file) = File::open(meta_path) {
            let mut reader = BufReader::new(meta_file);
            let mut count_buf = [0u8; 2];
            let mut overflow_ptr_buf = [0u8; 8];
            let mut block_id_buf = [0u8; 8];

            for shard_lock in &engine.shards {
                let mut shard = shard_lock.write();
                for i in 0..blocks_per_shard {
                    if reader.read_exact(&mut count_buf).is_err() {
                        break;
                    }
                    shard.directory[i].count = u16::from_le_bytes(count_buf);
                    if reader.read_exact(&mut *shard.directory[i].bloom).is_err() {
                        break;
                    }
                    if reader.read_exact(&mut overflow_ptr_buf).is_ok() {
                        shard.directory[i].overflow_block = u64::from_le_bytes(overflow_ptr_buf);
                    }
                }

                // Overflow metadata
                let mut of_count_buf = [0u8; 8];
                if reader.read_exact(&mut of_count_buf).is_ok() {
                    let overflow_count = u64::from_le_bytes(of_count_buf) as usize;
                    for _ in 0..overflow_count {
                        if reader.read_exact(&mut block_id_buf).is_err() {
                            break;
                        }
                        let block_id = u64::from_le_bytes(block_id_buf);
                        let mut meta = BlockMetadata::new();
                        if reader.read_exact(&mut count_buf).is_err() {
                            break;
                        }
                        meta.count = u16::from_le_bytes(count_buf);
                        if reader.read_exact(&mut *meta.bloom).is_err() {
                            break;
                        }
                        if reader.read_exact(&mut overflow_ptr_buf).is_err() {
                            break;
                        }
                        meta.overflow_block = u64::from_le_bytes(overflow_ptr_buf);
                        shard.overflow_meta.insert(block_id, meta);
                    }
                }
            }

            // Restore next_free_block
            let mut nfb_buf = [0u8; 8];
            if reader.read_exact(&mut nfb_buf).is_ok() {
                let saved = u64::from_le_bytes(nfb_buf);
                if saved > engine.next_free_block.load(Ordering::Relaxed) {
                    engine.next_free_block.store(saved, Ordering::Relaxed);
                }
            }
        }

        // Recover from WAL file.
        let wal_recovery_path = config.data_dir.join("main.wal");
        if let Ok(mut wal_file) = File::open(wal_recovery_path) {
            engine.recover_from_wal(&mut wal_file);
        }

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

        let slot = (folded % 4095) as usize;
        let block = (folded >> 12) % self.config.blocks_per_region;

        (block, slot)
    }

    pub fn get_or_fetch_block<'a>(
        &self,
        shard: &'a mut Shard,
        shard_idx: usize,
        block_id: u64,
        local_idx: usize,
        is_overflow: bool,
    ) -> std::io::Result<&'a mut DirtyBlock> {
        if shard.pool.contains_key(&block_id) {
            let entry = shard.pool.get_mut(&block_id).unwrap();
            entry.last_access = shard.access_counter.fetch_add(1, Ordering::Relaxed);
            return Ok(entry);
        }

        let capacity = self.config.pool_capacity / self.shards.len();
        if shard.pool.len() >= capacity {
            let victim_id = shard
                .pool
                .iter()
                .take(16)
                .min_by_key(|(_, b)| b.last_access)
                .map(|(&id, _)| id);

            if let Some(vid) = victim_id
                && let Some(entry) = shard.pool.remove(&vid)
                && entry.is_dirty
            {
                let _ = self.flusher_txs[shard_idx].send((vid, entry.data));
            }
        }

        let mut block = AlignedBlock::new();
        let has_data = if is_overflow {
            shard
                .overflow_meta
                .get(&block_id)
                .map_or(false, |m| m.count > 0)
        } else {
            shard.directory[local_idx].count > 0
        };
        if has_data {
            let offset = block_id * (ERASE_BLOCK_SIZE as u64);
            let _ = seek_read(&self.handle, block.as_mut_slice_u8(), offset);
        }

        let access = shard.access_counter.fetch_add(1, Ordering::Relaxed);
        shard.pool.insert(
            block_id,
            DirtyBlock {
                data: block,
                is_dirty: false,
                last_access: access,
            },
        );

        Ok(shard.pool.get_mut(&block_id).unwrap())
    }

    pub fn get(&self, key: u64) -> Option<[u8; 55]> {
        let num_shards = self.shards.len();
        let shard_idx = (key % num_shards as u64) as usize;
        let mut shard = self.shards[shard_idx].write();

        let total_blocks = self.config.region_count * self.config.blocks_per_region;
        let blocks_per_shard = (total_blocks / num_shards as u64) as usize;

        for region in 0..self.config.region_count {
            let effective_key = key ^ (region.wrapping_mul(WALTONS_CONSTANT));
            let (raw_block_id, start_slot) = self.predict_location(effective_key);

            let local_idx = (raw_block_id as usize) % blocks_per_shard;
            let block_id = (shard_idx as u64 * blocks_per_shard as u64) + local_idx as u64;

            if let Ok(entry) =
                self.get_or_fetch_block(&mut shard, shard_idx, block_id, local_idx, false)
                && let Some(idx) = entry.data.find_slot(key, start_slot)
            {
                let slot = &entry.data.slots[idx];

                // Found it
                if slot.key == key {
                    return Some(slot.payload);
                }

                // Exit early if we found an empty slot AND the block isn't "full".
                if slot.key == 0 && entry.data.count <= 3500 {
                    return None;
                }
            }
        }

        // Overflow chain from region-0 block
        let (r0_block, _) = self.predict_location(key);
        let r0_local_idx = (r0_block as usize) % blocks_per_shard;
        let mut overflow_id = shard.directory[r0_local_idx].overflow_block;

        while overflow_id != 0 {
            if let Ok(entry) = self.get_or_fetch_block(&mut shard, shard_idx, overflow_id, 0, true)
            {
                if let Some(idx) = entry.data.find_slot(key, 0) {
                    if entry.data.slots[idx].key == key {
                        return Some(entry.data.slots[idx].payload);
                    }
                    if entry.data.slots[idx].key == 0 && entry.data.count <= 3500 {
                        return None;
                    }
                }
            }
            overflow_id = shard
                .overflow_meta
                .get(&overflow_id)
                .map_or(0, |m| m.overflow_block);
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
    pub fn put(&self, key: u64, data: [u8; 55]) -> std::io::Result<()> {
        let shard_idx = (key % self.shards.len() as u64) as usize;

        // In Strict mode, log to WAL first and spin until the
        // epoch advances.
        // In Async mode, skip the WAL entirely.
        if self.wal_config.mode == Durability::Strict {
            let epoch_before = self.wal_epoch.load(Ordering::Acquire);
            let _ = self.wal_tx.send(WalEntry {
                key,
                is_tombstone: false,
                payload: data,
            });
            while self.wal_epoch.load(Ordering::Acquire) <= epoch_before {
                std::thread::yield_now();
            }
        }

        let mut shard = self.shards[shard_idx].write();

        for region in 0..self.config.region_count {
            let effective_key = key ^ (region.wrapping_mul(WALTONS_CONSTANT));
            let (raw_block_id, start_slot) = self.predict_location(effective_key);

            let total_blocks = self.config.region_count * self.config.blocks_per_region;
            let blocks_per_shard = (total_blocks / self.shards.len() as u64) as usize;
            let local_idx = (raw_block_id as usize) % blocks_per_shard;
            let global_block_id = (shard_idx as u64 * blocks_per_shard as u64) + local_idx as u64;

            let should_skip = {
                let meta = &shard.directory[local_idx];
                meta.count > 3500 && !meta.might_contain(key)
            };

            if should_skip {
                continue;
            }

            let entry =
                self.get_or_fetch_block(&mut shard, shard_idx, global_block_id, local_idx, false)?;

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
                    shard.directory[local_idx].count += 1;
                }
                shard.directory[local_idx].insert(key);

                return Ok(());
            }
        }

        let total_blocks = self.config.region_count * self.config.blocks_per_region;
        let blocks_per_shard = (total_blocks / self.shards.len() as u64) as usize;

        // Before overflowing, re-check all primary blocks for key itself.
        // Prevents stale copy.
        for region in 0..self.config.region_count {
            let effective_key = key ^ (region.wrapping_mul(WALTONS_CONSTANT));
            let (raw_block_id, start_slot) = self.predict_location(effective_key);
            let local_idx = (raw_block_id as usize) % blocks_per_shard;
            let global_block_id = (shard_idx as u64 * blocks_per_shard as u64) + local_idx as u64;

            // Only check blocks whose bloom filter says the key might exist
            if !shard.directory[local_idx].might_contain(key) {
                continue;
            }

            let entry =
                self.get_or_fetch_block(&mut shard, shard_idx, global_block_id, local_idx, false)?;
            if let Some(slot_idx) = entry.data.find_slot(key, start_slot) {
                if entry.data.slots[slot_idx].key == key {
                    // Key exists in a primary block, update in-place
                    entry.data.slots[slot_idx].payload = data;
                    entry.is_dirty = true;
                    return Ok(());
                }
            }
        }

        // Overflow chain
        let (r0_block, _) = self.predict_location(key);
        let r0_local_idx = (r0_block as usize) % blocks_per_shard;
        let r0_global = (shard_idx as u64 * blocks_per_shard as u64) + r0_local_idx as u64;

        // Track previous block so we can set its overflow pointer
        let mut prev_is_primary = true;
        let mut prev_block_id: u64 = r0_global;
        let prev_local_idx: usize = r0_local_idx;
        let mut current_overflow: u64 = shard.directory[r0_local_idx].overflow_block;

        // Walk existing overflow chain looking for space
        while current_overflow != 0 {
            let entry =
                self.get_or_fetch_block(&mut shard, shard_idx, current_overflow, 0, true)?;

            if let Some(slot_idx) = entry.data.find_slot(key, 0) {
                let slot_key = entry.data.slots[slot_idx].key;

                // Found the key (update) or an empty slot in a non-full block (insert)
                if slot_key == key {
                    entry.data.slots[slot_idx].payload = data;
                    entry.is_dirty = true;
                    return Ok(());
                }
                if slot_key == 0 && entry.data.count <= 3500 {
                    entry.data.count += 1;
                    entry.data.slots[slot_idx].key = key;
                    entry.data.slots[slot_idx].payload = data;
                    entry.is_dirty = true;
                    let meta = shard.overflow_meta.get_mut(&current_overflow).unwrap();
                    meta.count += 1;
                    meta.insert(key);
                    return Ok(());
                }
            }

            // This overflow block is also full, keep walking
            prev_is_primary = false;
            prev_block_id = current_overflow;
            current_overflow = shard
                .overflow_meta
                .get(&current_overflow)
                .map_or(0, |m| m.overflow_block);
        }

        // No space anywhere, allocate a new overflow block
        let new_block_id = self.alloc_overflow_block()?;

        // Link it from the previous block's overflow pointer
        if prev_is_primary {
            shard.directory[prev_local_idx].overflow_block = new_block_id;
        } else {
            shard
                .overflow_meta
                .get_mut(&prev_block_id)
                .unwrap()
                .overflow_block = new_block_id;
        }

        // Create metadata for new block
        let mut new_meta = BlockMetadata::new();
        new_meta.count = 1;
        new_meta.insert(key);
        shard.overflow_meta.insert(new_block_id, new_meta);

        // Fetch the new (zeroed) block and write the first slot
        let entry = self.get_or_fetch_block(&mut shard, shard_idx, new_block_id, 0, true)?;
        entry.data.slots[0].key = key;
        entry.data.slots[0].payload = data;
        entry.data.count = 1;
        entry.is_dirty = true;

        Ok(())
    }

    pub fn sync(&self) -> std::io::Result<()> {
        for shard_lock in &self.shards {
            let mut shard = shard_lock.write();
            for (&block_id, entry) in shard.pool.iter_mut() {
                if entry.is_dirty {
                    let offset = block_id * (ERASE_BLOCK_SIZE as u64);
                    seek_write(&self.handle, entry.data.as_slice_u8(), offset)?;
                    entry.is_dirty = false;
                }
            }
        }
        self.handle.sync_all()?;
        Ok(())
    }

    pub fn background_sync(&mut self) {
        for (shard_idx, shard_lock) in self.shards.iter().enumerate() {
            let mut shard = shard_lock.write();
            for (&block_id, entry) in shard.pool.iter_mut() {
                if entry.is_dirty {
                    let snapshot = entry.data.clone();
                    entry.is_dirty = false;
                    let _ = self.flusher_txs[shard_idx].send((block_id, snapshot));
                }
            }
        }
    }

    /// Recover from WAL file.
    fn recover_from_wal(&mut self, wal_handle: &mut File) {
        let num_shards = self.shards.len();
        let total_blocks = (self.config.region_count * self.config.blocks_per_region) as usize;
        let blocks_per_shard = total_blocks / num_shards;
        let wal_pages = (WAL_FILE_SIZE as usize) / 4096;

        // Read all entries and group by shard.
        // This avoids constantly locking/unlocking different shards.
        let mut shard_entries: Vec<Vec<WalEntry>> =
            (0..num_shards).map(|_| Vec::new()).collect();

        let mut read_buffer = [0u8; 4096];
        for page in 0..wal_pages {
            let offset = (page * 4096) as u64;
            let bytes_read = seek_read(wal_handle, &mut read_buffer, offset).unwrap_or(0);
            if bytes_read < 4096 {
                continue;
            }

            let entries: &[WalEntry; 64] = unsafe { std::mem::transmute(&read_buffer) };
            for entry in entries.iter() {
                if entry.key == 0 {
                    continue;
                }
                let shard_idx = (entry.key % num_shards as u64) as usize;
                shard_entries[shard_idx].push(*entry);
            }
        }

        // Replay each shard's entries while holding that shard's lock.
        for (shard_idx, entries) in shard_entries.into_iter().enumerate() {
            if entries.is_empty() {
                continue;
            }

            let shard_arc = &self.shards[shard_idx];
            let mut shard = shard_arc.write();

            for entry in &entries {
                let key = entry.key;
                let mut inserted = false;

                for region in 0..self.config.region_count {
                    let effective_key = key ^ (region.wrapping_mul(WALTONS_CONSTANT));
                    let (raw_block_id, start_slot) = self.predict_location(effective_key);

                    let local_idx = (raw_block_id as usize) % blocks_per_shard;
                    let global_block_id =
                        (shard_idx as u64 * blocks_per_shard as u64) + local_idx as u64;

                    if let Ok(block_entry) = self.get_or_fetch_block(
                        &mut shard,
                        shard_idx,
                        global_block_id,
                        local_idx,
                        false,
                    ) {
                        if block_entry.data.count > 3500 {
                            continue;
                        }

                        if let Some(slot_idx) = block_entry.data.find_slot(key, start_slot) {
                            let is_empty_slot = block_entry.data.slots[slot_idx].key == 0;

                            block_entry.data.slots[slot_idx].key = key;
                            block_entry.data.slots[slot_idx].payload = entry.payload;
                            block_entry.is_dirty = true;

                            if is_empty_slot {
                                block_entry.data.count += 1;
                                shard.directory[local_idx].count += 1;
                            }

                            shard.directory[local_idx].insert(key);
                            inserted = true;
                            break;
                        }
                    }
                }

                if inserted {
                    continue;
                }

                // Overflow chain from region-0 block
                let (r0_block, _) = self.predict_location(key);
                let r0_local_idx = (r0_block as usize) % blocks_per_shard;

                let mut prev_is_primary = true;
                let mut prev_block_id: u64 =
                    (shard_idx as u64 * blocks_per_shard as u64) + r0_local_idx as u64;
                let prev_local_idx: usize = r0_local_idx;
                let mut current_overflow: u64 = shard.directory[r0_local_idx].overflow_block;

                while current_overflow != 0 {
                    if let Ok(block_entry) =
                        self.get_or_fetch_block(&mut shard, shard_idx, current_overflow, 0, true)
                    {
                        if let Some(slot_idx) = block_entry.data.find_slot(key, 0) {
                            let slot_key = block_entry.data.slots[slot_idx].key;
                            if slot_key == key {
                                block_entry.data.slots[slot_idx].payload = entry.payload;
                                block_entry.is_dirty = true;
                                inserted = true;
                                break;
                            }
                            if slot_key == 0 && block_entry.data.count <= 3500 {
                                block_entry.data.count += 1;
                                block_entry.data.slots[slot_idx].key = key;
                                block_entry.data.slots[slot_idx].payload = entry.payload;
                                block_entry.is_dirty = true;
                                let meta = shard.overflow_meta.get_mut(&current_overflow).unwrap();
                                meta.count += 1;
                                meta.insert(key);
                                inserted = true;
                                break;
                            }
                        }
                    }
                    prev_is_primary = false;
                    prev_block_id = current_overflow;
                    current_overflow = shard
                        .overflow_meta
                        .get(&current_overflow)
                        .map_or(0, |m| m.overflow_block);
                }

                if !inserted {
                    if let Ok(new_block_id) = self.alloc_overflow_block() {
                        if prev_is_primary {
                            shard.directory[prev_local_idx].overflow_block = new_block_id;
                        } else {
                            shard
                                .overflow_meta
                                .get_mut(&prev_block_id)
                                .unwrap()
                                .overflow_block = new_block_id;
                        }

                        let mut new_meta = BlockMetadata::new();
                        new_meta.count = 1;
                        new_meta.insert(key);
                        shard.overflow_meta.insert(new_block_id, new_meta);

                        if let Ok(block_entry) =
                            self.get_or_fetch_block(&mut shard, shard_idx, new_block_id, 0, true)
                        {
                            block_entry.data.slots[0].key = key;
                            block_entry.data.slots[0].payload = entry.payload;
                            block_entry.data.count = 1;
                            block_entry.is_dirty = true;
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn start_linux_flusher(
        bg_handle: File,
        receivers: Vec<crossbeam_channel::Receiver<(u64, AlignedBlock)>>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            use io_uring::IoUring;
            use std::os::unix::io::AsRawFd;

            let mut ring: IoUring = IoUring::builder()
                .setup_sqpoll(2000)
                .build(256)
                .expect("failed to init uring");

            let fd = bg_handle.as_raw_fd();
            let mut in_flight: HashMap<u64, AlignedBlock> = HashMap::new();

            loop {
                let mut activity = false;
                let mut active_channels = 0;

                // Check every shard's private pipe
                for rx in &receivers {
                    loop {
                        match rx.try_recv() {
                            Ok((id, block)) => {
                                activity = true;
                                active_channels += 1;
                                Self::push_to_ring(&mut ring, &mut in_flight, fd, id, block);

                                // Prevent the ring from overflowing
                                if in_flight.len() >= 128 {
                                    let _ = ring.submit_and_wait(1);
                                    Self::reap_completions(&mut ring, &mut in_flight);
                                }
                            }
                            Err(crossbeam_channel::TryRecvError::Empty) => {
                                active_channels += 1;
                                break;
                            }
                            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                                break; // Sender was dropped in LiveWire::drop
                            }
                        }
                    }
                }

                if activity {
                    let _ = ring.submit();
                    Self::reap_completions(&mut ring, &mut in_flight);
                } else {
                    Self::reap_completions(&mut ring, &mut in_flight);

                    // Shutdown cleanly when the engine drops
                    if active_channels == 0 && in_flight.is_empty() {
                        break;
                    }

                    if in_flight.is_empty() {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
        })
    }

    #[cfg(target_os = "linux")]
    fn push_to_ring(
        ring: &mut io_uring::IoUring,
        in_flight: &mut HashMap<u64, AlignedBlock>,
        fd: i32,
        id: u64,
        block: AlignedBlock,
    ) {
        let offset = id * (ERASE_BLOCK_SIZE as u64);
        let write_e = io_uring::opcode::Write::new(
            io_uring::types::Fd(fd),
            block.ptr as *const u8,
            ERASE_BLOCK_SIZE as u32,
        )
        .offset(offset)
        .build()
        .user_data(id);

        unsafe {
            let _ = ring.submission().push(&write_e);
        }
        in_flight.insert(id, block);
    }

    #[cfg(target_os = "linux")]
    fn reap_completions(ring: &mut io_uring::IoUring, in_flight: &mut HashMap<u64, AlignedBlock>) {
        let mut cq = ring.completion();
        while let Some(cqe) = cq.next() {
            in_flight.remove(&cqe.user_data());
        }
    }

    /// Fire-and-forget write. Queues to WAL and inserts to memory
    /// immediately without waiting for the physical sync.
    /// Call flush_wal() later to guarantee durability.
    pub fn put_async(&self, key: u64, data: [u8; 55]) -> std::io::Result<()> {
        let shard_idx = (key % self.shards.len() as u64) as usize;

        // Only log to WAL in Strict mode. Async skips the
        // channel entirely to avoid back-pressure.
        if self.wal_config.mode == Durability::Strict {
            let _ = self.wal_tx.send(WalEntry {
                key,
                is_tombstone: false,
                payload: data,
            });
        }

        // Memory insert immediately
        let mut shard = self.shards[shard_idx].write();
        self.insert_to_memory(&mut shard, shard_idx, key, data)?;

        Ok(())
    }

    /// Vectorized write. Queues entries to the WAL and inserts
    /// to memory in one shot. Does NOT block for the physical
    /// sync, call flush_wal() after submitting all shards.
    pub fn put_batch(&self, shard_idx: usize, entries: &[(u64, [u8; 55])]) -> std::io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if self.wal_config.mode == Durability::Strict {
            for &(key, payload) in entries {
                let _ = self.wal_tx.send(WalEntry {
                    key,
                    is_tombstone: false,
                    payload,
                });
            }
        }

        // Grab the shard lock once for the entire batch
        let mut shard = self.shards[shard_idx].write();
        for &(key, payload) in entries {
            self.insert_to_memory(&mut shard, shard_idx, key, payload)?;
        }

        Ok(())
    }

    /// Blocks until every WAL entry sent before
    /// this call is physically on the device.
    pub fn flush_wal(&self) {
        if self.wal_config.mode != Durability::Strict {
            return;
        }

        // Read epoch before sending.
        let epoch_before = self.wal_epoch.load(Ordering::Acquire);
        let _ = self.wal_tx.send(WalEntry {
            key: 0,
            is_tombstone: false,
            payload: [0u8; 55],
        });

        // Spin until the epoch advances. When it does, at least
        // one full batch (including our fence) has been synced.
        while self.wal_epoch.load(Ordering::Acquire) <= epoch_before {
            std::thread::yield_now();
        }
    }

    fn insert_to_memory(
        &self,
        shard: &mut parking_lot::RwLockWriteGuard<'_, Shard>,
        shard_idx: usize,
        key: u64,
        data: [u8; 55],
    ) -> std::io::Result<()> {
        for region in 0..self.config.region_count {
            let effective_key = key ^ (region.wrapping_mul(WALTONS_CONSTANT));
            let (raw_block_id, start_slot) = self.predict_location(effective_key);

            let total_blocks = self.config.region_count * self.config.blocks_per_region;
            let blocks_per_shard = (total_blocks / self.shards.len() as u64) as usize;
            let local_idx = (raw_block_id as usize) % blocks_per_shard;
            let global_block_id = (shard_idx as u64 * blocks_per_shard as u64) + local_idx as u64;

            let should_skip = {
                let meta = &shard.directory[local_idx];
                meta.count > 3500 && !meta.might_contain(key)
            };

            if should_skip {
                continue;
            }

            let entry =
                self.get_or_fetch_block(shard, shard_idx, global_block_id, local_idx, false)?;

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
                    shard.directory[local_idx].count += 1;
                }
                shard.directory[local_idx].insert(key);

                return Ok(());
            }
        }

        // Dedup check
        let total_blocks = self.config.region_count * self.config.blocks_per_region;
        let blocks_per_shard = (total_blocks / self.shards.len() as u64) as usize;

        for region in 0..self.config.region_count {
            let effective_key = key ^ (region.wrapping_mul(WALTONS_CONSTANT));
            let (raw_block_id, start_slot) = self.predict_location(effective_key);
            let local_idx = (raw_block_id as usize) % blocks_per_shard;
            let global_block_id = (shard_idx as u64 * blocks_per_shard as u64) + local_idx as u64;

            if !shard.directory[local_idx].might_contain(key) {
                continue;
            }

            let entry =
                self.get_or_fetch_block(shard, shard_idx, global_block_id, local_idx, false)?;
            if let Some(slot_idx) = entry.data.find_slot(key, start_slot) {
                if entry.data.slots[slot_idx].key == key {
                    entry.data.slots[slot_idx].payload = data;
                    entry.is_dirty = true;
                    return Ok(());
                }
            }
        }

        // Overflow chain from region-0 block
        let (r0_block, _) = self.predict_location(key);
        let r0_local_idx = (r0_block as usize) % blocks_per_shard;

        let mut prev_is_primary = true;
        let mut prev_block_id: u64 =
            (shard_idx as u64 * blocks_per_shard as u64) + r0_local_idx as u64;
        let prev_local_idx: usize = r0_local_idx;
        let mut current_overflow: u64 = shard.directory[r0_local_idx].overflow_block;

        while current_overflow != 0 {
            let entry = self.get_or_fetch_block(shard, shard_idx, current_overflow, 0, true)?;

            if let Some(slot_idx) = entry.data.find_slot(key, 0) {
                let slot_key = entry.data.slots[slot_idx].key;

                if slot_key == key {
                    entry.data.slots[slot_idx].payload = data;
                    entry.is_dirty = true;
                    return Ok(());
                }
                if slot_key == 0 && entry.data.count <= 3500 {
                    entry.data.count += 1;
                    entry.data.slots[slot_idx].key = key;
                    entry.data.slots[slot_idx].payload = data;
                    entry.is_dirty = true;
                    let meta = shard.overflow_meta.get_mut(&current_overflow).unwrap();
                    meta.count += 1;
                    meta.insert(key);
                    return Ok(());
                }
            }

            prev_is_primary = false;
            prev_block_id = current_overflow;
            current_overflow = shard
                .overflow_meta
                .get(&current_overflow)
                .map_or(0, |m| m.overflow_block);
        }

        let new_block_id = self.alloc_overflow_block()?;

        if prev_is_primary {
            shard.directory[prev_local_idx].overflow_block = new_block_id;
        } else {
            shard
                .overflow_meta
                .get_mut(&prev_block_id)
                .unwrap()
                .overflow_block = new_block_id;
        }

        let mut new_meta = BlockMetadata::new();
        new_meta.count = 1;
        new_meta.insert(key);
        shard.overflow_meta.insert(new_block_id, new_meta);

        let entry = self.get_or_fetch_block(shard, shard_idx, new_block_id, 0, true)?;
        entry.data.slots[0].key = key;
        entry.data.slots[0].payload = data;
        entry.data.count = 1;
        entry.is_dirty = true;

        Ok(())
    }

    fn alloc_overflow_block(&self) -> std::io::Result<u64> {
        let block_id = self.next_free_block.fetch_add(1, Ordering::Relaxed);
        let required_size = (block_id + 1) * ERASE_BLOCK_SIZE as u64;
        let current_size = self.handle.metadata()?.len();
        if required_size > current_size {
            let grow_chunk = 256 * 1024 * 1024u64; // 256MB at a time
            let new_size = required_size.next_multiple_of(grow_chunk);
            self.handle.set_len(new_size)?;
        }
        Ok(block_id)
    }
}

impl Drop for LiveWire {
    fn drop(&mut self) {
        for shard_lock in &self.shards {
            let mut shard = shard_lock.write();
            for (&block_id, entry) in shard.pool.iter_mut() {
                if entry.is_dirty {
                    let offset = block_id * (ERASE_BLOCK_SIZE as u64);
                    let _ = seek_write(&self.handle, entry.data.as_slice_u8(), offset);
                    entry.is_dirty = false;
                }
            }
        }
        let _ = self.handle.sync_all();

        self.flusher_txs.clear();
        if let Some(handle) = self.flusher_thread.take() {
            let _ = handle.join();
        }

        // Drop the sender to signal the WAL thread to exit,
        // then join it so all queued entries get flushed.
        drop(std::mem::replace(
            &mut self.wal_tx,
            crossbeam_channel::bounded::<WalEntry>(1).0,
        ));
        if let Some(handle) = self.wal_thread.take() {
            let _ = handle.join();
        }

        // Zero out the WAL
        let wal_path = self.config.data_dir.join("main.wal");
        if let Ok(wal_handle) = OpenOptions::new().write(true).open(wal_path) {
            let _ = wal_handle.set_len(0);
            let _ = wal_handle.set_len(WAL_FILE_SIZE);
        }

        let meta_path = self.config.data_dir.join("main.meta");
        if let Ok(meta_file) = File::create(meta_path) {
            let mut writer = BufWriter::new(meta_file);

            for shard_lock in &self.shards {
                let shard = shard_lock.read();
                for meta in &shard.directory {
                    let _ = writer.write_all(&meta.count.to_le_bytes());
                    let _ = writer.write_all(&*meta.bloom);
                    let _ = writer.write_all(&meta.overflow_block.to_le_bytes());
                }
                let overflow_count = shard.overflow_meta.len() as u64;
                let _ = writer.write_all(&overflow_count.to_le_bytes());
                for (&block_id, meta) in &shard.overflow_meta {
                    let _ = writer.write_all(&block_id.to_le_bytes());
                    let _ = writer.write_all(&meta.count.to_le_bytes());
                    let _ = writer.write_all(&*meta.bloom);
                    let _ = writer.write_all(&meta.overflow_block.to_le_bytes());
                }
            }
            let _ = writer.write_all(&self.next_free_block.load(Ordering::Relaxed).to_le_bytes());
            let _ = writer.flush();
        }
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
    pub inner: Arc<LiveWire>,
}

impl JsonStore {
    pub fn new(inner: Arc<LiveWire>) -> Self {
        Self { inner }
    }

    pub fn put(&mut self, key: &str, value: &serde_json::Value) -> std::io::Result<()> {
        let hash = fnv1a(key);
        let serialized = serde_json::to_vec(value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let total_len = serialized.len() as u32;

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

    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
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
            let num_chunks = remaining.div_ceil(OVERFLOW_CHUNK_SIZE);

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

        if let Some(payload) = self.inner.get(hash) {
            let flags = payload[0];
            let total_len =
                u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]) as usize;

            if flags == FLAG_OVERFLOW && total_len > INLINE_DATA_SIZE {
                let remaining = total_len - INLINE_DATA_SIZE;
                let num_chunks = remaining.div_ceil(OVERFLOW_CHUNK_SIZE);
                let empty = [0u8; 55];
                for i in 0..num_chunks {
                    self.inner.put(overflow_key(hash, i as u64), empty)?;
                }
            }

            self.inner.put(hash, [0u8; 55])?;
        }

        Ok(())
    }

    pub fn sync(&mut self) -> std::io::Result<()> {
        self.inner.sync()
    }
}
