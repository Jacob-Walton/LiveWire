use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct LiveWireConfig {
    /// How many 265KB blocks to keep in RAM.
    /// 4096 = 1GB of RAM.
    pub pool_capacity: usize,

    /// How many blocks exist in a single "Region".
    /// 512 = 128MB per region.
    pub blocks_per_region: u64,

    /// How many regions the file is currently allowed to use.
    pub region_count: u64,

    /// How many operations before we automatically flush dirty blocks to disk
    pub auto_sync_threshold: u64,

    /// Directory to store files in
    pub data_dir: PathBuf,

    /// Number of shards to use
    pub num_shards: usize,
}

impl Default for LiveWireConfig {
    fn default() -> Self {
        Self {
            pool_capacity: 8192,    // 2GB memory footprint
            blocks_per_region: 512, // 128MB regions
            region_count: 32,       // 4GB total disk space
            auto_sync_threshold: 50_000,
            data_dir: PathBuf::from("./data"),
            num_shards: 16,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// Fire-and-foreget. Returns in < 200ns. Vulernable to sudden power loss.
    Async,
    /// Blocks until the SSD physically writes the log. Safe, but slow.
    Strict,
}

#[derive(Clone, Copy)]
pub struct WalConfig {
    pub mode: Durability,
    pub max_batch_size: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            mode: Durability::Strict,
            max_batch_size: 128,
        }
    }
}
