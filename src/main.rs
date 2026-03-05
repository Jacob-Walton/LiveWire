use colored::Colorize;
use live_wire::*;
use rayon::prelude::*;
use std::{env, path::PathBuf, sync::Arc, time::Instant};

fn main() {
    let data_path: PathBuf = env::var("LIVE_WIRE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./data"));
    let abs_path = if data_path.is_absolute() {
        data_path
    } else {
        env::current_dir().unwrap().join(data_path)
    };

    let live_wire_config = LiveWireConfig {
        data_dir: abs_path,
        num_shards: 16,
        pool_capacity: 8192,
        ..Default::default()
    };
    let wal_config = WalConfig {
        mode: Durability::Async,
        max_batch_size: 8192,
    };

    let live_wire = Arc::new(
        LiveWire::new(live_wire_config, wal_config).expect("Failed to initialize LiveWire"),
    );

    // Create a JsonStore.
    let mut store = JsonStore::new(Arc::clone(&live_wire));

    // Small documents (fit inline, < 50 bytes)
    let small_docs = vec![
        ("user:1", serde_json::json!({"name": "Alice", "age": 30})),
        ("user:2", serde_json::json!({"name": "Bob", "age": 25})),
        ("user:3", serde_json::json!({"name": "Charlie", "age": 35})),
    ];

    println!("{}", "Writing small documents".bold());
    for (key, doc) in &small_docs {
        store.put(key, doc).unwrap();
        let serialized = serde_json::to_vec(doc).unwrap();
        println!("  {} -> {} bytes", key, serialized.len());
    }

    println!("\n{}", "Reading back".bold());
    for (key, _) in &small_docs {
        let result = store.get(key);
        println!("  {} -> {}", key, result.unwrap());
    }

    // Sustained load test
    let total = 10_000_000;
    let batch = 100_000;
    let num_shards = 16;

    println!(
        "\n{}",
        format!("Sustained load: {} JSON documents\n", total).bold()
    );

    // Puts in batches
    let overall_start = Instant::now();
    for b in 0..(total / batch) {
        let start = Instant::now();

        // Give each Rayon thread 2048 documents to process at a time
        let range: Vec<usize> = ((b * batch)..((b + 1) * batch)).collect();
        range.par_chunks(2048).for_each(|chunk| {
            // Thread-local queues for each shard
            let mut shard_queues: Vec<Vec<(u64, [u8; 55])>> =
                vec![Vec::with_capacity(256); num_shards];

            // Hash and build payloads
            for &id in chunk {
                let mut key_buf = [0u8; 32];
                let key_str = format_to_buf(&mut key_buf, "bench:", id);
                let hash = fnv1a(key_str);

                let mut payload = [0u8; 55];
                payload[0] = 0x00;
                let len_bytes = 12u32.to_le_bytes();
                payload[1..5].copy_from_slice(&len_bytes);
                payload[5..13].copy_from_slice(&(id as u64).to_le_bytes());

                let shard_idx = (hash % num_shards as u64) as usize;
                shard_queues[shard_idx].push((hash, payload));
            }

            // Execute batches
            for (shard_idx, queue) in shard_queues.iter().enumerate() {
                live_wire.put_batch(shard_idx, queue).unwrap();
            }
        });

        let dur = start.elapsed();
        let so_far = (b + 1) * batch;
        println!(
            "  PUT batch {}/{}: {:?} ({:?}/op) | {} total written",
            b + 1,
            total / batch,
            dur,
            dur / batch as u32,
            so_far
        );
    }

    let put_total = overall_start.elapsed();
    println!(
        "\n  PUT total: {:?} ({:?}/op)\n",
        put_total,
        put_total / total as u32
    );

    // Gets in batches
    let overall_start = Instant::now();
    for b in 0..(total / batch) {
        let start = Instant::now();

        (0..batch).into_par_iter().for_each(|i| {
            let id = (b * batch) + i;
            let mut key_buf = [0u8; 32];
            let key_str = format_to_buf(&mut key_buf, "bench:", id);
            let hash = fnv1a(key_str);

            let _ = live_wire.get(hash);
        });

        let dur = start.elapsed();
        let so_far = (b + 1) * batch;
        println!(
            "  GET batch {}/{}: {:?} ({:?}/op) | {} total read",
            b + 1,
            total / batch,
            dur,
            dur / batch as u32,
            so_far
        );
    }
    let get_total = overall_start.elapsed();
    println!(
        "\n  GET total: {:?} ({:?}/op)",
        get_total,
        get_total / total as u32
    );

    // Final safety sync
    live_wire.sync().unwrap();
}

/// Helper to format strings into a stack buffer to avoid heap allocation
fn format_to_buf<'a>(buf: &'a mut [u8], prefix: &str, id: usize) -> &'a str {
    use std::io::Write;
    let pos = {
        let mut cursor = std::io::Cursor::new(&mut *buf);
        let _ = write!(cursor, "{}{}", prefix, id);
        cursor.position() as usize
    };
    std::str::from_utf8(&buf[..pos]).unwrap()
}

/// FNV-1a hash for string keys
fn fnv1a(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash & !(1u64 << 63)
}
