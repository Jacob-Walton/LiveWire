use std::{fs::OpenOptions, time::Instant};

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;

use colored::Colorize;
use live_wire::*;

fn main() {
    #[cfg(windows)]
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .custom_flags(live_wire::FILE_FLAG_NO_BUFFERING)
        .open("main.lw")
        .unwrap();

    #[cfg(unix)]
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open("main.lw")
        .unwrap();

    #[cfg(unix)]
    enable_direct_io(&file).unwrap();

    let live_wire_config = LiveWireConfig::default();
    let wal_config = WalConfig {
        mode: Durability::Async,
        max_batch_size: 64,
    };
    let live_wire =
        LiveWire::new(file, live_wire_config, wal_config).expect("Failed to initialize LiveWire");
    let mut store = JsonStore::new(live_wire);

    // Small documents (fit inline, < 50 bytes)
    let small_docs = vec![
        ("user:1", serde_json::json!({"name": "Alice", "age": 30})),
        ("user:2", serde_json::json!({"name": "Bob", "age": 25})),
        ("user:3", serde_json::json!({"name": "Charlie", "age": 35})),
    ];

    // Larger document (overflows into extra slots)
    let large_doc = serde_json::json!({
        "id": 1001,
        "name": "Warehouse Alpha",
        "location": {
            "city": "Manchester",
            "country": "United Kingdom",
            "postcode": "M1 1AA"
        },
        "inventory": [
            {"sku": "WR-001", "name": "Widget", "qty": 5000},
            {"sku": "GZ-042", "name": "Gizmo", "qty": 1200},
            {"sku": "TH-099", "name": "Thingamajig", "qty": 300}
        ],
        "active": true
    });

    println!("{}", "Writing small documents".bold());
    for (key, doc) in &small_docs {
        store.put(key, doc).unwrap();
        let serialized = serde_json::to_vec(doc).unwrap();
        println!("  {} -> {} bytes", key, serialized.len());
    }

    println!("\n{}", "Writing large document".bold());
    let serialized = serde_json::to_vec(&large_doc).unwrap();
    println!("  warehouse:1 -> {} bytes", serialized.len());
    store.put("warehouse:1", &large_doc).unwrap();

    println!("\n{}", "Reading back".bold());
    for (key, _) in &small_docs {
        let result = store.get(key);
        println!("  {} -> {}", key, result.unwrap());
    }

    let result = store.get("warehouse:1");
    println!("  warehouse:1 -> {}", result.unwrap());

    // Verify a missing key
    let missing = store.get("nonexistent");
    println!("\n  nonexistent -> {:?}", missing);

    // Sustained load test
    let total = 10_000_000;
    let batch = 1_000_000;

    println!(
        "\n{}",
        format!("Sustained load: {} JSON documents\n", total).bold()
    );

    // Puts in batches
    let overall_start = Instant::now();
    for b in 0..(total / batch) {
        let start = Instant::now();
        for i in (b * batch)..((b + 1) * batch) {
            let doc = serde_json::json!({"id": i, "val": i * 7});
            let key = format!("bench:{}", i);
            store.put(&key, &doc).unwrap();
        }
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
        for i in (b * batch)..((b + 1) * batch) {
            let key = format!("bench:{}", i);
            store.get(&key).unwrap();
        }
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

    store.sync().unwrap();

    // std::fs::remove_file("main.lw").unwrap();
}
