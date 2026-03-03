use std::{fs::OpenOptions, os::windows::fs::OpenOptionsExt, time::Instant};

use live_wire::*;

const FILE_FLAG_NO_BUFFERING: u32 = 0x20000000;

fn main() {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING)
        .open("main.lw")
        .unwrap();

    file.set_len(128 * 1024 * 1024).unwrap();

    let live_wire = LiveWire::new(file);

    let iterations = 10_000;

    let start = Instant::now();
    for i in 0..iterations {
        let mut data = [0u8; 55];
        data[0] = (i % 255) as u8;
        live_wire.put(i as u64, data).unwrap();
    }
    let duration = start.elapsed();
    println!(
        "Puts: {} iterations in {:?}, ({:?} per op)",
        iterations,
        duration,
        duration / iterations as u32
    );

    let start = Instant::now();
    for i in 0..iterations {
        live_wire.get(i as u64).unwrap();
    }
    let duration = start.elapsed();
    println!(
        "Gets: {} iterations in {:?}, ({:?} per op)",
        iterations,
        duration,
        duration / iterations as u32
    );

    std::fs::remove_file("main.lw").unwrap();
}
