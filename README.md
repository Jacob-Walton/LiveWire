# LiveWire

Rust-based document store which aims to optimise for SSDs rather than spinning
hard drives.

## How It Works

Most DBs use trees (B-Trees/LSMs). LiveWire uses mathematical scattering. Instead
of looking up where a key is, we calculate its address using a hash of the key
and a constant.

All entries are exactly 64 bytes which means that one slot = one CPU cache line.

## Stats (7950X3D / Gen5 NVMe)

### Async

- **PUT**: ~65ns/op (sustained)
- **GET**: ~34ns/op (sustained)

### Strict

- **PUT**: ~229ns/op (sustained)
- **GET**: ~32ns/op (sustained)

## Trade-offs

Because we can only optimise for two of three things (Read, Update, Memory),
we sacrifice space to gain O(1) speed. This means that LiveWire will use more
memory than most other data structures, but still within reasonable limits.

## Limits

We currently have no concurrency support and no support for asynchronous I/O.
This means that for Strict mode we have a massive, avoidable performance hit for
PUT.
