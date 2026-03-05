#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct WalEntry {
    pub key: u64,
    pub is_tombstone: bool,
    pub payload: [u8; 55],
}
