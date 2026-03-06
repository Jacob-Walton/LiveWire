use std::fs::File;

#[cfg(windows)]
pub const FILE_FLAG_NO_BUFFERING: u32 = 0x20000000;
#[cfg(windows)]
pub const FILE_FLAG_WRITE_THROUGH: u32 = 0x80000000;

#[cfg(unix)]
pub const O_DIRECT: i32 = 0o40000;

#[cfg(target_os = "macos")]
pub const F_NOCACHE: i32 = 48;

#[cfg(unix)]
pub fn enable_direct_io(handle: &File) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;

    let fd = handle.as_raw_fd();

    #[cfg(target_os = "macos")]
    {
        unsafe {
            if libc::fcntl(fd, F_NOCACHE, 1) == -1 {
                return Err(std::io::Error::last_os_error());
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            libc::fcntl(fd, libc::F_SETFL, flags | libc::O_DIRECT);
        }
    }

    Ok(())
}

pub(crate) fn seek_write(handle: &File, data: &[u8], offset: u64) -> std::io::Result<()> {
    let mut written = 0;

    while written < data.len() {
        let n = {
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;

                handle.write_at(&data[written..], offset + written as u64)?
            }

            #[cfg(windows)]
            {
                use std::os::windows::fs::FileExt;

                handle.seek_write(&data[written..], offset + written as u64)?
            }
        };

        written += n;
    }

    Ok(())
}

pub(crate) fn seek_read(handle: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    let mut read_total = 0;

    while read_total < buf.len() {
        let n = {
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;

                handle.read_at(&mut buf[read_total..], offset + read_total as u64)?
            }

            #[cfg(windows)]
            {
                use std::os::windows::fs::FileExt;

                handle.seek_read(&mut buf[read_total..], offset + read_total as u64)?
            }
        };

        if n == 0 {
            break; // EOF
        }

        read_total += n;
    }

    Ok(read_total)
}
