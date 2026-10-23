//! Small native (C++) helpers; see native/guard.cpp.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;

extern "C-unwind" {
    fn kalam_guarded_call(
        f: unsafe extern "C-unwind" fn(*mut c_void),
        data: *mut c_void,
        msg: *mut c_char,
        msg_len: usize,
    ) -> i32;
}

extern "C" {
    fn kalam_redirect_stderr(path_utf8: *const c_char) -> i32;
    fn kalam_available_memory_mb() -> i64;
}

#[cfg(test)]
extern "C-unwind" {
    fn kalam_test_throw(message: *const c_char);
}

/// Runs `f`, turning a C++ exception thrown inside it into `Err(message)`.
///
/// whisper.cpp's GPU backends throw C++ exceptions when the graphics driver
/// fails (for example when it runs out of memory). Unwinding into Rust aborts
/// the whole app, so every GPU call goes through here. After an `Err`, the
/// whisper state that was in use must not be touched again.
pub fn guard<R, F: FnOnce() -> R>(f: F) -> Result<R, String> {
    unsafe extern "C-unwind" fn call<R, F: FnOnce() -> R>(data: *mut c_void) {
        let slot = unsafe { &mut *data.cast::<(Option<F>, Option<R>)>() };
        if let Some(f) = slot.0.take() {
            slot.1 = Some(f());
        }
    }
    let mut slot: (Option<F>, Option<R>) = (Some(f), None);
    let mut msg = [0 as c_char; 512];
    let data = (&mut slot as *mut (Option<F>, Option<R>)).cast::<c_void>();
    let threw = unsafe { kalam_guarded_call(call::<R, F>, data, msg.as_mut_ptr(), msg.len()) } != 0;
    if threw {
        let text = unsafe { CStr::from_ptr(msg.as_ptr()) }.to_string_lossy().trim().to_string();
        return Err(if text.is_empty() { "unknown error".into() } else { text });
    }
    Ok(slot.1.take().expect("guarded call returned a value"))
}

/// Sends standard error (whisper.cpp, the GPU backend, Rust panics) to `path`.
pub fn redirect_stderr(path: &Path) -> bool {
    let Ok(path) = CString::new(path.to_string_lossy().as_bytes()) else { return false };
    unsafe { kalam_redirect_stderr(path.as_ptr()) == 0 }
}

/// Memory still available to new allocations, in MB (Windows only).
pub fn available_memory_mb() -> Option<u64> {
    let mb = unsafe { kalam_available_memory_mb() };
    (mb >= 0).then_some(mb as u64)
}

/// For tests: throw a C++ exception, as the GPU backend does when the driver fails.
#[cfg(test)]
pub fn throw_for_test(message: &str) {
    let message = CString::new(message).unwrap();
    unsafe { kalam_test_throw(message.as_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_returns_the_value() {
        assert_eq!(guard(|| 6 * 7), Ok(42));
    }

    #[test]
    fn guard_catches_cpp_exceptions() {
        let message = CString::new("vk::Device::allocateMemory: ErrorOutOfDeviceMemory").unwrap();
        let result = guard(|| unsafe { kalam_test_throw(message.as_ptr()) });
        assert_eq!(result, Err("vk::Device::allocateMemory: ErrorOutOfDeviceMemory".to_string()));
        // The guard still works afterwards.
        assert_eq!(guard(|| "ok"), Ok("ok"));
    }

    #[test]
    fn guard_lets_rust_panics_through() {
        let caught = std::panic::catch_unwind(|| guard(|| panic!("a Rust bug")));
        assert!(caught.is_err(), "a panic must not be swallowed as a GPU error");
    }

    #[test]
    fn available_memory_is_reported_on_windows() {
        if cfg!(windows) {
            assert!(available_memory_mb().is_some_and(|mb| mb > 0));
        }
    }
}
