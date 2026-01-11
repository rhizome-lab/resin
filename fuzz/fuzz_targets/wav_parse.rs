#![no_main]

use libfuzzer_sys::fuzz_target;
use rhizome_resin_audio::wav_from_bytes;

fuzz_target!(|data: &[u8]| {
    // wav_from_bytes should never panic on any input
    let _ = wav_from_bytes(data);
});
