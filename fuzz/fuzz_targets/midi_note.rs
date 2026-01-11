#![no_main]

use libfuzzer_sys::fuzz_target;
use rhizome_resin_audio::parse_note_name;

fuzz_target!(|data: &str| {
    // parse_note_name should never panic on any input
    let _ = parse_note_name(data);
});
