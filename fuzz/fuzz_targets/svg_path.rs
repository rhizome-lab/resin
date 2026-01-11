#![no_main]

use libfuzzer_sys::fuzz_target;
use rhizome_resin_vector::svg::parse_path_data;

fuzz_target!(|data: &str| {
    // parse_path_data should never panic on any input
    let _ = parse_path_data(data);
});
