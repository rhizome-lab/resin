#![no_main]

use libfuzzer_sys::fuzz_target;
use rhizome_resin_mesh::import_obj;

fuzz_target!(|data: &str| {
    // import_obj should never panic on any input
    let _ = import_obj(data);
});
