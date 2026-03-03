//! Display PMetal version and device information.
//!
//! ```sh
//! cargo run -p pmetal --example device_info
//! ```

fn main() {
    println!("PMetal v{}", pmetal::version::VERSION);
    println!();

    let info = pmetal::version::device_info();
    print!("{info}");
}
