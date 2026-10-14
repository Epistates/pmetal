use pmetal_core_derive::JobSpec;

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "t")]
struct S {
    #[job(label = "L", group = "G", min = 5, max = 1)]
    f: u32,
}

fn main() {}
