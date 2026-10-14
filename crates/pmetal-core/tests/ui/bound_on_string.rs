use pmetal_core_derive::JobSpec;

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "t")]
struct S {
    #[job(label = "L", group = "G", min = 1)]
    f: String,
}

fn main() {}
