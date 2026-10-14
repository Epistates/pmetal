use pmetal_core_derive::JobSpec;

#[derive(JobSpec)]
#[spec(kind = "Train Job", subcommand = "t")]
struct S {
    #[job(label = "L", group = "G")]
    f: String,
}

fn main() {}
