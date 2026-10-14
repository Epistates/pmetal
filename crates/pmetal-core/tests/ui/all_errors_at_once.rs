use pmetal_core_derive::JobSpec;

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "t")]
struct S {
    #[job(label = "A", group = "G", min = 1)]
    a: String,
    #[job(label = "B", group = "G", invert)]
    b: bool,
    #[job(group = "G")]
    c: String,
}

fn main() {}
