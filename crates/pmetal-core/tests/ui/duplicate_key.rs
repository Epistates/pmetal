use pmetal_core_derive::JobSpec;

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "t")]
struct S {
    #[job(label = "A", label = "B", group = "G")]
    f: String,
}

fn main() {}
