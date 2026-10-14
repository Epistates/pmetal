use pmetal_core_derive::JobSpec;

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "t")]
struct S {
    #[job(label = "L", group = "G", lable = "typo")]
    f: String,
}

fn main() {}
