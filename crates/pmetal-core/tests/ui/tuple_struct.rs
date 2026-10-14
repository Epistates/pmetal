use pmetal_core_derive::JobSpec;

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "t")]
struct S(String);

fn main() {}
