use pmetal_trainer::pretrain::create_model;

/// `phi` names Phi-1 / Phi-1.5 / Phi-2, whose block is not the Phi-3 one the
/// factory builds, so it is refused rather than built under that name.
#[test]
fn pretrain_factory_refuses_phi_2_by_name() {
    let err = create_model("phi", None)
        .err()
        .expect("phi is declined")
        .to_string();
    assert!(err.contains("Phi-2"), "{err}");
    assert!(err.contains("phi3"), "{err}");
}

#[test]
fn pretrain_factory_builds_phi3() {
    assert!(create_model("phi3", None).is_ok());
}
