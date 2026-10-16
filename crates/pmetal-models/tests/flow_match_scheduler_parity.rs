//! The flow-matching Euler scheduler against `diffusers`'
//! `FlowMatchEulerDiscreteScheduler`.
//!
//! Six schedules: FLUX.1-dev at two image sizes (exponential dynamic shift),
//! FLUX.1-schnell (none), SD3's static shift on the scheduler's own default
//! sigmas, a static shift stretched to a terminal sigma, and the linear
//! dynamic shift. For each: the sigmas (with the terminal zero) and
//! timesteps, one Euler step, and `scale_noise`.

use pmetal_models::sampling::diffusion::{
    FlowMatchConfig, FlowMatchEulerDiscreteScheduler, calculate_shift, flux_sigmas,
};
use pmetal_runtime::compat::Array;
use serde_json::Value;

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|v| v.as_f64().expect("a number") as f32)
        .collect()
}

fn array(value: &Value, shape: &[i32]) -> Array {
    Array::from_slice(&floats(value), shape)
}

fn to_vec(a: &Array) -> Vec<f32> {
    pmetal_cache::test_utils::to_f32_vec_eval(a)
}

fn close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance,
            "{name}[{i}]: {g} against {w} (tolerance {tolerance})"
        );
    }
}

#[test]
fn schedules_and_steps_match_diffusers() {
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/flow_match_scheduler_reference.json"))
            .expect("fixture parses");
    let shape = [2, 6];
    let sample = array(&reference["sample"], &shape);
    let model_output = array(&reference["model_output"], &shape);
    let noise = array(&reference["noise"], &shape);

    for case in reference["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let config: FlowMatchConfig =
            serde_json::from_value(case["config"].clone()).expect("config parses");
        let steps = case["steps"].as_u64().expect("steps") as usize;
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(config.clone()).expect("builds");

        // mu comes from the token count, the way the pipeline computes it.
        let mu = case["seq_len"]
            .as_u64()
            .filter(|_| config.use_dynamic_shifting)
            .map(|tokens| {
                calculate_shift(
                    tokens as usize,
                    config.base_image_seq_len,
                    config.max_image_seq_len,
                    config.base_shift.unwrap_or(0.5),
                    config.max_shift.unwrap_or(1.15),
                )
            });
        if let Some(want) = case["mu"].as_f64() {
            assert!((mu.expect("mu") - want).abs() < 1e-12, "{name}: mu");
        }
        let sigmas_in = case["sigmas_in"]
            .as_array()
            .map(|_| floats(&case["sigmas_in"]));
        if let Some(given) = &sigmas_in {
            // The pipeline's own recipe for the unshifted levels.
            close(
                &format!("{name} flux_sigmas"),
                &flux_sigmas(steps),
                given,
                1e-7,
            );
        }
        scheduler
            .set_timesteps(steps, sigmas_in.as_deref(), mu)
            .expect("set_timesteps");

        // float32 arithmetic in a different order: a few ulps of a value <= 1000.
        close(
            &format!("{name} sigmas"),
            scheduler.sigmas(),
            &floats(&case["sigmas"]),
            2e-6,
        );
        close(
            &format!("{name} timesteps"),
            scheduler.timesteps(),
            &floats(&case["timesteps"]),
            1e-3,
        );

        let index = case["step_index"].as_u64().expect("step_index") as usize;
        let stepped = scheduler.step(&model_output, index, &sample).expect("step");
        close(
            &format!("{name} step"),
            &to_vec(&stepped),
            &floats(&case["step_prev"]),
            2e-6,
        );
        let scaled = scheduler
            .scale_noise(&sample, &noise, index)
            .expect("scale_noise");
        close(
            &format!("{name} scale_noise"),
            &to_vec(&scaled),
            &floats(&case["scaled"]),
            2e-6,
        );
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    }
}

/// The schedules differ: the test above can only fail if they do.
#[test]
fn the_cases_exercise_different_shifts() {
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/flow_match_scheduler_reference.json"))
            .expect("fixture parses");
    let cases = reference["cases"].as_array().expect("cases");
    let second_sigma = |name: &str| {
        let case = cases.iter().find(|c| c["name"] == name).expect("case");
        floats(&case["sigmas"])[1]
    };
    // Same number of steps would make these equal if the shift were ignored.
    assert!(second_sigma("dev_4096_tokens") > second_sigma("schnell"));
    assert!((second_sigma("dev_4096_tokens") - second_sigma("dev_1024_tokens")).abs() > 1e-3);
}
