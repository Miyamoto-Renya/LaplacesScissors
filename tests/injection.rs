//! 수치 주입·공유 decoder·원본 가중치와 분리된 개입·저장 수정안 재실행.
use anyhow::{Context, Result};
use laplace::{
    gpu::Gpu,
    hub,
    injection::{self, Options},
    model::{DecoderTrace, Model, WeightDelta},
};
use safetensors::{Dtype, serialize_to_file, tensor::TensorView};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

fn checkpoint(root: &Path) -> Result<()> {
    std::fs::create_dir_all(root)?;
    let mut data = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    let mut insert = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
        data.insert(
            name.into(),
            (
                shape,
                values.into_iter().flat_map(f32::to_le_bytes).collect(),
            ),
        );
    };
    let wave = |rows: usize, cols: usize, scale: f64, phase: f64| {
        (0..rows * cols)
            .map(|i| {
                let a = (i / cols + 1) as f64;
                let b = (i % cols + 2) as f64;
                (scale * ((a * b * 0.37 + phase).sin() + 0.2 * ((a + 2.) * (b + 1.) * 0.71).cos()))
                    as f32
            })
            .collect::<Vec<_>>()
    };
    insert("embedding", vec![16, 8], wave(16, 8, 0.8, 0.));
    insert("readout", vec![16, 8], wave(16, 8, 0.4, 0.7));
    insert("gain", vec![8], vec![1.; 8]);
    insert(
        "final_gain",
        vec![8],
        (0..8).map(|i| 0.8 + i as f32 * 0.05).collect(),
    );
    for (name, rows, cols, phase) in [
        ("query", 8, 8, 0.2),
        ("key", 4, 8, 0.5),
        ("value", 4, 8, 0.9),
        ("output", 8, 8, 1.4),
        ("gate", 12, 8, 0.1),
        ("up", 12, 8, 0.8),
        ("down", 8, 12, 1.9),
    ] {
        insert(name, vec![rows, cols], wave(rows, cols, 0.12, phase));
    }
    let views = data
        .iter()
        .map(|(name, (shape, bytes))| {
            Ok((
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, None, &root.join("model.safetensors"))?;
    let norm = json!({"kind":"rms","weight":"gain","epsilon":1e-5});
    let matrix = |name| json!({"weight":name});
    let graph = json!({"version":1,"source":"numeric_regression","hidden_size":8,"vocab_size":16,"max_context":8,
        "embedding":"embedding","embedding_scale":1.7,
        "final_norm":{"kind":"layer","weight":"final_gain","epsilon":1e-5,"gain_offset":0.1},
        "readout":matrix("readout"),"blocks":[{"input_norm":norm,"post_attention_norm":norm,
            "attention":{"query":matrix("query"),"key":matrix("key"),"value":matrix("value"),"output":matrix("output"),
                "query_heads":2,"kv_heads":1,"head_dim":4,"rotary_dim":4,"rope_theta":10000.,"score_scale":0.5},
            "mlp":{"gate":matrix("gate"),"up":matrix("up"),"down":matrix("down"),"intermediate_size":12,"activation":"silu"}}]});
    std::fs::write(root.join("laplace.graph.json"), serde_json::to_vec(&graph)?)?;
    Ok(())
}
fn options(output: PathBuf) -> Options {
    Options {
        directions: 3,
        prefix_length: 1,
        seed: 7,
        probe_angle: 0.05,
        max_angle_degrees: 3.,
        max_edit_norm: Some(1.),
        tolerance: 2e-5,
        coordinates: 4,
        intervention_fraction: 0.1,
        time_budget: Duration::from_secs(90),
        inputs: None,
        proposal: None,
        output,
    }
}
#[test]
#[ignore = "LAPLACE_TEST_CACHE와 LAPLACE_PYTHON으로 수치 주입 전체 경로를 실행한다"]
fn numeric_pipeline_and_exact_decoder_entry() -> Result<()> {
    let cache =
        PathBuf::from(std::env::var_os("LAPLACE_TEST_CACHE").context("LAPLACE_TEST_CACHE")?);
    let python = PathBuf::from(std::env::var_os("LAPLACE_PYTHON").context("LAPLACE_PYTHON")?);
    let work = cache.join(format!(
        "numeric-{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        std::process::id()
    ));
    let root = work.join("model");
    checkpoint(&root)?;
    let files = hub::resolve_weights(root.to_str().unwrap(), None, &cache)?;
    assert!(files.files.iter().all(|f| !f.name.contains("tokenizer")));
    let original_files = serde_json::to_value(&files.files)?;
    let mut gpu = Gpu::new(&cache, &python)?;
    let mut model = Model::load(&root, &mut gpu, 4)?;
    let prefix = model.embedding_vector(&mut gpu, 2)?;
    let vector = model.embedding_vector(&mut gpu, 7)?;
    model.advance(&mut gpu, 2, None)?;
    let tokens = model.advance(&mut gpu, 7, None)?;
    model.reset();
    model.inject(&mut gpu, &prefix, None, None, None)?;
    let mut trace = DecoderTrace::default();
    let injected = model.inject(&mut gpu, &vector, None, None, Some(&mut trace))?;
    assert_eq!(tokens.hidden, injected.hidden);
    assert_eq!(tokens.logits, injected.logits);
    assert_eq!(trace.branches.len(), 2);
    let edit = model.prepare_intervention(
        "output",
        vec![WeightDelta {
            row: 2,
            column: 3,
            delta: 0.2,
        }],
    )?;
    model.reset();
    model.inject(&mut gpu, &prefix, None, Some(&edit), None)?;
    let altered = model.inject(&mut gpu, &vector, None, Some(&edit), None)?;
    assert_ne!(altered.hidden, injected.hidden);
    model.reset();
    model.inject(&mut gpu, &prefix, None, None, None)?;
    let restored = model.inject(&mut gpu, &vector, None, None, None)?;
    assert_eq!(restored.logits, injected.logits);
    assert!(
        model
            .prepare_intervention(
                "output",
                vec![WeightDelta {
                    row: 8,
                    column: 0,
                    delta: 1.
                }]
            )
            .is_err()
    );
    drop(model);
    drop(gpu);

    let report = injection::run(files.clone(), &cache, &python, options(work.join("scan")))?;
    assert_eq!(report.phase, "complete");
    assert_eq!(report.scan.len(), 3);
    assert_eq!(report.computation.candidate_decoder_steps, 0);
    assert!(report.transport.as_ref().unwrap().absolute_residual < 1e-6);
    assert!(report.baseline.as_ref().unwrap().secant_absolute_residual < 1e-12);
    assert!(
        report
            .candidates
            .iter()
            .all(|c| c.weight_delta_norm_squared <= 1. + 1e-10)
    );
    let proposal = report.selected.context("구별도 수정안")?;
    assert_eq!(proposal.version, 3);
    assert!(proposal.target_margin.is_none());
    let mut replay = options(work.join("replay"));
    replay.inputs = Some(work.join("scan/inputs.json"));
    replay.proposal = Some(work.join("scan/proposal.json"));
    let repeated = injection::run(files.clone(), &cache, &python, replay)?;
    assert_eq!(repeated.outcome, "replayed_separation_rotation");
    let measured = repeated.replay.unwrap();
    assert!(
        (measured["after"]["jeffreys"].as_f64().unwrap() - proposal.separation.unwrap().after)
            .abs()
            < 1e-7
    );
    assert!(
        measured["protection"]["maximum_scaled_change"]
            .as_f64()
            .unwrap()
            <= 2e-5
    );
    let after = hub::resolve_weights(root.to_str().unwrap(), None, &cache)?;
    assert_eq!(original_files, serde_json::to_value(after.files)?);
    println!(
        "numeric input, layer-normalized transport, finite intervention, protected replay: {}",
        work.display()
    );
    Ok(())
}
