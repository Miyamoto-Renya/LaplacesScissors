//! 전체 모델에서 복구·보호·EOS·공유 embedding과 readout 분리를 검증한다.
use anyhow::{Context, Result, ensure};
use laplace::{
    diagnosis::{self, Options},
    gpu::Gpu,
    hub,
    model::{DeviceRotation, Model},
    probes::Case,
    rotation::ReadoutRotation,
    text::Text,
};
use safetensors::{Dtype, serialize_to_file, tensor::TensorView};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
fn fixture(root: &Path, tied: bool) -> Result<()> {
    fixture_with_bias(root, tied, -0.2)
}
fn fixture_with_bias(root: &Path, tied: bool, head_bias: f32) -> Result<()> {
    std::fs::create_dir_all(root)?;
    let mut tensors = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    let mut put = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
        tensors.insert(
            name.into(),
            (shape, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        );
    };
    put(
        "embed",
        vec![6, 4],
        vec![
            0., 0., -1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
            0., 0., 1.,
        ],
    );
    put(
        "head",
        vec![6, 4],
        vec![
            0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0.,
        ],
    );
    put("bias", vec![6], vec![head_bias, 0., -10., -10., -10., -10.]);
    put("norm", vec![4], vec![(0.25f32 + 1e-5).sqrt(); 4]);
    for (name, rows, cols) in [
        ("q", 4, 4),
        ("k", 2, 4),
        ("v", 2, 4),
        ("o", 4, 4),
        ("gate", 3, 4),
        ("up", 3, 4),
        ("down", 4, 3),
    ] {
        put(name, vec![rows, cols], vec![0.; rows * cols]);
    }
    let views = tensors
        .iter()
        .map(|(name, (shape, bytes))| {
            Ok((
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, None, &root.join("model.safetensors"))?;
    let norm = json!({"weight":"norm","kind":"rms","epsilon":1e-5});
    let linear = |name: &str| json!({"weight":name});
    let graph = json!({"version":1,"source":"regression_fixture","hidden_size":4,"vocab_size":6,"max_context":16,"embedding":"embed","final_norm":norm,"readout":{"weight":if tied{"embed"}else{"head"},"bias":"bias"},"blocks":[{"input_norm":norm,"post_attention_norm":norm,"attention":{"query":linear("q"),"key":linear("k"),"value":linear("v"),"output":linear("o"),"query_heads":2,"kv_heads":1,"head_dim":2,"rotary_dim":2,"rope_theta":10000.,"score_scale":0.70710677},"mlp":{"gate":linear("gate"),"up":linear("up"),"down":linear("down"),"intermediate_size":3,"activation":"silu"}}],"eos_token_ids":[1]});
    std::fs::write(root.join("laplace.graph.json"), serde_json::to_vec(&graph)?)?;
    std::fs::write(root.join("config.json"), br#"{"eos_token_id":1}"#)?;
    std::fs::write(
        root.join("tokenizer_config.json"),
        br#"{"chat_template":"{{ messages[0].content }}","eos_token":"</s>"}"#,
    )?;
    let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":1,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"A":0,"</s>":1,"bad":2,"good":3,"[UNK]":4,"pad":5},"unk_token":"[UNK]"}});
    std::fs::write(root.join("tokenizer.json"), serde_json::to_vec(&tokenizer)?)?;
    Ok(())
}

#[test]
#[ignore = "세 합성 checkpoint에서 수정과 분리된 평가의 규모 기여를 실행한다"]
fn scale_pipeline_preserves_signed_attribution() -> Result<()> {
    let cache =
        PathBuf::from(std::env::var_os("LAPLACE_TEST_CACHE").context("LAPLACE_TEST_CACHE")?);
    let python = PathBuf::from(std::env::var_os("LAPLACE_PYTHON").context("LAPLACE_PYTHON")?);
    let work = cache.join(format!(
        "scale-{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        std::process::id()
    ));
    let mut checkpoints = Vec::new();
    for (index, bias) in [-0.1, -0.2, -0.35].into_iter().enumerate() {
        let root = work.join(format!("model-{index}"));
        fixture_with_bias(&root, false, bias)?;
        checkpoints.push(json!({"model":root,"resource":2f64.powi(index as i32)}));
    }
    let case = |id: &str, prompt: &str| Case {
        id: id.into(),
        family: "synthetic_scale".into(),
        prompt: prompt.into(),
        choices: vec!["A".into(), "other".into()],
        expected: 0,
        oracle: json!({"answer":"A"}),
    };
    let repair = [case("preserve", "good"), case("repair", "bad")];
    let evaluation = case("evaluation", "unseen");
    std::fs::write(
        work.join("repair.jsonl"),
        repair
            .iter()
            .map(|c| serde_json::to_string(c).unwrap() + "\n")
            .collect::<String>(),
    )?;
    std::fs::write(
        work.join("evaluation.jsonl"),
        serde_json::to_string(&evaluation)? + "\n",
    )?;
    let manifest = work.join("series.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(
            &json!({"axis":"synthetic_resource","conditions":"fixed decoder and embeddings; prescribed readout bias",
        "reference":"fixed synthetic exponent","exponent":0.5,"max_edit_norm":10.,
        "cases":"repair.jsonl","evaluation":"evaluation.jsonl","checkpoints":checkpoints}),
        )?,
    )?;
    let report = laplace::scaling::run(
        &manifest,
        &cache,
        &python,
        Options {
            cases: Vec::new(),
            context: 16,
            max_new_tokens: 3,
            max_planes: 4,
            max_angle_degrees: 120.,
            margin: 0.1,
            tolerance: 2e-5,
            time_budget: Duration::from_secs(60),
            output: work.join("run"),
        },
        diagnosis::Policy::default(),
    )?;
    assert_eq!(report.phase, "complete");
    assert_eq!(report.checkpoints.len(), 3);
    assert!(report.checkpoints.iter().all(|c| c.selected.is_some()));
    assert_eq!(report.contrasts.len(), 1);
    assert!(report.contrasts[0].attribution_residual < 1e-5);
    assert!(
        report
            .checkpoints
            .iter()
            .all(|c| c.evaluation.loss_after < c.evaluation.loss_before)
    );
    println!(
        "scale input, separate evaluation, finite edits and signed contribution sum: {}",
        work.display()
    );
    Ok(())
}
#[test]
#[ignore = "실제 GPU에서 수정안 계산과 생성 검증을 끝까지 실행한다"]
fn full_repair_and_runtime_readout_separation() -> Result<()> {
    let cache =
        PathBuf::from(std::env::var_os("LAPLACE_TEST_CACHE").context("LAPLACE_TEST_CACHE")?);
    let python = PathBuf::from(std::env::var_os("LAPLACE_PYTHON").context("LAPLACE_PYTHON")?);
    let work = cache.join(format!(
        "fixture-{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        std::process::id()
    ));
    let root = work.join("model");
    fixture(&root, false)?;
    let files = hub::resolve(root.to_str().unwrap(), None, &cache)?;
    let before = files.files.clone();
    let text = Text::load(&root, false)?;
    let encoded = text.target("bad", "A")?;
    assert_eq!(encoded.prompt, vec![2]);
    assert_eq!(encoded.continuation, vec![0, 1]);
    let cases = ["good", "bad"]
        .into_iter()
        .map(|p| Case {
            id: p.into(),
            family: "algebraic_fixture".into(),
            prompt: p.into(),
            choices: vec!["A".into(), "other".into()],
            expected: 0,
            oracle: json!({"expected":"A"}),
        })
        .collect();
    let report = diagnosis::diagnose(
        files.clone(),
        &cache,
        &python,
        Options {
            cases,
            context: 16,
            max_new_tokens: 3,
            max_planes: 4,
            max_angle_degrees: 120.,
            margin: 0.1,
            tolerance: 2e-5,
            time_budget: Duration::from_secs(120),
            output: work.join("run"),
        },
    )?;
    ensure!(
        report.outcome == "verified_rotation",
        "검증 결과 {}",
        report.outcome
    );
    assert!(report.baseline[0].correct);
    assert!(!report.baseline[1].correct);
    assert_eq!(report.protected_rank, 1);
    assert!(report.parameters["weight_storage"]["F32"].as_u64().unwrap() > 0);
    let proposal = report.selected.context("채택한 회전 수정안")?;
    let verified = report
        .candidates
        .iter()
        .filter_map(|c| c.verification.as_ref())
        .find(|v| v.accepted)
        .unwrap();
    assert!(verified.target_min_margin >= 0.1);
    assert!(verified.protected_max_scaled_change <= 2e-5);
    assert_eq!(report.computation.verification_decoder_steps, 0);
    assert_eq!(verified.decoder_steps, 0);
    assert_eq!(
        verified.readout_calls,
        report.protected_prefixes + report.target_prefixes
    );
    assert!(
        verified
            .generations
            .iter()
            .all(|g| g.correct && g.generation.tokens == vec![0, 1])
    );
    let mut gpu = Gpu::new(&cache, &python)?;
    let mut model = Model::load(&root, &mut gpu, 16)?;
    proposal.validate(&files, &model)?;
    let mut changed = files.clone();
    changed
        .files
        .iter_mut()
        .find(|f| f.name.ends_with(".safetensors"))
        .unwrap()
        .sha256 = "different".into();
    assert!(proposal.validate(&changed, &model).is_err());
    let edit = DeviceRotation::upload_edit(&gpu, &proposal.rotation, 4)?;
    for (case, cached) in report.cases.iter().zip(&verified.generations) {
        let prompt = text.encode(&text.render(&case.prompt)?)?;
        let replay = diagnosis::generate(
            &mut model,
            &mut gpu,
            &text,
            &prompt,
            3,
            Some(&edit),
            &diagnosis::Deadline::after(Duration::from_secs(30))?,
        )?;
        assert_eq!(replay.tokens, cached.generation.tokens);
        assert_eq!(replay.stopped, cached.generation.stopped);
    }
    model.reset();
    let original = model.advance(&mut gpu, 2, None)?;
    model.reset();
    let repaired = model.advance(&mut gpu, 2, Some(&edit))?;
    assert_eq!(original.hidden, repaired.hidden);
    assert_ne!(original.logits, repaired.logits);
    let branch = model.advance(&mut gpu, 0, Some(&edit))?;
    model.rewind(1)?;
    let replay = model.advance(&mut gpu, 0, Some(&edit))?;
    assert_eq!(branch.hidden, replay.hidden);
    assert_eq!(branch.logits, replay.logits);
    assert!(model.rewind(9).is_err());
    assert!(model.step(&mut gpu, 6, None).is_err());
    let after = hub::resolve(root.to_str().unwrap(), None, &cache)?;
    assert_eq!(
        serde_json::to_value(before)?,
        serde_json::to_value(after.files)?
    );
    drop(model);
    drop(gpu);
    let tied_root = work.join("tied");
    fixture(&tied_root, true)?;
    let mut gpu = Gpu::new(&cache, &python)?;
    let mut model = Model::load(&tied_root, &mut gpu, 16)?;
    let specification = ReadoutRotation {
        u: vec![1., 0., 0., 0.],
        v: vec![0., 1., 0., 0.],
        tangent: 1. / 3.,
    };
    let rotation = DeviceRotation::upload(&gpu, &specification, 4)?;
    let original = model.advance(&mut gpu, 2, None)?;
    model.reset();
    let rotated = model.advance(&mut gpu, 2, Some(&rotation))?;
    assert_eq!(original.hidden, rotated.hidden);
    assert_ne!(original.logits, rotated.logits);
    let images = model.head_image(&mut gpu, &[0., 0., 0., 0.])?;
    assert!(images.iter().all(|v| *v == 0.));
    println!(
        "verified repair, EOS, protected prefixes, checkpoint hashes, KV rewind, tied readout; {}",
        work.display()
    );
    Ok(())
}
