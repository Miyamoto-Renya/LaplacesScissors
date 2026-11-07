//! 공식 Qwen3 저장 가중치에 대한 독립 Rust 전체 문맥 기준 계산.
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use laplace::{gpu::Gpu, model::Model, text::Text};
use memmap2::Mmap;
use nalgebra::DMatrix;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    path::PathBuf,
    time::Instant,
};

fn floats(source: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let tensor = source.tensor(name)?;
    Ok(match tensor.dtype() {
        Dtype::BF16 => tensor
            .data()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        Dtype::F16 => tensor
            .data()
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        Dtype::F32 => tensor
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        other => anyhow::bail!("CPU 기준 입력 dtype: {other:?}"),
    })
}

// 각 열이 한 token인 전체 문맥 행렬을 사용한다.
fn projection(source: &SafeTensors<'_>, name: &str, x: &DMatrix<f32>) -> Result<DMatrix<f32>> {
    let tensor = source.tensor(name)?;
    let shape = tensor.shape();
    let weights = DMatrix::from_row_slice(shape[0], shape[1], &floats(source, name)?);
    Ok(weights * x)
}

fn rms(x: &DMatrix<f32>, gain: &[f32], epsilon: f64) -> DMatrix<f32> {
    let mut y = x.clone();
    let width = gain.len();
    for group in y.as_mut_slice().chunks_exact_mut(width) {
        let inverse = (group.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / width as f64
            + epsilon)
            .sqrt()
            .recip();
        for (v, &g) in group.iter_mut().zip(gain) {
            *v = (f64::from(*v) * inverse * f64::from(g)) as f32;
        }
    }
    y
}

fn rope(x: &mut DMatrix<f32>, heads: usize, dimension: usize, theta: f64) {
    for token in 0..x.ncols() {
        for head in 0..heads {
            for pair in 0..dimension / 2 {
                let phase = token as f64 / theta.powf(2. * pair as f64 / dimension as f64);
                let (sine, cosine) = phase.sin_cos();
                let first = head * dimension + pair;
                let second = first + dimension / 2;
                let a = f64::from(x[(first, token)]);
                let b = f64::from(x[(second, token)]);
                x[(first, token)] = (a * cosine - b * sine) as f32;
                x[(second, token)] = (b * cosine + a * sine) as f32;
            }
        }
    }
}

fn causal_attention(
    q: &DMatrix<f32>,
    k: &DMatrix<f32>,
    v: &DMatrix<f32>,
    qheads: usize,
    kvheads: usize,
    dim: usize,
) -> DMatrix<f32> {
    let mut output = DMatrix::zeros(q.nrows(), q.ncols());
    for token in 0..q.ncols() {
        for head in 0..qheads {
            let kvhead = head / (qheads / kvheads);
            let mut scores = (0..=token)
                .map(|time| {
                    (0..dim)
                        .map(|d| {
                            f64::from(q[(head * dim + d, token)])
                                * f64::from(k[(kvhead * dim + d, time)])
                        })
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - maximum).exp();
            }
            let denominator = scores.iter().sum::<f64>();
            for d in 0..dim {
                output[(head * dim + d, token)] = (scores
                    .iter()
                    .enumerate()
                    .map(|(time, score)| score * f64::from(v[(kvhead * dim + d, time)]))
                    .sum::<f64>()
                    / denominator) as f32;
            }
        }
    }
    output
}

fn cpu_forward(source: &SafeTensors<'_>, config: &Value, tokens: &[u32]) -> Result<Vec<f32>> {
    let integer = |name: &str| {
        config[name]
            .as_u64()
            .with_context(|| name.to_owned())
            .map(|v| v as usize)
    };
    let width = integer("hidden_size")?;
    let layers = integer("num_hidden_layers")?;
    let heads = integer("num_attention_heads")?;
    let kvheads = integer("num_key_value_heads")?;
    let dimension = integer("head_dim")?;
    let epsilon = config["rms_norm_eps"].as_f64().context("rms_norm_eps")?;
    let theta = config["rope_theta"].as_f64().context("rope_theta")?;
    let embedding = floats(source, "model.embed_tokens.weight")?;
    let mut x = DMatrix::zeros(width, tokens.len());
    for (column, &token) in tokens.iter().enumerate() {
        x.column_mut(column)
            .copy_from_slice(&embedding[token as usize * width..(token as usize + 1) * width]);
    }
    drop(embedding);
    for layer in 0..layers {
        let prefix = format!("model.layers.{layer}");
        let normalized = rms(
            &x,
            &floats(source, &format!("{prefix}.input_layernorm.weight"))?,
            epsilon,
        );
        let query = projection(
            source,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &normalized,
        )?;
        let key = projection(
            source,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &normalized,
        )?;
        let value = projection(
            source,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &normalized,
        )?;
        let mut query = rms(
            &query,
            &floats(source, &format!("{prefix}.self_attn.q_norm.weight"))?,
            epsilon,
        );
        let mut key = rms(
            &key,
            &floats(source, &format!("{prefix}.self_attn.k_norm.weight"))?,
            epsilon,
        );
        rope(&mut query, heads, dimension, theta);
        rope(&mut key, kvheads, dimension, theta);
        let attended = causal_attention(&query, &key, &value, heads, kvheads, dimension);
        x += projection(
            source,
            &format!("{prefix}.self_attn.o_proj.weight"),
            &attended,
        )?;
        let normalized = rms(
            &x,
            &floats(source, &format!("{prefix}.post_attention_layernorm.weight"))?,
            epsilon,
        );
        let mut gate = projection(
            source,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &normalized,
        )?;
        let up = projection(source, &format!("{prefix}.mlp.up_proj.weight"), &normalized)?;
        for (gate, &up) in gate.iter_mut().zip(up.iter()) {
            *gate = (f64::from(*gate) / (1. + (-f64::from(*gate)).exp()) * f64::from(up)) as f32;
        }
        x += projection(source, &format!("{prefix}.mlp.down_proj.weight"), &gate)?;
    }
    let last = x.column(tokens.len() - 1).into_owned();
    let last = DMatrix::from_column_slice(width, 1, last.as_slice());
    let normalized = rms(&last, &floats(source, "model.norm.weight")?, epsilon);
    Ok(projection(source, "lm_head.weight", &normalized)?
        .as_slice()
        .to_vec())
}

fn top(logits: &[f32], text: &Text, count: usize) -> Result<Vec<Value>> {
    let mut ids = (0..logits.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    ids[..count]
        .iter()
        .map(|&id| Ok(json!({"token":id,"text":text.decode(&[id as u32])?,"logit":logits[id]})))
        .collect()
}

#[test]
#[ignore = "공식 Qwen3 전체 모델의 Rust CPU 문맥 계산과 Triton KV 실행을 비교한다"]
fn original_qwen_cpu_full_context_matches_gpu_logits() -> Result<()> {
    let started = Instant::now();
    let root = PathBuf::from(std::env::var_os("LAPLACE_TEST_MODEL").context("LAPLACE_TEST_MODEL")?);
    let cache =
        PathBuf::from(std::env::var_os("LAPLACE_TEST_CACHE").context("LAPLACE_TEST_CACHE")?);
    let python = PathBuf::from(std::env::var_os("LAPLACE_PYTHON").context("LAPLACE_PYTHON")?);
    let config: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
    ensure!(
        config["model_type"] == "qwen3",
        "이 기준 계산은 Qwen3 연산 식을 사용한다"
    );
    let text = Text::load(&root, false)?;
    let prompt = text.render("What is 2 + 2? Answer with just the number.")?;
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    let tokens = text.encode(&prompt)?;
    let map = unsafe { Mmap::map(&File::open(root.join("model.safetensors"))?)? };
    let checkpoint_sha = hex::encode(Sha256::digest(&map));
    let source = SafeTensors::deserialize(&map)?;
    let cpu_started = Instant::now();
    let cpu = cpu_forward(&source, &config, &tokens)?;
    let cpu_seconds = cpu_started.elapsed().as_secs_f64();
    let mut gpu = Gpu::new(&cache, &python)?;
    let mut model = Model::load(&root, &mut gpu, tokens.len())?;
    let mut actual = Vec::new();
    for &token in &tokens {
        actual = model.step(&mut gpu, token, None)?;
    }
    let error = cpu
        .iter()
        .zip(&actual)
        .map(|(&c, &g)| (f64::from(c) - f64::from(g)).powi(2))
        .sum::<f64>();
    let norm = cpu.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    let relative = (error / norm).sqrt();
    let maximum = cpu
        .iter()
        .zip(&actual)
        .map(|(&c, &g)| (f64::from(c) - f64::from(g)).abs())
        .fold(0., f64::max);
    let evidence = json!({"model":root,"source_sha256":checkpoint_sha,"device":gpu.name(),
        "prompt":prompt,"input_token_ids":tokens,"weight_precision":"original checkpoint BF16",
        "gpu_activation_precision":"F32","cpu_matrix_precision":"F32","cpu_reduction_precision":"F64",
        "cpu_execution":"Rust nalgebra matrix products, full-context causal attention",
        "relative_l2":relative,"maximum_absolute":maximum,"cpu_top":top(&cpu,&text,5)?,
        "gpu_top":top(&actual,&text,5)?,"cpu_seconds":cpu_seconds,"seconds":started.elapsed().as_secs_f64()});
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    fs::write(
        cache.join("qwen-independent-forward.json"),
        serde_json::to_vec_pretty(&evidence)?,
    )?;
    // F32 순서 차이를 포함한 전체 네트워크 대조의 상대 오차 기준.
    assert!(
        relative < f64::from(f32::EPSILON.sqrt()),
        "CPU/GPU logit relative L2 {relative}"
    );
    assert_eq!(
        top(&cpu, &text, 1)?[0]["token"],
        top(&actual, &text, 1)?[0]["token"]
    );
    Ok(())
}
