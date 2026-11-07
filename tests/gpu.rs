//! 독립 CPU 수치식으로 GPU 연산과 출력 회전을 검증한다.
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use laplace::{
    gpu::{AttentionConfig, DType, Gpu, NormConfig, RopeConfig},
    rotation::ReadoutRotation,
};
use std::path::PathBuf;
fn close(actual: &[f32], expected: &[f64], tolerance: f64) -> Result<()> {
    ensure!(actual.len() == expected.len(), "결과 길이");
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            a.is_finite() && (f64::from(*a) - b).abs() <= tolerance * (1. + b.abs()),
            "원소 {i}: GPU {a}, CPU {b}"
        );
    }
    Ok(())
}
#[test]
#[ignore = "LAPLACE_TEST_CACHE와 LAPLACE_PYTHON으로 실제 GPU를 실행한다"]
fn decoder_operations_and_rotation() -> Result<()> {
    let cache =
        PathBuf::from(std::env::var_os("LAPLACE_TEST_CACHE").context("LAPLACE_TEST_CACHE")?);
    let python = PathBuf::from(std::env::var_os("LAPLACE_PYTHON").context("LAPLACE_PYTHON")?);
    let mut gpu = Gpu::new(&cache, &python)?;
    let x = [0.25f32, -2., 1.5, 0.75, -0.3];
    let w = (0..15).map(|i| (i as f32 - 7.) / 9.).collect::<Vec<_>>();
    let bias = [0.3, -0.4, 0.5];
    let input = gpu.upload_f32(&x)?;
    let bias_gpu = gpu.upload_f32(&bias)?;
    for kind in [DType::F32, DType::F16, DType::BF16] {
        let rounded = w
            .iter()
            .map(|v| match kind {
                DType::F32 => *v,
                DType::F16 => f16::from_f32(*v).to_f32(),
                DType::BF16 => bf16::from_f32(*v).to_f32(),
            })
            .collect::<Vec<_>>();
        let bytes = w
            .iter()
            .flat_map(|v| match kind {
                DType::F32 => v.to_le_bytes().to_vec(),
                DType::F16 => f16::from_f32(*v).to_bits().to_le_bytes().to_vec(),
                DType::BF16 => bf16::from_f32(*v).to_bits().to_le_bytes().to_vec(),
            })
            .collect::<Vec<_>>();
        let weight = gpu.upload_raw(&bytes, kind)?;
        let output = gpu.linear(&input, &weight, Some(&bias_gpu), 3, 5)?;
        let expected = rounded
            .chunks_exact(5)
            .zip(bias)
            .map(|(row, b)| {
                row.iter()
                    .zip(x)
                    .map(|(a, x)| f64::from(*a) * f64::from(x))
                    .sum::<f64>()
                    + f64::from(b)
            })
            .collect::<Vec<_>>();
        close(&gpu.read(&output)?, &expected, 2e-6)?;
        let row = gpu.embedding(&weight, 2, 5)?;
        close(
            &gpu.read(&row)?,
            &rounded[10..]
                .iter()
                .map(|v| f64::from(*v))
                .collect::<Vec<_>>(),
            0.,
        )?;
    }
    let gain_values = [1., 0.5, 1.2, 0.8, 1.5];
    let gain = gpu.upload_f32(&gain_values)?;
    for center in [false, true] {
        let normalized = gpu.norm(
            &input,
            &gain,
            Some(&input),
            NormConfig {
                groups: 1,
                width: 5,
                epsilon: 1e-5,
                center,
            },
        )?;
        let mean = if center {
            x.iter().map(|x| f64::from(*x)).sum::<f64>() / 5.
        } else {
            0.
        };
        let variance = x
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / 5.;
        let expected = x
            .iter()
            .zip(gain_values)
            .map(|(v, g)| {
                (f64::from(*v) - mean) / (variance + 1e-5).sqrt() * f64::from(g) + f64::from(*v)
            })
            .collect::<Vec<_>>();
        close(&gpu.read(&normalized)?, &expected, 2e-6)?;
    }
    let rotary_input = (0..12).map(|i| i as f32 / 7. - 1.).collect::<Vec<_>>();
    let r = gpu.upload_f32(&rotary_input)?;
    for interleaved in [false, true] {
        let result = gpu.rope(
            &r,
            7,
            RopeConfig {
                heads: 2,
                dim: 6,
                theta: 10000.,
                rotary: 4,
                interleaved,
            },
        )?;
        let mut expected = rotary_input
            .iter()
            .map(|v| f64::from(*v))
            .collect::<Vec<_>>();
        for head in 0..2 {
            for pair in 0..2 {
                let (a, b) = if interleaved {
                    (head * 6 + 2 * pair, head * 6 + 2 * pair + 1)
                } else {
                    (head * 6 + pair, head * 6 + pair + 2)
                };
                let phase = 7. / 10000f64.powf(pair as f64 / 2.);
                let (sin, cos) = phase.sin_cos();
                expected[a] = f64::from(rotary_input[a]) * cos - f64::from(rotary_input[b]) * sin;
                expected[b] = f64::from(rotary_input[b]) * cos + f64::from(rotary_input[a]) * sin;
            }
        }
        close(&gpu.read(&result)?, &expected, 4e-6)?;
    }
    for kind in [0, 1, 2, 3] {
        let result = gpu.gated(&input, &input, kind)?;
        let expected = x
            .iter()
            .map(|v| {
                let x = f64::from(*v);
                let activated = match kind {
                    0 => x / (1. + (-x).exp()),
                    1 => {
                        let z = x / std::f64::consts::SQRT_2;
                        let sign = z.signum();
                        let t = 1. / (1. + 0.3275911 * z.abs());
                        let erf = sign
                            * (1.
                                - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                                    - 0.284496736)
                                    * t
                                    + 0.254829592)
                                    * t
                                    * (-z * z).exp());
                        0.5 * x * (1. + erf)
                    }
                    2 => x.max(0.),
                    _ => {
                        0.5 * x
                            * (1.
                                + (std::f64::consts::FRAC_2_SQRT_PI / std::f64::consts::SQRT_2
                                    * (x + 0.044715 * x.powi(3)))
                                .tanh())
                    }
                };
                activated * x
            })
            .collect::<Vec<_>>();
        close(&gpu.read(&result)?, &expected, 3e-6)?;
    }
    let sum = gpu.combine(&input, &input, -0.25)?;
    close(
        &gpu.read(&sum)?,
        &x.iter().map(|v| f64::from(*v) * 0.75).collect::<Vec<_>>(),
        2e-7,
    )?;
    let keys = gpu.zeros(35 * 6)?;
    let values = gpu.zeros(35 * 6)?;
    let queries = (0..12).map(|i| i as f32 / 11. - 0.4).collect::<Vec<_>>();
    let query = gpu.upload_f32(&queries)?;
    let mut stored_k = Vec::new();
    let mut stored_v = Vec::new();
    for time in 0..35 {
        let k = (0..6)
            .map(|d| ((time * 6 + d) as f32 / 13.).sin())
            .collect::<Vec<_>>();
        let v = (0..6)
            .map(|d| ((time * 6 + d) as f32 / 17.).cos())
            .collect::<Vec<_>>();
        stored_k.push(k.clone());
        stored_v.push(v.clone());
        let key = gpu.upload_f32(&k)?;
        let value = gpu.upload_f32(&v)?;
        let result = gpu.attention(
            &query,
            &key,
            &value,
            (&keys, &values),
            time,
            AttentionConfig {
                qheads: 4,
                kvheads: 2,
                dim: 3,
                scale: 0.7,
            },
        )?;
        if [0, 31, 34].contains(&time) {
            let mut expected = Vec::new();
            for head in 0..4 {
                let kv = head / 2;
                let scores = (0..=time)
                    .map(|t| {
                        (0..3)
                            .map(|d| {
                                f64::from(queries[head * 3 + d])
                                    * f64::from(stored_k[t][kv * 3 + d])
                            })
                            .sum::<f64>()
                            * 0.7
                    })
                    .collect::<Vec<_>>();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores.iter().map(|s| (s - m).exp()).collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f64>();
                for d in 0..3 {
                    expected.push(
                        weights
                            .iter()
                            .enumerate()
                            .map(|(t, w)| w * f64::from(stored_v[t][kv * 3 + d]))
                            .sum::<f64>()
                            / denominator,
                    );
                }
            }
            close(&gpu.read(&result)?, &expected, 5e-6)?;
        }
    }
    let rotation = ReadoutRotation {
        u: vec![0.6, 0.8, 0., 0., 0.],
        v: vec![0., 0., 1., 0., 0.],
        tangent: -0.37,
    };
    let u = gpu.upload_f32(&rotation.u.iter().map(|v| *v as f32).collect::<Vec<_>>())?;
    let v = gpu.upload_f32(&rotation.v.iter().map(|v| *v as f32).collect::<Vec<_>>())?;
    let (a, b) = rotation.coefficients();
    let result = gpu.rotate(&input, &u, &v, a as f32, b as f32)?;
    close(
        &gpu.read(&result)?,
        &rotation.apply(&x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>())?,
        2e-6,
    )?;
    gpu.synchronize()?;
    println!(
        "GPU contracts verified: {} {} {}",
        gpu.backend(),
        gpu.name(),
        gpu.architecture()
    );
    Ok(())
}
