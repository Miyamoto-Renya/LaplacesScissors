//! 두 유한 출력의 구별도, 정규화 차이와 보호 회전의 해석적 방향.
use crate::rotation::{ProtectedSpace, ReadoutRotation, dot};
use anyhow::{Result, ensure};
use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};

pub(crate) fn doubles(values: &[f32]) -> Vec<f64> {
    values.iter().map(|v| f64::from(*v)).collect()
}
pub(crate) fn difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}
pub(crate) fn length(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

pub(crate) struct Distribution {
    pub probability: Vec<f64>,
    logarithm: Vec<f64>,
}
impl Distribution {
    pub fn new(logits: &[f32]) -> Result<Self> {
        ensure!(
            logits.len() > 1 && logits.iter().all(|v| v.is_finite()),
            "유한한 vocabulary 점수를 지정하세요."
        );
        let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mass = logits
            .iter()
            .map(|v| (f64::from(*v) - peak).exp())
            .sum::<f64>()
            .ln();
        let logarithm: Vec<_> = logits
            .iter()
            .map(|v| (f64::from(*v) - peak) - mass)
            .collect();
        let probability = logarithm.iter().map(|v| v.exp()).collect();
        Ok(Self {
            probability,
            logarithm,
        })
    }
    pub fn fisher(&self, vector: &[f64]) -> Vec<f64> {
        let mean = dot(&self.probability, vector);
        self.probability
            .iter()
            .zip(vector)
            .map(|(p, v)| p * (v - mean))
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairMetric {
    pub jeffreys: f64,
    pub probability_difference_norm: f64,
    pub secant_absolute_residual: f64,
    pub secant_relative_residual: f64,
}
pub(crate) struct ProbabilityPair {
    pub plus: Distribution,
    pub minus: Distribution,
    pub delta_logits: Vec<f64>,
    pub delta_probability: Vec<f64>,
    pub metric: PairMetric,
}
impl ProbabilityPair {
    pub fn new(plus: &[f32], minus: &[f32]) -> Result<Self> {
        ensure!(
            plus.len() == minus.len(),
            "두 출력의 vocabulary를 맞추세요."
        );
        let delta_logits: Vec<_> = plus
            .iter()
            .zip(minus)
            .map(|(a, b)| f64::from(*a) - f64::from(*b))
            .collect();
        let plus = Distribution::new(plus)?;
        let minus = Distribution::new(minus)?;
        let delta_probability = difference(&plus.probability, &minus.probability);
        let mut means = Vec::with_capacity(delta_logits.len());
        let mut divergence = 0.;
        for i in 0..delta_logits.len() {
            let log_difference = plus.logarithm[i] - minus.logarithm[i];
            divergence += delta_probability[i] * log_difference;
            let distance = log_difference.abs();
            let factor = if distance == 0. {
                1.
            } else {
                -(-distance).exp_m1() / distance
            };
            means.push(plus.logarithm[i].max(minus.logarithm[i]).exp() * factor);
        }
        let mean_mass: f64 = means.iter().sum();
        ensure!(
            mean_mass > 0. && mean_mass.is_finite(),
            "두 확률의 로그평균 범위를 확인하세요."
        );
        let offset = dot(&means, &delta_logits) / mean_mass;
        let residual = means
            .iter()
            .zip(&delta_logits)
            .zip(&delta_probability)
            .map(|((m, z), p)| (m * (z - offset) - p).powi(2))
            .sum::<f64>()
            .sqrt();
        let probability_difference_norm = length(&delta_probability);
        let metric = PairMetric {
            jeffreys: divergence.max(0.),
            probability_difference_norm,
            secant_absolute_residual: residual,
            secant_relative_residual: residual / probability_difference_norm.max(f64::MIN_POSITIVE),
        };
        Ok(Self {
            plus,
            minus,
            delta_logits,
            delta_probability,
            metric,
        })
    }
    /// 두 hidden의 중간값을 고정할 때 출력 차이에 작용하는 실제 목적함수 미분.
    pub fn separation_gradient(&self) -> Vec<f64> {
        let a = self.plus.fisher(&self.delta_logits);
        let b = self.minus.fisher(&self.delta_logits);
        self.delta_probability
            .iter()
            .zip(a.iter().zip(&b))
            .map(|(p, (a, b))| p + 0.5 * (a + b))
            .collect()
    }
}

pub(crate) fn weak_combination(
    responses: &[Vec<f64>],
    base: &Distribution,
) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(
        !responses.is_empty() && responses.iter().all(|r| r.len() == base.probability.len()),
        "방향 응답의 vocabulary와 개수를 확인하세요."
    );
    let transformed: Vec<_> = responses.iter().map(|r| base.fisher(r)).collect();
    let rank = responses.len();
    let gram = DMatrix::from_fn(rank, rank, |i, j| {
        0.5 * (dot(&responses[i], &transformed[j]) + dot(&responses[j], &transformed[i]))
    });
    let spectrum = SymmetricEigen::new(gram);
    let index = (0..rank)
        .min_by(|i, j| spectrum.eigenvalues[*i].total_cmp(&spectrum.eigenvalues[*j]))
        .unwrap();
    let mut coefficients: Vec<_> = spectrum
        .eigenvectors
        .column(index)
        .iter()
        .copied()
        .collect();
    let pivot = coefficients
        .iter()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap();
    if *pivot < 0. {
        coefficients.iter_mut().for_each(|v| *v = -*v);
    }
    Ok((coefficients, spectrum.eigenvalues.iter().copied().collect()))
}

/// diag(g)·normalization의 유한 차이 전치를 O(width)에 적용한다.
pub(crate) fn norm_secant_transpose(
    plus: &[f32],
    minus: &[f32],
    gain: &[f64],
    epsilon: f64,
    centered: bool,
    vector: &[f64],
) -> Result<Vec<f64>> {
    let width = plus.len();
    ensure!(
        width > 0
            && minus.len() == width
            && gain.len() == width
            && vector.len() == width
            && epsilon.is_finite()
            && epsilon > 0.,
        "정규화 차이의 차원과 epsilon을 확인하세요."
    );
    let mut left = doubles(plus);
    let mut right = doubles(minus);
    if centered {
        for side in [&mut left, &mut right] {
            let mean = side.iter().sum::<f64>() / width as f64;
            side.iter_mut().for_each(|v| *v -= mean);
        }
    }
    let radius = |x: &[f64]| (dot(x, x) / width as f64 + epsilon).sqrt();
    let (a, b) = (radius(&left), radius(&right));
    let midpoint: Vec<_> = left
        .iter()
        .zip(&right)
        .map(|(x, y)| (x + y) * 0.5)
        .collect();
    let weighted: Vec<_> = vector.iter().zip(gain).map(|(v, g)| v * g).collect();
    let radial = 2. * dot(&midpoint, &weighted) / (width as f64 * a * b * (a + b));
    let mut result: Vec<_> = weighted
        .iter()
        .zip(&midpoint)
        .map(|(v, m)| 0.5 * (a.recip() + b.recip()) * v - radial * m)
        .collect();
    if centered {
        let mean = result.iter().sum::<f64>() / width as f64;
        result.iter_mut().for_each(|v| *v -= mean);
    }
    ensure!(
        result.iter().all(|v| v.is_finite()),
        "유한한 정규화 차이 응답을 확인하세요."
    );
    Ok(result)
}

pub(crate) fn separating_plane(
    space: &ProtectedSpace,
    delta: &[f64],
    gradient: &[f64],
) -> Result<Option<(ReadoutRotation, f64)>> {
    let mut first = space.project(delta)?;
    let magnitude = length(&first);
    if magnitude <= 1e-10 * length(delta).max(f64::MIN_POSITIVE) {
        return Ok(None);
    }
    first.iter_mut().for_each(|v| *v /= magnitude);
    let mut second = space.project(gradient)?;
    for _ in 0..2 {
        let parallel = dot(&first, &second);
        for (v, u) in second.iter_mut().zip(&first) {
            *v -= parallel * u;
        }
    }
    let response = length(&second);
    if response <= 1e-10 * length(gradient).max(f64::MIN_POSITIVE) {
        return Ok(None);
    }
    // 제품의 J=uvᵀ-vuᵀ 방향에 맞춰 두 번째 축을 정한다.
    second.iter_mut().for_each(|v| *v /= -response);
    Ok(Some((
        ReadoutRotation {
            u: first,
            v: second,
            tangent: 0.,
        },
        magnitude * response,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_softmax_difference_keeps_gauge_and_equal_endpoints() -> Result<()> {
        for (a, b) in [
            (vec![3., -12., 1.], vec![-7., 2., 0.]),
            (vec![1000., -1000., 0.], vec![-1000., 1000., 0.]),
            (vec![2., 2., 2.], vec![2., 2., 2.]),
            (vec![5., 2., -1.], vec![2., -1., -4.]),
        ] {
            let pair = ProbabilityPair::new(&a, &b)?;
            assert!(pair.metric.secant_absolute_residual < 1e-12);
            assert!(pair.metric.jeffreys >= 0.);
        }
        assert_eq!(
            ProbabilityPair::new(&[5., 2., -1.], &[2., -1., -4.])?
                .metric
                .jeffreys,
            0.
        );
        Ok(())
    }
    #[test]
    fn normalization_secant_matches_finite_rms_and_centered_changes() -> Result<()> {
        let a = [2f32, -3., 0.5, 7.];
        let b = [-4f32, 1., 0.25, 0.];
        let gain = [0.2, 3., -0.7, 1.];
        let output = [1., -2., 0.5, 0.25];
        for centered in [false, true] {
            let evaluate = |x: &[f32]| {
                let mean = if centered {
                    x.iter().map(|v| f64::from(*v)).sum::<f64>() / 4.
                } else {
                    0.
                };
                let shifted: Vec<_> = x.iter().map(|v| f64::from(*v) - mean).collect();
                let scale = (dot(&shifted, &shifted) / 4. + 1e-5).sqrt();
                shifted
                    .iter()
                    .zip(gain)
                    .map(|(v, g)| v * g / scale)
                    .collect::<Vec<_>>()
            };
            let adjoint = norm_secant_transpose(&a, &b, &gain, 1e-5, centered, &output)?;
            let predicted = dot(&adjoint, &difference(&doubles(&a), &doubles(&b)));
            let measured = dot(&output, &difference(&evaluate(&a), &evaluate(&b)));
            assert!((predicted - measured).abs() < 1e-12);
        }
        Ok(())
    }
    #[test]
    fn rotation_direction_increases_separation_and_fixes_midpoint() -> Result<()> {
        let mut space = ProtectedSpace::new(4);
        space.include(&[0., 0., 2., 0.])?;
        let delta = [1., 0., 0., 0.];
        let gradient = [0.2, 3., 7., 0.];
        let (mut plane, derivative) = separating_plane(&space, &delta, &gradient)?.unwrap();
        plane.tangent = (0.001f64 / 2.).tan();
        let change = difference(&plane.apply(&delta)?, &delta);
        assert!((dot(&gradient, &change) / 0.001 - derivative).abs() < 0.001);
        assert_eq!(plane.apply(&[0., 0., 2., 0.])?, vec![0., 0., 2., 0.]);
        assert!((length(&plane.apply(&delta)?) - 1.).abs() < 1e-14);
        Ok(())
    }
}
