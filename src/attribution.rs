//! 유한 readout 수정의 손실 변화와 가중치 행·좌표별 기여를 계산한다.
use crate::{
    diagnosis::Deadline,
    gpu::Gpu,
    model::{Frame, Model},
    response::ReadoutEdit,
    text::Text,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub struct LossPoint<'a> {
    pub before: &'a Frame,
    pub after: &'a [f32],
    pub target: u32,
    pub weight: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoordinateCredit {
    pub column: usize,
    pub delta_weight: f64,
    pub credit: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowCredit {
    pub row: u32,
    pub token: String,
    pub credit: f64,
    pub restored_loss: f64,
    pub restoration_increase: f64,
    pub coordinate_sum: f64,
    pub coordinate_sum_residual: f64,
    pub leading_coordinates: Vec<CoordinateCredit>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attribution {
    pub unit: String,
    pub path: String,
    pub loss_before: f64,
    pub loss_after: f64,
    pub credit_sum: f64,
    pub integration_residual: f64,
    pub integration_converged: bool,
    pub panels: usize,
    pub points: usize,
    pub leading_rows: Vec<RowCredit>,
    pub remaining_rows_credit: f64,
}
pub fn loss(logits: &[f32], target: u32) -> f64 {
    let top = logits
        .iter()
        .map(|x| f64::from(*x))
        .fold(f64::NEG_INFINITY, f64::max);
    top + logits
        .iter()
        .map(|x| (f64::from(*x) - top).exp())
        .sum::<f64>()
        .ln()
        - f64::from(logits[target as usize])
}
fn restored_loss(z: &[f32], row: usize, original: f32, target: u32) -> f64 {
    let maximum = z
        .iter()
        .enumerate()
        .map(|(j, &v)| f64::from(if j == row { original } else { v }))
        .fold(f64::NEG_INFINITY, f64::max);
    let total = z
        .iter()
        .enumerate()
        .map(|(j, &v)| (f64::from(if j == row { original } else { v }) - maximum).exp())
        .sum::<f64>();
    maximum + total.ln()
        - f64::from(if row == target as usize {
            original
        } else {
            z[target as usize]
        })
}
fn integrated_probabilities(
    points: &[LossPoint<'_>],
    panels: usize,
    deadline: &Deadline,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let vocab = points[0].after.len();
    let mut credits = vec![0.; vocab];
    let mut gradients = Vec::with_capacity(points.len());
    for point in points {
        let mut integral = vec![0.; vocab];
        let mut probabilities = vec![0.; vocab];
        for k in 0..=panels {
            deadline.check()?;
            let t = k as f64 / panels as f64;
            let mut maximum = f64::NEG_INFINITY;
            for (j, z) in probabilities.iter_mut().enumerate() {
                *z = f64::from(point.before.logits[j]) * (1. - t) + f64::from(point.after[j]) * t;
                maximum = maximum.max(*z);
            }
            let mut total = 0.;
            for z in &mut probabilities {
                *z = (*z - maximum).exp();
                total += *z;
            }
            let weight = (if k == 0 || k == panels {
                1.
            } else if k % 2 == 0 {
                2.
            } else {
                4.
            }) / (3. * panels as f64);
            for (j, &p) in probabilities.iter().enumerate() {
                integral[j] +=
                    weight * (p / total - if j == point.target as usize { 1. } else { 0. });
            }
        }
        for j in 0..vocab {
            credits[j] -= point.weight
                * (f64::from(point.after[j]) - f64::from(point.before.logits[j]))
                * integral[j];
        }
        gradients.push(integral);
    }
    Ok((credits, gradients))
}
pub fn analyze(
    model: &Model,
    gpu: &mut Gpu,
    text: &Text,
    edit: &ReadoutEdit,
    points: &[LossPoint<'_>],
    row_count: usize,
    tolerance: f64,
    unit: &str,
    deadline: &Deadline,
) -> Result<Attribution> {
    ensure!(
        !points.is_empty() && tolerance.is_finite() && tolerance > 0.,
        "기여 분석의 관측과 적분 허용 오차를 지정하세요."
    );
    let vocab = model.vocab_size();
    ensure!(
        points.iter().all(|p| p.after.len() == vocab
            && p.before.logits.len() == vocab
            && (p.target as usize) < vocab
            && p.weight.is_finite()
            && p.weight >= 0.),
        "기여 분석의 관측 조건을 확인하세요."
    );
    let before: f64 = points
        .iter()
        .map(|p| p.weight * loss(&p.before.logits, p.target))
        .sum();
    let after: f64 = points
        .iter()
        .map(|p| p.weight * loss(p.after, p.target))
        .sum();
    let expected = before - after;
    let mut panels = 16;
    let (credits, gradients, residual) = loop {
        let (c, g) = integrated_probabilities(points, panels, deadline)?;
        let error = (c.iter().sum::<f64>() - expected).abs();
        if error <= tolerance * (1. + expected.abs()) || panels >= 256 {
            break (c, g, error);
        }
        panels *= 2;
    };
    let mut ranking: Vec<usize> = (0..vocab).collect();
    let count = row_count.min(vocab);
    if count < vocab {
        ranking.select_nth_unstable_by(count, |&a, &b| {
            credits[b]
                .abs()
                .total_cmp(&credits[a].abs())
                .then(a.cmp(&b))
        });
    }
    ranking.truncate(count);
    ranking.sort_by(|&a, &b| {
        credits[b]
            .abs()
            .total_cmp(&credits[a].abs())
            .then(a.cmp(&b))
    });
    let subspace = edit.subspace();
    let mut leading = Vec::new();
    for row in ranking {
        deadline.check()?;
        let source = model
            .head_row(gpu, row as u32)?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        let delta = subspace.delta_row(&source);
        let mut coordinate = vec![0.; source.len()];
        for (point, gradient) in points.iter().zip(&gradients) {
            for j in 0..coordinate.len() {
                coordinate[j] -=
                    point.weight * delta[j] * f64::from(point.before.hidden[j]) * gradient[row];
            }
        }
        let sum = coordinate.iter().sum::<f64>();
        let mut cols: Vec<usize> = (0..coordinate.len()).collect();
        cols.sort_by(|&a, &b| {
            coordinate[b]
                .abs()
                .total_cmp(&coordinate[a].abs())
                .then(a.cmp(&b))
        });
        let restored = points
            .iter()
            .map(|p| p.weight * restored_loss(p.after, row, p.before.logits[row], p.target))
            .sum::<f64>();
        leading.push(RowCredit {
            row: row as u32,
            token: text.decode(&[row as u32])?,
            credit: credits[row],
            restored_loss: restored,
            restoration_increase: restored - after,
            coordinate_sum: sum,
            coordinate_sum_residual: (sum - credits[row]).abs(),
            leading_coordinates: cols
                .into_iter()
                .take(6)
                .map(|column| CoordinateCredit {
                    column,
                    delta_weight: delta[column],
                    credit: coordinate[column],
                })
                .collect(),
        });
    }
    let sum = credits.iter().sum::<f64>();
    let remaining = sum - leading.iter().map(|r| r.credit).sum::<f64>();
    Ok(Attribution{unit:unit.into(),path:"straight readout chord; native F32 endpoint logits; F64 softmax integration; coordinate residual measures finite precision".into(),
        loss_before:before,loss_after:after,credit_sum:sum,integration_residual:residual,
        integration_converged:residual<=tolerance*(1.+expected.abs()),panels,points:points.len(),leading_rows:leading,remaining_rows_credit:remaining})
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn signed_credits_sum_to_finite_loss_change() {
        let frame = Frame {
            hidden: vec![1.],
            logits: vec![3., -1., 0.],
        };
        let after = vec![0., 2., 1.];
        let points = [LossPoint {
            before: &frame,
            after: &after,
            target: 1,
            weight: 1.,
        }];
        let (c, _) = integrated_probabilities(
            &points,
            128,
            &Deadline::after(Duration::from_secs(2)).unwrap(),
        )
        .unwrap();
        assert!((c.iter().sum::<f64>() - (loss(&frame.logits, 1) - loss(&after, 1))).abs() < 1e-8);
        assert!(c[2] < 0.);
        assert!((restored_loss(&after, 1, -1., 1) - loss(&[0., -1., 1.], 1)).abs() < 1e-14);
    }
}
