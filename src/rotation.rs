//! 보호 공간과 유리 회전 좌표로 출력 행렬의 수정안을 계산한다.
pub use crate::interval::{FeasibleSet, FeasibleSummary, Interval};
use anyhow::{Result, ensure};
use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};

pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
fn length(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}
fn unit(x: &mut [f64]) -> bool {
    let n = length(x);
    if !n.is_finite() || n <= 1e-13 {
        return false;
    }
    x.iter_mut().for_each(|v| *v /= n);
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadoutRotation {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub tangent: f64,
}
impl ReadoutRotation {
    pub fn validate(&self, width: usize) -> Result<()> {
        ensure!(
            width > 1 && self.u.len() == width && self.v.len() == width,
            "회전 벡터의 차원을 확인하세요."
        );
        ensure!(
            self.u.iter().chain(&self.v).all(|x| x.is_finite()) && self.tangent.is_finite(),
            "유한 회전 좌표가 필요합니다."
        );
        ensure!(
            (dot(&self.u, &self.u) - 1.).abs() < 1e-8
                && (dot(&self.v, &self.v) - 1.).abs() < 1e-8
                && dot(&self.u, &self.v).abs() < 1e-8,
            "회전 평면에는 직교 단위 벡터를 지정하세요."
        );
        ensure!(
            self.tangent.abs() <= 1e6,
            "반각 좌표의 절댓값은 1e6 이내로 지정하세요."
        );
        Ok(())
    }
    pub fn coefficients(&self) -> (f64, f64) {
        let t = self.tangent;
        (-2. * t * t / (1. + t * t), 2. * t / (1. + t * t))
    }
    pub fn apply(&self, hidden: &[f64]) -> Result<Vec<f64>> {
        self.validate(hidden.len())?;
        ensure!(
            hidden.iter().all(|x| x.is_finite()),
            "유한 내부 벡터가 필요합니다."
        );
        let (a, b) = self.coefficients();
        let x = dot(&self.u, hidden);
        let y = dot(&self.v, hidden);
        Ok(hidden
            .iter()
            .enumerate()
            .map(|(i, h)| h + self.u[i] * (a * x + b * y) + self.v[i] * (a * y - b * x))
            .collect())
    }
    pub fn norm_squared(&self, wu: &[f32], wv: &[f32]) -> f64 {
        let k = wu
            .iter()
            .chain(wv)
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>();
        4. * self.tangent.powi(2) / (1. + self.tangent.powi(2)) * k
    }
}

pub struct ProtectedSpace {
    width: usize,
    basis: Vec<Vec<f64>>,
    pub samples: usize,
}
impl ProtectedSpace {
    pub fn new(width: usize) -> Self {
        Self {
            width,
            basis: Vec::new(),
            samples: 0,
        }
    }
    pub fn rank(&self) -> usize {
        self.basis.len()
    }
    pub fn project(&self, values: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            values.len() == self.width && values.iter().all(|x| x.is_finite()),
            "보호 공간의 입력 차원과 유한값을 확인하세요."
        );
        let mut residual = values.to_vec();
        for _ in 0..2 {
            for axis in &self.basis {
                let c = dot(axis, &residual);
                for (x, a) in residual.iter_mut().zip(axis) {
                    *x -= c * a;
                }
            }
        }
        Ok(residual)
    }
    pub fn include(&mut self, values: &[f64]) -> Result<()> {
        let mut residual = self.project(values)?;
        self.samples += 1;
        if length(&residual) > 1e-10 * length(values).max(f64::MIN_POSITIVE)
            && self.rank() < self.width
            && unit(&mut residual)
        {
            self.basis.push(residual);
        }
        Ok(())
    }
    fn plane(&self, first: &[f64], second: &[f64]) -> Result<Option<ReadoutRotation>> {
        let mut u = self.project(first)?;
        if !unit(&mut u) {
            return Ok(None);
        }
        let mut v = self.project(second)?;
        for _ in 0..2 {
            let c = dot(&u, &v);
            for (x, a) in v.iter_mut().zip(&u) {
                *x -= c * a;
            }
        }
        if !unit(&mut v) {
            return Ok(None);
        }
        Ok(Some(ReadoutRotation { u, v, tangent: 0. }))
    }
}

pub struct Pressure {
    pub hidden: Vec<f64>,
    pub difference: Vec<f64>,
    pub weight: f64,
}
/// 부족한 토큰 margin의 외적을 작은 반대칭 행렬로 모아 회전 평면을 구한다.
pub fn planes(
    space: &ProtectedSpace,
    pressures: &[Pressure],
    maximum: usize,
) -> Result<Vec<ReadoutRotation>> {
    ensure!(maximum > 0, "양의 후보 평면 수를 지정하세요.");
    let mut span = ProtectedSpace::new(space.width);
    let mut projected = Vec::new();
    for p in pressures {
        ensure!(
            p.weight.is_finite() && p.weight >= 0.,
            "유한한 양의 부족량을 지정하세요."
        );
        let a = space.project(&p.difference)?;
        let b = space.project(&p.hidden)?;
        span.include(&a)?;
        span.include(&b)?;
        projected.push((a, b, p.weight));
    }
    let n = span.rank();
    let mut result = Vec::new();
    let mut insert = |plane: ReadoutRotation| {
        let duplicate = result.iter().any(|old: &ReadoutRotation| {
            let overlap = dot(&old.u, &plane.u).powi(2)
                + dot(&old.u, &plane.v).powi(2)
                + dot(&old.v, &plane.u).powi(2)
                + dot(&old.v, &plane.v).powi(2);
            overlap > 2. - 1e-8
        });
        if !duplicate && result.len() < maximum {
            result.push(plane);
        }
    };
    if n >= 2 {
        let scale = projected
            .iter()
            .map(|p| p.2)
            .fold(0., f64::max)
            .max(f64::MIN_POSITIVE);
        let mut skew = DMatrix::<f64>::zeros(n, n);
        for (a, b, w) in &projected {
            let x: Vec<_> = span.basis.iter().map(|v| dot(a, v)).collect();
            let y: Vec<_> = span.basis.iter().map(|v| dot(b, v)).collect();
            for row in 0..n {
                for col in 0..n {
                    skew[(row, col)] += w / scale * (x[row] * y[col] - y[row] * x[col]);
                }
            }
        }
        let eigen = SymmetricEigen::new(skew.transpose() * &skew);
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by(|a, b| eigen.eigenvalues[*b].total_cmp(&eigen.eigenvalues[*a]));
        for index in order {
            if eigen.eigenvalues[index] <= 1e-20 {
                continue;
            }
            let x = eigen.eigenvectors.column(index);
            let y = -&skew * x;
            let mut a = vec![0.; space.width];
            let mut b = a.clone();
            for k in 0..n {
                for j in 0..space.width {
                    a[j] += x[k] * span.basis[k][j];
                    b[j] += y[k] * span.basis[k][j];
                }
            }
            if let Some(p) = space.plane(&a, &b)? {
                insert(p);
            }
        }
    }
    for (a, b, _) in projected {
        if let Some(p) = space.plane(&a, &b)? {
            insert(p);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_gram_and_protected_outputs_survive_finite_rotation() {
        let rotation = ReadoutRotation {
            u: vec![0.6, 0.8, 0., 0.],
            v: vec![0., 0., 1., 0.],
            tangent: 0.73,
        };
        let rows: Vec<Vec<f64>> = (0..7)
            .map(|r| (0..4).map(|c| ((r * 4 + c) as f64 / 3.).sin()).collect())
            .collect();
        let columns: Vec<_> = (0..4)
            .map(|j| {
                let mut e = vec![0.; 4];
                e[j] = 1.;
                rotation.apply(&e).unwrap()
            })
            .collect();
        let changed: Vec<Vec<_>> = rows
            .iter()
            .map(|r| columns.iter().map(|c| dot(r, c)).collect())
            .collect();
        for (i, row) in rows.iter().enumerate() {
            for (j, other) in rows.iter().enumerate() {
                assert!((dot(row, other) - dot(&changed[i], &changed[j])).abs() < 2e-14);
            }
            assert!(
                (dot(row, &[-0.8, 0.6, 0., 2.]) - dot(&changed[i], &[-0.8, 0.6, 0., 2.])).abs()
                    < 2e-14
            );
        }
        let actual: f64 = rows
            .iter()
            .flatten()
            .zip(changed.iter().flatten())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let wu: Vec<_> = rows.iter().map(|w| dot(w, &rotation.u) as f32).collect();
        let wv: Vec<_> = rows.iter().map(|w| dot(w, &rotation.v) as f32).collect();
        assert!((actual - rotation.norm_squared(&wu, &wv)).abs() < 1e-6);
    }
    #[test]
    fn interval_intersections_match_independent_polynomial_evaluation() {
        let mut state = 1234567u64;
        for _ in 0..200 {
            let mut coefficients = Vec::new();
            let mut feasible = FeasibleSet::new(3.).unwrap();
            for _ in 0..8 {
                let mut draw = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % 17) as f64 - 8.
                };
                let (a, b, c) = (draw(), draw(), draw());
                coefficients.push((a, b, c));
                feasible.constrain(a, b, c).unwrap();
                for j in 0..601 {
                    let t = -3. + j as f64 / 100.;
                    let values: Vec<_> = coefficients
                        .iter()
                        .map(|(a, b, c)| a * t * t + b * t + c)
                        .collect();
                    if values.iter().any(|v| v.abs() < 1e-10) {
                        continue;
                    }
                    let expected = values.iter().all(|v| *v > 0.);
                    let actual = feasible
                        .intervals()
                        .iter()
                        .any(|s| s.lower <= t && t <= s.upper);
                    assert_eq!(actual, expected, "{coefficients:?} at {t}");
                }
                if let Some(t) = feasible.nearest() {
                    assert!(
                        coefficients
                            .iter()
                            .all(|(a, b, c)| a * t * t + b * t + c >= -1e-10)
                    );
                }
            }
        }
    }
    #[test]
    fn simultaneous_repair_and_protected_axis() {
        let mut f = FeasibleSet::new(2.).unwrap();
        f.constrain(-0.3, 2., -0.3).unwrap();
        f.constrain(1.8, 0., -0.2).unwrap();
        let t = f.nearest().unwrap();
        assert!((t - 1. / 3.).abs() < 1e-12);
        let edit = ReadoutRotation {
            u: vec![1., 0., 0.],
            v: vec![0., 1., 0.],
            tangent: t,
        };
        let w = [0., -1., 1.];
        let first = dot(&w, &edit.apply(&[1., 0., 0.]).unwrap());
        let second = dot(&w, &edit.apply(&[0., 1., 0.]).unwrap());
        assert!((first - 0.6).abs() < 1e-12 && (second + 0.8).abs() < 1e-12);
        assert_eq!(edit.apply(&[0., 0., 1.]).unwrap(), vec![0., 0., 1.]);
        assert!((first * first + second * second - 1.).abs() < 1e-12);
    }
    #[test]
    fn disconnected_feasible_sets_and_singleton() {
        let mut s = FeasibleSet::new(3.).unwrap();
        s.constrain(1., 0., -1.).unwrap();
        assert_eq!(
            s.intervals(),
            vec![
                Interval {
                    lower: -3.,
                    upper: -1.
                },
                Interval {
                    lower: 1.,
                    upper: 3.
                }
            ]
        );
        s.constrain(-1., 4., -4.).unwrap();
        assert_eq!(s.nearest(), Some(2.));
        s.constrain(0., -1., 1.).unwrap();
        assert!(s.nearest().is_none());
    }
    #[test]
    fn contradictory_observations_have_empty_intersection() {
        let mut s = FeasibleSet::new(2.).unwrap();
        s.constrain(0., 1., -1.).unwrap();
        s.constrain(0., -1., -1.).unwrap();
        assert!(s.nearest().is_none());
    }
    #[test]
    fn protected_span_is_invariant_to_redundant_scaled_inputs() {
        let mut a = ProtectedSpace::new(4);
        a.include(&[1., 2., 0., 0.]).unwrap();
        a.include(&[2., 4., 0., 0.]).unwrap();
        assert_eq!(a.rank(), 1);
        let ps = vec![Pressure {
            hidden: vec![0., 0., 1., 0.],
            difference: vec![0., 0., 0., 1.],
            weight: 1.,
        }];
        let p = planes(&a, &ps, 4).unwrap();
        assert_eq!(p.len(), 1);
        assert!(dot(&p[0].u, &[1., 2., 0., 0.]).abs() < 1e-12);
        assert!(dot(&p[0].v, &[1., 2., 0., 0.]).abs() < 1e-12);
    }
    #[test]
    fn malformed_rotations_fail_at_boundary() {
        assert!(
            ReadoutRotation {
                u: vec![1., 0.],
                v: vec![1., 0.],
                tangent: 0.
            }
            .validate(2)
            .is_err()
        );
        assert!(
            ReadoutRotation {
                u: vec![1., 0.],
                v: vec![0., 1.],
                tangent: f64::NAN
            }
            .validate(2)
            .is_err()
        );
    }
}
