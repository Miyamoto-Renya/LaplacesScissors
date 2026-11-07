//! 관측 사이의 응답 Gram으로 보호 조건을 지키는 공동 회전을 구성한다.
use crate::rotation::{ProtectedSpace, ReadoutRotation, dot};
use anyhow::{Result, ensure};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubspaceRotation {
    pub basis: Vec<Vec<f64>>,
    pub transform: Vec<Vec<f64>>,
}
impl SubspaceRotation {
    pub fn identity(basis: Vec<Vec<f64>>) -> Self {
        let n = basis.len();
        Self {
            basis,
            transform: (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1. } else { 0. }).collect())
                .collect(),
        }
    }
    pub fn matrix(&self) -> DMatrix<f64> {
        DMatrix::from_fn(self.basis.len(), self.basis.len(), |i, j| {
            self.transform[i][j]
        })
    }
    pub fn set_matrix(&mut self, q: &DMatrix<f64>) {
        self.transform = (0..q.nrows())
            .map(|i| (0..q.ncols()).map(|j| q[(i, j)]).collect())
            .collect();
    }
    pub fn validate(&self, width: usize) -> Result<()> {
        let n = self.basis.len();
        ensure!(
            n > 0 && n <= width && n <= 256 && self.transform.len() == n,
            "공동 회전의 rank와 내부 행렬 크기를 확인하세요."
        );
        ensure!(
            self.basis
                .iter()
                .all(|v| v.len() == width && v.iter().all(|x| x.is_finite()))
                && self
                    .transform
                    .iter()
                    .all(|v| v.len() == n && v.iter().all(|x| x.is_finite())),
            "공동 회전에는 유한한 기저와 정방행렬이 필요합니다."
        );
        for i in 0..n {
            for j in 0..=i {
                ensure!(
                    (dot(&self.basis[i], &self.basis[j]) - if i == j { 1. } else { 0. }).abs()
                        < 1e-8,
                    "공동 회전 기저의 직교 조건을 확인하세요."
                );
            }
        }
        let q = self.matrix();
        ensure!(
            (q.transpose() * &q - DMatrix::identity(n, n)).amax() < 1e-8 && q.determinant() > 0.,
            "공동 회전의 직교성과 방향을 확인하세요."
        );
        Ok(())
    }
    pub fn coordinates(&self, x: &[f64]) -> Vec<f64> {
        self.basis.iter().map(|u| dot(u, x)).collect()
    }
    pub fn apply(&self, h: &[f64]) -> Vec<f64> {
        let x = DVector::from_vec(self.coordinates(h));
        let delta = (self.matrix() - DMatrix::identity(x.len(), x.len())) * x;
        let mut result = h.to_vec();
        for (u, c) in self.basis.iter().zip(delta.iter()) {
            for (value, axis) in result.iter_mut().zip(u) {
                *value += c * axis;
            }
        }
        result
    }
    pub fn delta_row(&self, row: &[f64]) -> Vec<f64> {
        let x = DVector::from_vec(self.coordinates(row));
        let delta = (self.matrix().transpose() - DMatrix::identity(x.len(), x.len())) * x;
        (0..row.len())
            .map(|j| {
                self.basis
                    .iter()
                    .zip(delta.iter())
                    .map(|(u, c)| u[j] * c)
                    .sum()
            })
            .collect()
    }
    pub fn angle_degrees(&self) -> f64 {
        let q = self.matrix();
        let eig = SymmetricEigen::new((&q + q.transpose()) * 0.5);
        eig.eigenvalues.min().clamp(-1., 1.).acos().to_degrees()
    }
}

/// 고정 기저의 출력 상 F = WU에서 한 번 구성하는 가중치 비용.
pub struct WeightMetric {
    gram: DMatrix<f64>,
}
impl WeightMetric {
    pub fn from_images(images: &[Vec<f32>]) -> Result<Self> {
        let rows = images.first().map_or(0, Vec::len);
        ensure!(
            rows > 0
                && images
                    .iter()
                    .all(|v| v.len() == rows && v.iter().all(|x| x.is_finite())),
            "출력 기저의 크기와 유한한 원소를 확인하세요."
        );
        let image = DMatrix::from_iterator(
            rows,
            images.len(),
            images.iter().flatten().map(|&x| f64::from(x)),
        );
        Ok(Self {
            gram: image.transpose() * image,
        })
    }
    pub fn norm_squared(&self, rotation: &DMatrix<f64>) -> f64 {
        let delta = rotation - DMatrix::identity(rotation.nrows(), rotation.ncols());
        delta.component_mul(&(&self.gram * &delta)).sum().max(0.)
    }
}

/// 관측 열공간과 방향을 보존하는 직교 완성을 기저 설정 시 고정한다.
pub struct ObservedSpace {
    frame: DMatrix<f64>,
    pub rank: usize,
}
impl ObservedSpace {
    pub fn new(coordinates: &[Vec<f64>], width: usize) -> Result<Self> {
        ensure!(
            width > 0
                && coordinates
                    .iter()
                    .all(|x| x.len() == width && x.iter().all(|v| v.is_finite())),
            "관측 좌표의 차원과 유한한 원소를 확인하세요."
        );
        let h = DMatrix::from_fn(width, coordinates.len(), |i, j| coordinates[j][i]);
        if h.ncols() == 0 {
            return Ok(Self {
                frame: DMatrix::identity(width, width),
                rank: 0,
            });
        }
        let svd = h.svd(true, false);
        let cutoff = (svd.singular_values.max() * 1e-9).max(1e-12);
        let selected: Vec<usize> = (0..svd.singular_values.len())
            .filter(|&i| svd.singular_values[i] > cutoff)
            .collect();
        let u = svd.u.expect("requested left singular vectors");
        let basis = DMatrix::from_fn(width, selected.len(), |i, j| u[(i, selected[j])]);
        Ok(Self {
            frame: oriented_frame(&basis)?,
            rank: selected.len(),
        })
    }
    pub fn complete(&self, rotation: &DMatrix<f64>, metric: &WeightMetric) -> Result<DMatrix<f64>> {
        let n = rotation.nrows();
        if self.rank == n {
            return Ok(rotation.clone());
        }
        if self.rank == 0 {
            return Ok(DMatrix::identity(n, n));
        }
        let observed = self.frame.columns(0, self.rank);
        let moved = rotation * observed;
        let destination = oriented_frame(&moved)?;
        let remaining = n - self.rank;
        let source_free = self.frame.columns(self.rank, remaining);
        let target_free = destination.columns(self.rank, remaining);
        let cross = source_free.transpose() * &metric.gram * target_free;
        let svd = cross.svd(true, true);
        let left = svd.u.expect("requested left singular vectors");
        let mut right = svd
            .v_t
            .expect("requested right singular vectors")
            .transpose();
        if (&right * left.transpose()).determinant() < 0. {
            let smallest = svd.singular_values.imin();
            right.column_mut(smallest).scale_mut(-1.);
        }
        Ok(moved * observed.transpose()
            + target_free * (right * left.transpose()) * source_free.transpose())
    }
}

fn oriented_frame(basis: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let n = basis.nrows();
    let mut frame = DMatrix::zeros(n, n);
    frame.columns_mut(0, basis.ncols()).copy_from(basis);
    let mut count = basis.ncols();
    for axis in 0..n {
        if count == n {
            break;
        }
        let mut value = DVector::zeros(n);
        value[axis] = 1.;
        for _ in 0..2 {
            for j in 0..count {
                let column = frame.column(j);
                value -= column * column.dot(&value);
            }
        }
        let length = value.norm();
        if length > 1e-8 {
            frame.set_column(count, &(value / length));
            count += 1;
        }
    }
    ensure!(count == n, "관측 공간의 직교 완성을 확인하세요.");
    if basis.ncols() < n && frame.determinant() < 0. {
        frame.column_mut(n - 1).scale_mut(-1.);
    }
    Ok(frame)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReadoutEdit {
    Plane(ReadoutRotation),
    Joint(SubspaceRotation),
}
impl From<ReadoutRotation> for ReadoutEdit {
    fn from(p: ReadoutRotation) -> Self {
        Self::Plane(p)
    }
}
impl ReadoutEdit {
    pub fn validate(&self, width: usize) -> Result<()> {
        match self {
            Self::Plane(p) => p.validate(width),
            Self::Joint(q) => q.validate(width),
        }
    }
    pub fn subspace(&self) -> SubspaceRotation {
        match self {
            Self::Joint(q) => q.clone(),
            Self::Plane(p) => {
                let (a, b) = p.coefficients();
                SubspaceRotation {
                    basis: vec![p.u.clone(), p.v.clone()],
                    transform: vec![vec![1. + a, b], vec![-b, 1. + a]],
                }
            }
        }
    }
    pub fn angle_degrees(&self) -> f64 {
        match self {
            Self::Plane(p) => 2. * p.tangent.abs().atan().to_degrees(),
            Self::Joint(q) => q.angle_degrees(),
        }
    }
}

pub fn extend_basis(
    basis: &mut Vec<Vec<f64>>,
    value: &[f64],
    protected: &ProtectedSpace,
    maximum: usize,
) -> Result<()> {
    if basis.len() >= maximum {
        return Ok(());
    }
    let mut residual = protected.project(value)?;
    let reference = dot(&residual, &residual).sqrt();
    for _ in 0..2 {
        for u in basis.iter() {
            let projection = dot(u, &residual);
            for (x, y) in residual.iter_mut().zip(u) {
                *x -= projection * y;
            }
        }
    }
    let length = dot(&residual, &residual).sqrt();
    if length > 1e-9 * reference && length > 1e-12 {
        for x in &mut residual {
            *x /= length;
        }
        basis.push(residual);
    }
    Ok(())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseRecord {
    pub rank: usize,
    pub demand: Vec<f64>,
    pub pressure: Vec<f64>,
    pub predicted: Vec<f64>,
    pub residual: f64,
    pub generator_norm: f64,
    #[serde(default)]
    pub solve_dimension: usize,
}

enum ResponseKernel {
    Thin {
        left: DMatrix<f64>,
        right: DMatrix<f64>,
        symplectic: DMatrix<f64>,
        signed_gram: DMatrix<f64>,
        gram: DMatrix<f64>,
        cross_cost: DMatrix<f64>,
        square_cost: DMatrix<f64>,
        base_cost: f64,
    },
    Dense(DMatrix<f64>),
}

pub struct ResponsePath<'a> {
    base: &'a DMatrix<f64>,
    metric: &'a WeightMetric,
    kernel: ResponseKernel,
}
impl ResponsePath<'_> {
    pub fn trial(&self, step: f64) -> Result<(DMatrix<f64>, f64)> {
        ensure!(
            step.is_finite() && step >= 0.,
            "유한한 공동 수정 보폭을 지정하세요."
        );
        match &self.kernel {
            ResponseKernel::Thin {
                left,
                right,
                symplectic,
                signed_gram,
                gram,
                cross_cost,
                square_cost,
                base_cost,
            } => {
                let n = signed_gram.nrows();
                let t = (DMatrix::identity(n, n) - signed_gram * (step * 0.5))
                    .lu()
                    .solve(&(symplectic * step))
                    .ok_or_else(|| anyhow::anyhow!("축소 회전 연립방정식의 해를 확인하세요."))?;
                let rotation = self.base + left * &t * right;
                let cross = 2. * cross_cost.component_mul(&t.transpose()).sum();
                let square = t.component_mul(&(square_cost * &t * gram)).sum();
                let mut cost = base_cost + cross + square;
                let roundoff = 64. * f64::EPSILON * (base_cost + cross.abs() + square.abs());
                if !cost.is_finite() || cost <= roundoff {
                    cost = self.metric.norm_squared(&rotation);
                }
                Ok((rotation, cost))
            }
            ResponseKernel::Dense(generator) => {
                let eye = DMatrix::identity(generator.nrows(), generator.ncols());
                let local = (&eye - generator * (step * 0.5))
                    .lu()
                    .solve(&(&eye + generator * (step * 0.5)))
                    .ok_or_else(|| anyhow::anyhow!("공동 회전 연립방정식의 해를 확인하세요."))?;
                let rotation = self.base * local;
                let cost = self.metric.norm_squared(&rotation);
                Ok((rotation, cost))
            }
        }
    }
}

pub fn response<'a>(
    base: &'a DMatrix<f64>,
    metric: &'a WeightMetric,
    d: &[Vec<f64>],
    h: &[Vec<f64>],
    r: &[f64],
) -> Result<(ResponsePath<'a>, ResponseRecord)> {
    let m = r.len();
    ensure!(
        m > 0 && d.len() == m && h.len() == m,
        "관측과 개선 요구량의 수를 맞추세요."
    );
    let width = d[0].len();
    ensure!(
        width > 0
            && base.shape() == (width, width)
            && metric.gram.shape() == base.shape()
            && d.iter()
                .chain(h)
                .all(|v| v.len() == width && v.iter().all(|x| x.is_finite()))
            && r.iter().all(|x| x.is_finite()),
        "유한 응답 벡터의 차원을 맞추세요."
    );
    let gram = DMatrix::from_fn(m, m, |i, j| {
        dot(&d[i], &d[j]) * dot(&h[i], &h[j]) - dot(&d[i], &h[j]) * dot(&h[i], &d[j])
    });
    let scales: Vec<f64> = (0..m)
        .map(|i| gram[(i, i)].max(0.).sqrt().max(1e-30))
        .collect();
    let normalized = DMatrix::from_fn(m, m, |i, j| gram[(i, j)] / scales[i] / scales[j]);
    let eig = SymmetricEigen::new(normalized);
    let cutoff = eig.eigenvalues.max().max(0.) * 1e-11;
    let rhs = DVector::from_iterator(m, r.iter().zip(&scales).map(|(r, s)| r / s));
    let mut coordinates = eig.eigenvectors.transpose() * rhs;
    let mut rank = 0;
    for i in 0..m {
        if eig.eigenvalues[i] > cutoff && eig.eigenvalues[i] > 0. {
            coordinates[i] /= eig.eigenvalues[i];
            rank += 1;
        } else {
            coordinates[i] = 0.;
        }
    }
    let solution = eig.eigenvectors * coordinates;
    let pressure: Vec<f64> = (0..m).map(|i| solution[i] / scales[i]).collect();
    let predicted = &gram * DVector::from_vec(pressure.clone());
    let generator_norm = pressure
        .iter()
        .zip(predicted.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .max(0.)
        .sqrt();
    let solve_dimension = (2 * m).min(width);
    let kernel = if 2 * m < width {
        let factors = DMatrix::from_fn(width, 2 * m, |i, j| {
            let p = pressure[j % m];
            p.abs().sqrt()
                * if j < m {
                    d[j][i]
                } else {
                    p.signum() * h[j - m][i]
                }
        });
        let symplectic = DMatrix::from_fn(2 * m, 2 * m, |i, j| {
            if i < m && j == i + m {
                1.
            } else if j < m && i == j + m {
                -1.
            } else {
                0.
            }
        });
        let gram = factors.transpose() * &factors;
        let left = base * &factors;
        let weighted = &metric.gram * &left;
        let cross_cost = (&left - &factors).transpose() * &weighted;
        let square_cost = left.transpose() * weighted;
        ResponseKernel::Thin {
            left,
            right: factors.transpose(),
            signed_gram: &symplectic * &gram,
            symplectic,
            gram,
            cross_cost,
            square_cost,
            base_cost: metric.norm_squared(base),
        }
    } else {
        ResponseKernel::Dense(DMatrix::from_fn(width, width, |a, b| {
            (0..m)
                .map(|i| pressure[i] * (d[i][a] * h[i][b] - h[i][a] * d[i][b]))
                .sum()
        }))
    };
    let record = ResponseRecord {
        rank,
        demand: r.to_vec(),
        pressure,
        predicted: predicted.as_slice().to_vec(),
        residual: predicted
            .iter()
            .zip(r)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max),
        generator_norm,
        solve_dimension,
    };
    Ok((
        ResponsePath {
            base,
            metric,
            kernel,
        },
        record,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coupled_response_preserves_protected_axis_and_row_gram() {
        let h = vec![vec![1., 0., 0.], vec![0., 1., 0.]];
        let d = vec![vec![0., 1., 0.], vec![0., 0., 1.]];
        let identity = DMatrix::identity(3, 3);
        let metric = WeightMetric {
            gram: identity.clone(),
        };
        let (path, record) = response(&identity, &metric, &d, &h, &[0.2, 0.3]).unwrap();
        assert!(record.residual < 1e-13);
        let mut edit = SubspaceRotation::identity(vec![
            vec![1., 0., 0., 0.],
            vec![0., 1., 0., 0.],
            vec![0., 0., 1., 0.],
        ]);
        edit.set_matrix(&path.trial(1.).unwrap().0);
        edit.validate(4).unwrap();
        assert_eq!(edit.apply(&[0., 0., 0., 2.]), vec![0., 0., 0., 2.]);
        let a = [0.2, 1., -0.4, 0.7];
        let b = [2., -0.3, 0.8, 0.1];
        assert!((dot(&a, &b) - dot(&edit.apply(&a), &edit.apply(&b))).abs() < 1e-12);
        let mut corrupt = edit;
        corrupt.transform[0][0] += 0.1;
        assert!(corrupt.validate(4).is_err());
    }

    #[test]
    fn dependent_thin_response_matches_dense_rotation_and_weight_change() {
        let n = 8;
        let h: Vec<Vec<f64>> = (0..3)
            .map(|j| {
                (0..n)
                    .map(|i| ((i + 1 + 2 * (j % 2)) as f64).sin())
                    .collect()
            })
            .collect();
        let d: Vec<Vec<f64>> = (0..3)
            .map(|j| (0..n).map(|i| ((3 * i + 1 + j % 2) as f64).cos()).collect())
            .collect();
        let images: Vec<Vec<f32>> = (0..n)
            .map(|j| (0..11).map(|i| ((i * 3 + j * 7) as f32).sin()).collect())
            .collect();
        let metric = WeightMetric::from_images(&images).unwrap();
        let image = DMatrix::from_fn(11, n, |i, j| f64::from(images[j][i]));
        let identity = DMatrix::identity(n, n);
        let mut rotated = identity.clone();
        let (s, c) = 0.37f64.sin_cos();
        rotated[(0, 0)] = c;
        rotated[(0, 1)] = -s;
        rotated[(1, 0)] = s;
        rotated[(1, 1)] = c;
        for base in [&identity, &rotated] {
            let (path, record) = response(base, &metric, &d, &h, &[0.2, -0.13, 0.2]).unwrap();
            assert_eq!(record.solve_dimension, 6);
            assert_eq!(record.rank, 2);
            let omega = DMatrix::from_fn(n, n, |a, b| {
                (0..3)
                    .map(|i| record.pressure[i] * (d[i][a] * h[i][b] - h[i][a] * d[i][b]))
                    .sum()
            });
            for step in [0., 1e-8, 0.0625, 1.] {
                let direct = base
                    * (&identity - &omega * (step / 2.))
                        .lu()
                        .solve(&(&identity + &omega * (step / 2.)))
                        .unwrap();
                let (actual, cost) = path.trial(step).unwrap();
                assert!((&actual - &direct).amax() < 1e-13);
                let direct_cost = (&image * (&actual - &identity)).norm_squared();
                assert!(
                    (cost - direct_cost).abs() < 1e-12 * direct_cost + 1e-25,
                    "step={step} cost={cost:e} direct={direct_cost:e}"
                );
                assert!(
                    (metric.norm_squared(&actual) - direct_cost).abs()
                        < 1e-12 * direct_cost.max(1e-16)
                );
            }
        }
    }

    #[test]
    fn completion_preserves_observations_with_minimum_oriented_weight_cost() {
        let metric = WeightMetric {
            gram: DMatrix::from_diagonal(&DVector::from_vec(vec![1., 3., 5., 7.])),
        };
        let mut best = DMatrix::identity(4, 4);
        let (s, c) = 2.2f64.sin_cos();
        best[(0, 0)] = c;
        best[(0, 1)] = -s;
        best[(1, 0)] = s;
        best[(1, 1)] = c;
        let mut original = best.clone();
        let (s, c) = 0.7f64.sin_cos();
        original[(2, 2)] = c;
        original[(2, 3)] = -s;
        original[(3, 2)] = s;
        original[(3, 3)] = c;
        let observed =
            ObservedSpace::new(&[vec![1., 0., 0., 0.], vec![2., 0., 0., 0.]], 4).unwrap();
        assert_eq!(observed.rank, 1);
        let completed = observed.complete(&original, &metric).unwrap();
        assert!((completed.column(0) - original.column(0)).amax() < 1e-13);
        assert!((completed.transpose() * &completed - DMatrix::identity(4, 4)).amax() < 1e-13);
        assert!((completed.determinant() - 1.).abs() < 1e-13);
        assert!(metric.norm_squared(&completed) < metric.norm_squared(&original));
        assert!((metric.norm_squared(&completed) - metric.norm_squared(&best)).abs() < 1e-12);
        let unconstrained = ObservedSpace::new(&[], 4)
            .unwrap()
            .complete(&original, &metric)
            .unwrap();
        assert_eq!(metric.norm_squared(&unconstrained), 0.);
    }
}
