//! 열린 금지 구간을 합쳐 이차부등식의 공통 해와 최소 반각을 구한다.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BTreeMap};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Copy, Debug)]
struct Endpoint(f64);
impl PartialEq for Endpoint {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for Endpoint {}
impl PartialOrd for Endpoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Endpoint {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 == other.0 {
            Ordering::Equal
        } else {
            self.0.total_cmp(&other.0)
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Work {
    pub constraints: usize,
    pub screened: usize,
    pub square_roots: usize,
    pub inserted_intervals: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeasibleSummary {
    pub intervals: Vec<Interval>,
    #[serde(flatten)]
    pub work: Work,
}
#[derive(Clone, Debug)]
pub struct FeasibleSet {
    bound: f64,
    forbidden: BTreeMap<Endpoint, f64>,
    pub work: Work,
}
impl FeasibleSet {
    pub fn new(bound: f64) -> Result<Self> {
        ensure!(
            bound.is_finite() && bound > 0. && bound <= 1e6,
            "반각 범위는 0 초과 1e6 이하로 지정하세요."
        );
        Ok(Self {
            bound,
            forbidden: BTreeMap::new(),
            work: Work::default(),
        })
    }
    fn containing(&self, x: f64) -> Option<(f64, f64)> {
        self.forbidden
            .range(..Endpoint(x))
            .next_back()
            .and_then(|(a, &b)| (x < b).then_some((a.0, b)))
    }
    fn hull(&self) -> Option<(f64, f64)> {
        let left = self.containing(-self.bound).map_or(-self.bound, |(_, b)| b);
        let right = self.containing(self.bound).map_or(self.bound, |(a, _)| a);
        (left <= right).then_some((left, right))
    }
    pub fn is_empty(&self) -> bool {
        self.hull().is_none()
    }
    fn exclude(&mut self, mut a: f64, mut b: f64) {
        if a >= b || b <= -self.bound || a >= self.bound {
            return;
        }
        self.work.inserted_intervals += 1;
        if let Some((&key, &upper)) = self.forbidden.range(..=Endpoint(a)).next_back()
            && upper > a
        {
            a = key.0;
            b = b.max(upper);
            self.forbidden.remove(&key);
        }
        while let Some((&key, &upper)) = self.forbidden.range(Endpoint(a)..).next() {
            if key.0 >= b {
                break;
            }
            b = b.max(upper);
            self.forbidden.remove(&key);
        }
        self.forbidden.insert(Endpoint(a), b);
    }
    pub fn constrain(&mut self, a: f64, b: f64, c: f64) -> Result<()> {
        ensure!(
            [a, b, c].iter().all(|x| x.is_finite()),
            "유한 이차부등식 계수가 필요합니다."
        );
        self.work.constraints += 1;
        let Some((left, right)) = self.hull() else {
            return Ok(());
        };
        let magnitude = a.abs().max(b.abs()).max(c.abs());
        if magnitude == 0. {
            self.work.screened += 1;
            return Ok(());
        }
        // 2의 거듭제곱으로 나누면 이진 계수의 정확한 중근 관계가 유지된다.
        let exponent = (magnitude.to_bits() >> 52) & 0x7ff;
        let scale = f64::from_bits(exponent.max(1) << 52);
        let (a, b, c) = (a / scale, b / scale, c / scale);
        let certain = |t: f64| {
            let value = a.mul_add(t * t, b.mul_add(t, c));
            let error = 8. * f64::EPSILON * (a.abs() * t * t + b.abs() * t.abs() + c.abs());
            value >= error || (a == 0. && b == 0. && c >= 0.)
        };
        let vertex_ok = if a > 0. {
            let vertex = -b / (2. * a);
            vertex <= left || vertex >= right || 4. * a * c - b * b >= 0.
        } else {
            true
        };
        if certain(left) && certain(right) && vertex_ok {
            self.work.screened += 1;
            return Ok(());
        }
        let inf = f64::INFINITY;
        if a == 0. {
            if b > 0. {
                self.exclude(-inf, -c / b);
            } else if b < 0. {
                self.exclude(-c / b, inf);
            } else if c < 0. {
                self.exclude(-inf, inf);
            }
            return Ok(());
        }
        let discriminant = b.mul_add(b, -4. * a * c);
        if discriminant < 0. {
            if a < 0. {
                self.exclude(-inf, inf);
            }
        } else if discriminant == 0. {
            if a < 0. {
                let root = -b / (2. * a);
                self.exclude(-inf, root);
                self.exclude(root, inf);
            }
        } else {
            self.work.square_roots += 1;
            let (first, second) = if b == 0. {
                let root = (-c / a).sqrt();
                (-root, root)
            } else {
                let q = -0.5 * (b + discriminant.sqrt().copysign(b));
                (q / a, c / q)
            };
            let (lo, hi) = (first.min(second), first.max(second));
            if a > 0. {
                self.exclude(lo, hi);
            } else {
                self.exclude(-inf, lo);
                self.exclude(hi, inf);
            }
        }
        Ok(())
    }
    pub fn nearest(&self) -> Option<f64> {
        self.hull()?;
        match self.containing(0.) {
            None => Some(0.),
            Some((a, b)) => [a, b]
                .into_iter()
                .filter(|t| t.abs() <= self.bound)
                .min_by(|a, b| a.abs().total_cmp(&b.abs()).then(a.total_cmp(b))),
        }
    }
    pub fn intervals(&self) -> Vec<Interval> {
        let mut cursor = -self.bound;
        let mut result = Vec::new();
        for (a, &b) in &self.forbidden {
            if a.0 >= cursor {
                result.push(Interval {
                    lower: cursor,
                    upper: a.0.min(self.bound),
                });
            }
            cursor = cursor.max(b);
            if cursor > self.bound {
                return result;
            }
        }
        result.push(Interval {
            lower: cursor,
            upper: self.bound,
        });
        result
    }
    pub fn summary(&self) -> FeasibleSummary {
        FeasibleSummary {
            intervals: self.intervals(),
            work: self.work.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn touching_open_intervals_leave_singletons_and_symmetric_ties() {
        let mut f = FeasibleSet::new(3.).unwrap();
        f.constrain(1., 0., -2.).unwrap();
        f.constrain(-1., 0., 2.).unwrap();
        assert_eq!(
            f.intervals(),
            vec![
                Interval {
                    lower: -2f64.sqrt(),
                    upper: -2f64.sqrt()
                },
                Interval {
                    lower: 2f64.sqrt(),
                    upper: 2f64.sqrt()
                }
            ]
        );
        assert_eq!(f.nearest(), Some(-2f64.sqrt()));
        let mut double = FeasibleSet::new(1.).unwrap();
        double.constrain(-9000., 600., -10.).unwrap();
        assert_eq!(double.nearest(), Some(1. / 30.));
        double.constrain(0., 1., -0.1).unwrap();
        assert!(double.is_empty());
    }
    #[test]
    fn boundary_holes_and_screening_preserve_the_domain() {
        let mut f = FeasibleSet::new(1.).unwrap();
        f.constrain(1., 0., -1.).unwrap();
        assert_eq!(
            f.intervals(),
            vec![
                Interval {
                    lower: -1.,
                    upper: -1.
                },
                Interval {
                    lower: 1.,
                    upper: 1.
                }
            ]
        );
        f.constrain(0., 1., 0.).unwrap();
        assert_eq!(f.nearest(), Some(1.));
        f.constrain(1., 0., 1.).unwrap();
        assert_eq!(f.work.screened, 1);
        f.constrain(0., -1., 0.5).unwrap();
        assert!(f.is_empty());
    }
}
