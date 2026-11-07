//! 가중치에서 만든 수치 신호의 구별도와 상쇄 좌표를 추적한다.
use crate::{
    contrast::{self, Distribution, PairMetric, ProbabilityPair, difference, doubles, length},
    diagnosis::{BudgetExpired, Deadline, Proposal, SeparationObjective, save},
    gpu::Gpu,
    graph::NormKind,
    hub::ModelFiles,
    model::{DecoderTrace, DeviceRotation, Frame, Model, SparseIntervention, WeightDelta},
    response::ReadoutEdit,
    rotation::{ProtectedSpace, dot},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct Options {
    pub directions: usize,
    pub prefix_length: usize,
    pub seed: u64,
    pub probe_angle: f64,
    pub max_angle_degrees: f64,
    pub max_edit_norm: Option<f64>,
    pub tolerance: f64,
    pub coordinates: usize,
    pub intervention_fraction: f64,
    pub time_budget: Duration,
    pub inputs: Option<PathBuf>,
    pub proposal: Option<PathBuf>,
    pub output: PathBuf,
}
impl Options {
    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=128).contains(&self.directions) && self.prefix_length <= 4096,
            "방향 수는 1..=128, 접두사 길이는 0..=4096으로 지정하세요."
        );
        ensure!(
            self.probe_angle.is_finite()
                && self.probe_angle > 0.
                && self.probe_angle < std::f64::consts::FRAC_PI_2,
            "주입 각도는 0과 π/2 사이의 라디안으로 지정하세요."
        );
        ensure!(
            self.max_angle_degrees.is_finite()
                && self.max_angle_degrees > 0.
                && self.max_angle_degrees < 179.
                && self.max_edit_norm.is_none_or(|v| v.is_finite() && v > 0.)
                && self.tolerance.is_finite()
                && self.tolerance > 0.,
            "회전 각도·수정 노름·허용 오차의 양의 범위를 확인하세요."
        );
        ensure!(
            self.coordinates <= 128
                && self.intervention_fraction.is_finite()
                && (0. ..=1.).contains(&self.intervention_fraction),
            "좌표 수는 0..=128, 개입 비율은 0..=1로 지정하세요."
        );
        ensure!(
            self.proposal.is_none() || self.inputs.is_some(),
            "수정안 재실행에는 저장한 주입 입력을 지정하세요."
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inputs {
    pub version: u32,
    pub seed: Option<u64>,
    pub embedding_rows: Vec<[u32; 2]>,
    pub prefix: Vec<Vec<f32>>,
    pub anchor: Vec<f32>,
    pub basis: Vec<Vec<f64>>,
    pub probe_angle: f64,
    #[serde(default)]
    pub selected_direction: Option<Vec<f64>>,
    #[serde(default)]
    pub controls: Vec<Vec<f32>>,
    #[serde(default)]
    pub protected_hidden: Vec<Vec<f32>>,
}
impl Inputs {
    fn validate(&self, width: usize) -> Result<()> {
        ensure!(
            self.version == 1
                && self.prefix.len() <= 4096
                && (1..=128).contains(&self.basis.len())
                && self.basis.len() < width
                && self.protected_hidden.len() <= 4096
                && self.controls.len() <= 64,
            "주입 입력의 version·위치·기저 크기를 확인하세요."
        );
        ensure!(
            self.probe_angle.is_finite()
                && self.probe_angle > 0.
                && self.probe_angle < std::f64::consts::FRAC_PI_2,
            "유한한 구면 주입 각도를 지정하세요."
        );
        for x in self
            .prefix
            .iter()
            .chain(std::iter::once(&self.anchor))
            .chain(&self.protected_hidden)
            .chain(&self.controls)
        {
            ensure!(
                x.len() == width && x.iter().all(|v| v.is_finite()),
                "주입·보호 벡터의 폭과 유한값을 확인하세요."
            );
        }
        let center = doubles(&self.anchor);
        let radius = length(&center);
        ensure!(
            radius > 0. && radius.is_finite(),
            "양의 길이를 가진 기준 벡터를 지정하세요."
        );
        for (i, axis) in self.basis.iter().enumerate() {
            self.validate_axis(axis, &center, radius)?;
            for previous in &self.basis[..i] {
                ensure!(
                    dot(axis, previous).abs() < 1e-7,
                    "주입 기저의 직교 조건을 확인하세요."
                );
            }
        }
        if let Some(axis) = &self.selected_direction {
            self.validate_axis(axis, &center, radius)?;
        }
        Ok(())
    }
    fn validate_axis(&self, axis: &[f64], anchor: &[f64], radius: f64) -> Result<()> {
        ensure!(
            axis.len() == anchor.len()
                && axis.iter().all(|v| v.is_finite())
                && (dot(axis, axis) - 1.).abs() < 1e-7
                && dot(axis, anchor).abs() < 1e-7 * radius,
            "기준 벡터에 직교하는 단위 주입 방향을 지정하세요."
        );
        Ok(())
    }
    fn pair(&self, direction: &[f64]) -> [Vec<f32>; 2] {
        let radius = length(&doubles(&self.anchor));
        [1., -1.].map(|sign| {
            self.anchor
                .iter()
                .zip(direction)
                .map(|(x, v)| {
                    (self.probe_angle.cos() * f64::from(*x)
                        + sign * self.probe_angle.sin() * radius * v) as f32
                })
                .collect()
        })
    }
}

struct RowSequence {
    state: u64,
    count: u64,
}
impl RowSequence {
    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) % self.count) as u32
    }
    fn pair(&mut self) -> [u32; 2] {
        let first = self.next();
        let mut second = self.next();
        if first == second {
            second = ((u64::from(second) + 1) % self.count) as u32;
        }
        [first, second]
    }
}
fn generated_inputs(
    model: &Model,
    gpu: &mut Gpu,
    options: &Options,
    deadline: &Deadline,
) -> Result<Inputs> {
    let width = model.graph().hidden_size;
    ensure!(
        width > 1,
        "수치 주입에는 두 차원 이상의 hidden이 필요합니다."
    );
    let mut rows = RowSequence {
        state: options.seed,
        count: model.vocab_size() as u64,
    };
    let mut origins = Vec::new();
    let mut mix = |subtract: bool| -> Result<Vec<f64>> {
        deadline.check()?;
        let pair = rows.pair();
        origins.push(pair);
        let a = model.embedding_vector(gpu, pair[0])?;
        let b = model.embedding_vector(gpu, pair[1])?;
        Ok(a.iter()
            .zip(b)
            .map(|(a, b)| {
                (f64::from(*a)
                    + if subtract {
                        -f64::from(b)
                    } else {
                        f64::from(b)
                    })
                    / 2f64.sqrt()
            })
            .collect())
    };
    let mut prefix = Vec::new();
    for _ in 0..options.prefix_length {
        prefix.push(mix(false)?.into_iter().map(|v| v as f32).collect());
    }
    let mut anchor = Vec::new();
    for _ in 0..64 {
        let candidate: Vec<f32> = mix(false)?.into_iter().map(|v| v as f32).collect();
        if length(&doubles(&candidate)) > 1e-12 {
            anchor = candidate;
            break;
        }
    }
    ensure!(
        !anchor.is_empty(),
        "embedding 행 혼합에서 양의 길이를 가진 기준 상태를 확인하세요."
    );
    let mut span = ProtectedSpace::new(width);
    span.include(&doubles(&anchor))?;
    let target = options.directions.min(width - 1);
    let mut basis = Vec::new();
    for _ in 0..target * 32 {
        let mut axis = span.project(&mix(true)?)?;
        let norm = length(&axis);
        if norm > 1e-10 {
            axis.iter_mut().for_each(|v| *v /= norm);
            span.include(&axis)?;
            basis.push(axis);
        }
        if basis.len() == target {
            break;
        }
    }
    ensure!(
        !basis.is_empty(),
        "embedding 차이에서 주입 가능한 접선 방향을 확인하세요."
    );
    Ok(Inputs {
        version: 1,
        seed: Some(options.seed),
        embedding_rows: origins,
        prefix,
        anchor,
        basis,
        probe_angle: options.probe_angle,
        selected_direction: None,
        controls: Vec::new(),
        protected_hidden: Vec::new(),
    })
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Computation {
    pub observation_decoder_steps: usize,
    pub intervention_decoder_steps: usize,
    pub replay_decoder_steps: usize,
    pub candidate_decoder_steps: usize,
    pub candidate_readouts: usize,
    pub protection_readouts: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectionResult {
    pub index: usize,
    pub metric: PairMetric,
    pub hidden_difference_norm: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Coordinate {
    pub row: usize,
    pub column: usize,
    pub weight: f64,
    pub signed_work: f64,
    pub head: Option<usize>,
}
impl PartialEq for Coordinate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Coordinate {}
impl PartialOrd for Coordinate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Coordinate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.signed_work
            .total_cmp(&other.signed_work)
            .then(self.row.cmp(&other.row))
            .then(self.column.cmp(&other.column))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BranchWork {
    pub layer: usize,
    pub tensor: String,
    pub signed_work: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoordinateWork {
    pub branch: usize,
    pub tensor: String,
    pub sum: f64,
    pub reconstruction_residual: f64,
    pub leading_negative: Vec<Coordinate>,
    pub remaining_credit: f64,
    pub head_credit: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transport {
    pub input_work: f64,
    pub branches: Vec<BranchWork>,
    pub sum: f64,
    pub observed_jeffreys: f64,
    pub absolute_residual: f64,
    pub coordinates: Option<CoordinateWork>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterventionResult {
    pub tensor: String,
    pub relative_change: f64,
    pub entries: Vec<WeightDelta>,
    pub delta_norm_squared: f64,
    pub metric: PairMetric,
    pub prefix_recomputed: usize,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Protection {
    pub maximum_absolute_change: f64,
    pub maximum_scaled_change: f64,
}
impl Protection {
    fn measure(&mut self, before: &[f32], after: &[f32]) {
        for (a, b) in before.iter().zip(after) {
            let change = (f64::from(*a) - f64::from(*b)).abs();
            self.maximum_absolute_change = self.maximum_absolute_change.max(change);
            self.maximum_scaled_change = self
                .maximum_scaled_change
                .max(change / (1. + f64::from(*a).abs()));
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub angle_degrees: f64,
    pub weight_delta_norm_squared: f64,
    pub metric: PairMetric,
    pub protection: Protection,
    pub midpoint: Protection,
    pub accepted: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub version: u32,
    pub phase: String,
    pub outcome: String,
    pub stop_reason: Option<String>,
    pub source: ModelFiles,
    pub device: Value,
    pub parameters: Value,
    pub input_sha256: Option<String>,
    pub scan: Vec<DirectionResult>,
    pub eigenvalues: Vec<f64>,
    pub selected_by: Option<String>,
    pub baseline: Option<PairMetric>,
    pub transport: Option<Transport>,
    pub interventions: Vec<InterventionResult>,
    pub rotation_direction: Option<String>,
    pub angular_derivative: Option<f64>,
    pub protected_rank: usize,
    pub candidates: Vec<Candidate>,
    pub selected: Option<Proposal>,
    pub replay: Option<Value>,
    pub computation: Computation,
    pub elapsed_seconds: f64,
}

fn prefill(
    model: &mut Model,
    gpu: &mut Gpu,
    inputs: &Inputs,
    edit: Option<&DeviceRotation>,
    intervention: Option<&SparseIntervention>,
    deadline: &Deadline,
    count: &mut usize,
) -> Result<()> {
    model.reset();
    for vector in &inputs.prefix {
        deadline.check()?;
        model.inject(gpu, vector, edit, intervention, None)?;
        *count += 1;
    }
    Ok(())
}
fn observe_pair(
    model: &mut Model,
    gpu: &mut Gpu,
    inputs: &Inputs,
    direction: &[f64],
    edit: Option<&DeviceRotation>,
    intervention: Option<&SparseIntervention>,
    trace: bool,
    deadline: &Deadline,
    count: &mut usize,
) -> Result<([Frame; 2], [DecoderTrace; 2])> {
    let vectors = inputs.pair(direction);
    let mut traces = [DecoderTrace::default(), DecoderTrace::default()];
    let mut frames = Vec::new();
    for i in 0..2 {
        deadline.check()?;
        model.rewind(inputs.prefix.len())?;
        let frame = model.inject(
            gpu,
            &vectors[i],
            edit,
            intervention,
            if trace { Some(&mut traces[i]) } else { None },
        )?;
        *count += 1;
        frames.push(frame);
    }
    Ok((frames.try_into().unwrap(), traces))
}

fn trace_transport(
    model: &Model,
    inputs: &Inputs,
    direction: &[f64],
    frames: &[Frame; 2],
    traces: &[DecoderTrace; 2],
    count: usize,
    deadline: &Deadline,
) -> Result<Transport> {
    let pair = ProbabilityPair::new(&frames[0].logits, &frames[1].logits)?;
    let output = model.readout_transpose(&pair.delta_probability)?;
    let norm = &model.graph().final_norm;
    let adjoint = contrast::norm_secant_transpose(
        &traces[0].residual,
        &traces[1].residual,
        &model.final_gain()?,
        f64::from(norm.epsilon),
        norm.kind == NormKind::Layer,
        &output,
    )?;
    let vectors = inputs.pair(direction);
    let input_work = dot(
        &adjoint,
        &difference(&doubles(&vectors[0]), &doubles(&vectors[1])),
    );
    let branches: Vec<_> = traces[0]
        .branches
        .iter()
        .zip(&traces[1].branches)
        .map(|(a, b)| BranchWork {
            layer: a.layer,
            tensor: a.tensor.clone(),
            signed_work: dot(
                &adjoint,
                &difference(&doubles(&a.output), &doubles(&b.output)),
            ),
        })
        .collect();
    let sum = input_work + branches.iter().map(|b| b.signed_work).sum::<f64>();
    let mut transport = Transport {
        input_work,
        branches,
        sum,
        observed_jeffreys: pair.metric.jeffreys,
        absolute_residual: (sum - pair.metric.jeffreys).abs(),
        coordinates: None,
    };
    if count == 0 {
        return Ok(transport);
    }
    let Some((index, branch)) = transport
        .branches
        .iter()
        .enumerate()
        .filter(|(_, b)| b.signed_work < 0.)
        .min_by(|(_, a), (_, b)| a.signed_work.total_cmp(&b.signed_work))
    else {
        return Ok(transport);
    };
    let observation = &traces[0].branches[index];
    let features = difference(
        &doubles(&observation.input),
        &doubles(&traces[1].branches[index].input),
    );
    let matrix = model.matrix(&branch.tensor)?;
    let mut leading = BinaryHeap::<Coordinate>::new();
    let mut total = 0.;
    let mut heads = observation
        .head_width
        .map(|width| vec![0.; matrix.cols / width])
        .unwrap_or_default();
    for (flat, weight) in matrix.values().enumerate() {
        let (row, column) = (flat / matrix.cols, flat % matrix.cols);
        if column == 0 && row % 64 == 0 {
            deadline.check()?;
        }
        let work = adjoint[row] * weight * features[column];
        total += work;
        let head = observation.head_width.map(|width| column / width);
        if let Some(head) = head {
            heads[head] += work;
        }
        if work < 0. && (leading.len() < count || work < leading.peek().unwrap().signed_work) {
            if leading.len() == count {
                leading.pop();
            }
            leading.push(Coordinate {
                row,
                column,
                weight,
                signed_work: work,
                head,
            });
        }
    }
    let mut coordinates = leading.into_vec();
    coordinates.sort();
    let remaining_credit = total - coordinates.iter().map(|c| c.signed_work).sum::<f64>();
    transport.coordinates = Some(CoordinateWork {
        branch: index,
        tensor: branch.tensor.clone(),
        sum: total,
        reconstruction_residual: total - branch.signed_work,
        leading_negative: coordinates,
        remaining_credit,
        head_credit: heads,
    });
    Ok(transport)
}

fn protected_scores(
    model: &Model,
    gpu: &mut Gpu,
    inputs: &Inputs,
    deadline: &Deadline,
    count: &mut usize,
) -> Result<Vec<Vec<f32>>> {
    let mut scores = Vec::new();
    for h in &inputs.protected_hidden {
        deadline.check()?;
        scores.push(model.readout(gpu, h, None)?);
        *count += 1;
    }
    Ok(scores)
}
fn protection(
    model: &Model,
    gpu: &mut Gpu,
    inputs: &Inputs,
    before: &[Vec<f32>],
    rotation: &DeviceRotation,
    deadline: &Deadline,
    count: &mut usize,
) -> Result<Protection> {
    let mut result = Protection::default();
    for (hidden, original) in inputs.protected_hidden.iter().zip(before) {
        deadline.check()?;
        let changed = model.readout(gpu, hidden, Some(rotation))?;
        *count += 1;
        result.measure(original, &changed);
    }
    Ok(result)
}
fn midpoint_change(before: &[Frame; 2], after: &[Vec<f32>; 2]) -> Protection {
    let mut result = Protection::default();
    for i in 0..before[0].logits.len() {
        let original = 0.5 * (f64::from(before[0].logits[i]) + f64::from(before[1].logits[i]));
        let changed = 0.5 * (f64::from(after[0][i]) + f64::from(after[1][i]));
        let error = (changed - original).abs();
        result.maximum_absolute_change = result.maximum_absolute_change.max(error);
        result.maximum_scaled_change = result
            .maximum_scaled_change
            .max(error / (1. + original.abs()));
    }
    result
}
fn input_digest(path: &Path) -> Result<String> {
    Ok(hex::encode(Sha256::digest(std::fs::read(path)?)))
}

fn repair(
    model: &Model,
    gpu: &mut Gpu,
    files: &ModelFiles,
    inputs: &Inputs,
    frames: &[Frame; 2],
    options: &Options,
    report: &mut Report,
    deadline: &Deadline,
) -> Result<()> {
    let width = model.graph().hidden_size;
    let pair = ProbabilityPair::new(&frames[0].logits, &frames[1].logits)?;
    let delta = difference(&doubles(&frames[0].hidden), &doubles(&frames[1].hidden));
    let common: Vec<_> = frames[0]
        .hidden
        .iter()
        .zip(&frames[1].hidden)
        .map(|(a, b)| 0.5 * (f64::from(*a) + f64::from(*b)))
        .collect();
    let mut protected = ProtectedSpace::new(width);
    protected.include(&common)?;
    for h in &inputs.protected_hidden {
        protected.include(&doubles(h))?;
    }
    report.protected_rank = protected.rank();
    deadline.check()?;
    let gradient = model.readout_transpose(&pair.separation_gradient())?;
    let mut direction = contrast::separating_plane(&protected, &delta, &gradient)?;
    report.rotation_direction = Some("analytic_jeffreys_gradient".into());
    if direction.is_none() {
        // 구별도의 일차 응답이 0일 때 출력 행 차이로 유한 후보를 구성한다.
        let mut rows: Vec<_> = (0..model.vocab_size()).collect();
        rows.sort_unstable_by(|a, b| {
            pair.plus.probability[*b]
                .total_cmp(&pair.plus.probability[*a])
                .then(a.cmp(b))
        });
        let reference = doubles(&model.head_row(gpu, rows[0] as u32)?);
        for index in rows.into_iter().skip(1).take(32) {
            deadline.check()?;
            let row = doubles(&model.head_row(gpu, index as u32)?);
            if let Some((plane, _)) =
                contrast::separating_plane(&protected, &delta, &difference(&row, &reference))?
            {
                let slope = dot(&gradient, &plane.u) * dot(&plane.v, &delta)
                    - dot(&gradient, &plane.v) * dot(&plane.u, &delta);
                direction = Some((plane, slope));
                report.rotation_direction = Some("output_row_contrast_seed".into());
                break;
            }
        }
    }
    let Some((mut plane, derivative)) = direction else {
        report.outcome = "protected_search_finished".into();
        return Ok(());
    };
    report.angular_derivative = Some(derivative);
    let wu = model.head_image(gpu, &plane.u)?;
    let wv = model.head_image(gpu, &plane.v)?;
    let energy = wu
        .iter()
        .chain(&wv)
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>();
    let mut angle = options.max_angle_degrees.to_radians();
    if let Some(limit) = options.max_edit_norm {
        if energy > 0. {
            angle = angle.min(2. * (limit / (2. * energy.sqrt())).min(1.).asin());
        }
    }
    let original = protected_scores(
        model,
        gpu,
        inputs,
        deadline,
        &mut report.computation.protection_readouts,
    )?;
    let mut best = pair.metric.jeffreys;
    let mut best_edit = None;
    for fraction in [0.04, 0.2, 1.] {
        deadline.check()?;
        let theta = fraction * angle;
        plane.tangent = (theta * 0.5).tan();
        let edit = ReadoutEdit::Plane(plane.clone());
        let state = DeviceRotation::upload_edit(gpu, &edit, width)?;
        let changed = [
            model.readout(gpu, &frames[0].hidden, Some(&state))?,
            model.readout(gpu, &frames[1].hidden, Some(&state))?,
        ];
        report.computation.candidate_readouts += 2;
        let measured = ProbabilityPair::new(&changed[0], &changed[1])?.metric;
        let preservation = protection(
            model,
            gpu,
            inputs,
            &original,
            &state,
            deadline,
            &mut report.computation.protection_readouts,
        )?;
        let midpoint = midpoint_change(frames, &changed);
        let cost = plane.norm_squared(&wu, &wv);
        let accepted = measured.jeffreys
            > pair.metric.jeffreys + options.tolerance * pair.metric.jeffreys.max(1e-12)
            && preservation.maximum_scaled_change <= options.tolerance
            && midpoint.maximum_scaled_change <= options.tolerance
            && options
                .max_edit_norm
                .is_none_or(|cap| cost <= cap * cap * (1. + 1e-12));
        if accepted && measured.jeffreys > best {
            best = measured.jeffreys;
            best_edit = Some(edit);
        }
        report.candidates.push(Candidate {
            angle_degrees: theta.to_degrees(),
            weight_delta_norm_squared: cost,
            metric: measured,
            protection: preservation,
            midpoint,
            accepted,
        });
    }
    if let Some(edit) = best_edit {
        let objective = SeparationObjective {
            metric: "jeffreys".into(),
            before: pair.metric.jeffreys,
            after: best,
            probe_input_sha256: report.input_sha256.clone().context("주입 입력 해시")?,
        };
        let proposal = Proposal::for_separation(files, model, edit, objective);
        proposal.validate(files, model)?;
        save(&options.output.join("proposal.json"), &proposal)?;
        report.selected = Some(proposal);
        report.outcome = "verified_separation_rotation".into();
    } else {
        report.outcome = "candidate_search_finished".into();
    }
    Ok(())
}

fn replay(
    model: &mut Model,
    gpu: &mut Gpu,
    files: &ModelFiles,
    inputs: &Inputs,
    options: &Options,
    report: &mut Report,
    deadline: &Deadline,
) -> Result<()> {
    let proposal: Proposal =
        serde_json::from_slice(&std::fs::read(options.proposal.as_ref().unwrap())?)?;
    proposal.validate(files, model)?;
    let objective = proposal
        .separation
        .as_ref()
        .context("수치 주입 목적의 수정안을 지정하세요.")?;
    ensure!(
        Some(&objective.probe_input_sha256) == report.input_sha256.as_ref(),
        "수정안을 계산한 주입 입력 SHA256을 맞추세요."
    );
    let direction = inputs
        .selected_direction
        .as_ref()
        .context("저장한 선택 방향")?;
    prefill(
        model,
        gpu,
        inputs,
        None,
        None,
        deadline,
        &mut report.computation.replay_decoder_steps,
    )?;
    let (baseline, _) = observe_pair(
        model,
        gpu,
        inputs,
        direction,
        None,
        None,
        false,
        deadline,
        &mut report.computation.replay_decoder_steps,
    )?;
    let device = DeviceRotation::upload_edit(gpu, &proposal.rotation, model.graph().hidden_size)?;
    prefill(
        model,
        gpu,
        inputs,
        Some(&device),
        None,
        deadline,
        &mut report.computation.replay_decoder_steps,
    )?;
    let (changed, _) = observe_pair(
        model,
        gpu,
        inputs,
        direction,
        Some(&device),
        None,
        false,
        deadline,
        &mut report.computation.replay_decoder_steps,
    )?;
    let before = ProbabilityPair::new(&baseline[0].logits, &baseline[1].logits)?.metric;
    let after = ProbabilityPair::new(&changed[0].logits, &changed[1].logits)?.metric;
    let original = protected_scores(
        model,
        gpu,
        inputs,
        deadline,
        &mut report.computation.protection_readouts,
    )?;
    let preserved = protection(
        model,
        gpu,
        inputs,
        &original,
        &device,
        deadline,
        &mut report.computation.protection_readouts,
    )?;
    let midpoint = midpoint_change(
        &baseline,
        &[changed[0].logits.clone(), changed[1].logits.clone()],
    );
    let accepted = after.jeffreys > before.jeffreys
        && preserved.maximum_scaled_change <= options.tolerance
        && midpoint.maximum_scaled_change <= options.tolerance;
    save(
        &options.output.join("replay-frames.json"),
        &json!({"baseline":baseline,"changed":changed}),
    )?;
    report.baseline = Some(before.clone());
    report.replay = Some(
        json!({"before":before,"after":after,"protection":preserved,"midpoint":midpoint,
        "accepted":accepted,"saved_after":objective.after,"decoder":"full_forward_from_saved_numeric_inputs"}),
    );
    report.selected = Some(proposal);
    report.outcome = if accepted {
        "replayed_separation_rotation"
    } else {
        "replay_measured"
    }
    .into();
    Ok(())
}

pub fn run(files: ModelFiles, cache: &Path, python: &Path, options: Options) -> Result<Report> {
    options.validate()?;
    ensure!(
        !options.output.join("report.json").exists()
            && !options.output.join("inputs.json").exists(),
        "새 결과 폴더를 지정하세요."
    );
    let started = Instant::now();
    let deadline = Deadline::after(options.time_budget)?;
    let mut report = Report {
        version: 1,
        phase: "loading".into(),
        outcome: "pending".into(),
        stop_reason: None,
        source: files.clone(),
        device: Value::Null,
        parameters: json!({"directions":options.directions,"prefix_length":options.prefix_length,"seed":options.seed,
            "probe_angle_radians":options.probe_angle,"max_angle_degrees":options.max_angle_degrees,"max_edit_norm":options.max_edit_norm,
            "tolerance":options.tolerance,"coordinates":options.coordinates,"intervention_fraction":options.intervention_fraction,
            "time_budget_seconds":options.time_budget.as_secs_f64(),"input_path":options.inputs,"proposal_path":options.proposal,
            "metric":"Jeffreys divergence of injected output distributions; temperature 1","edit_scope":"virtual_readout",
            "input_scope":"continuous vectors at decoder entrance after embedding scale","precision":"native F32 decoder and rotation; original F32/F16/BF16 weights; F64 algebra"}),
        input_sha256: None,
        scan: Vec::new(),
        eigenvalues: Vec::new(),
        selected_by: None,
        baseline: None,
        transport: None,
        interventions: Vec::new(),
        rotation_direction: None,
        angular_derivative: None,
        protected_rank: 0,
        candidates: Vec::new(),
        selected: None,
        replay: None,
        computation: Computation::default(),
        elapsed_seconds: 0.,
    };
    save(&options.output.join("manifest.json"), &report)?;
    let calculation = (|| -> Result<()> {
        let provided = options
            .inputs
            .as_ref()
            .map(|path| -> Result<Inputs> { Ok(serde_json::from_slice(&std::fs::read(path)?)?) })
            .transpose()?;
        let capacity = provided
            .as_ref()
            .map_or(options.prefix_length, |i| i.prefix.len())
            .checked_add(1)
            .context("주입 context")?;
        ensure!(capacity <= 4097, "수치 접두사는 4096개 이하로 지정하세요.");
        let mut gpu = Gpu::new(cache, python)?;
        let mut model = Model::load(&files.root, &mut gpu, capacity)?;
        report.device =
            json!({"backend":gpu.backend(),"name":gpu.name(),"architecture":gpu.architecture()});
        report.parameters["weight_storage"] = json!(model.weight_storage());
        report.parameters["stored_parameters"] = json!(model.parameter_count());
        deadline.check()?;
        let mut inputs = match provided {
            Some(i) => i,
            None => generated_inputs(&model, &mut gpu, &options, &deadline)?,
        };
        inputs.validate(model.graph().hidden_size)?;
        report.parameters["effective_directions"] = json!(inputs.basis.len());
        report.parameters["effective_prefix_length"] = json!(inputs.prefix.len());
        report.parameters["effective_probe_angle_radians"] = json!(inputs.probe_angle);
        save(&options.output.join("manifest.json"), &report)?;
        if options.proposal.is_some() {
            report.input_sha256 = Some(input_digest(options.inputs.as_ref().unwrap())?);
            save(&options.output.join("inputs.json"), &inputs)?;
            return replay(
                &mut model,
                &mut gpu,
                &files,
                &inputs,
                &options,
                &mut report,
                &deadline,
            );
        }
        report.phase = "probe_scan".into();
        save(&options.output.join("inputs.json"), &inputs)?;
        prefill(
            &mut model,
            &mut gpu,
            &inputs,
            None,
            None,
            &deadline,
            &mut report.computation.observation_decoder_steps,
        )?;
        deadline.check()?;
        let baseline = model.inject(&mut gpu, &inputs.anchor, None, None, None)?;
        report.computation.observation_decoder_steps += 1;
        let distribution = Distribution::new(&baseline.logits)?;
        let mut responses = Vec::new();
        for (index, axis) in inputs.basis.iter().enumerate() {
            let (frames, _) = observe_pair(
                &mut model,
                &mut gpu,
                &inputs,
                axis,
                None,
                None,
                false,
                &deadline,
                &mut report.computation.observation_decoder_steps,
            )?;
            let pair = ProbabilityPair::new(&frames[0].logits, &frames[1].logits)?;
            responses.push(pair.delta_logits);
            report.scan.push(DirectionResult {
                index,
                metric: pair.metric,
                hidden_difference_norm: length(&difference(
                    &doubles(&frames[0].hidden),
                    &doubles(&frames[1].hidden),
                )),
            });
        }
        let (coefficients, eigenvalues) = contrast::weak_combination(&responses, &distribution)?;
        report.eigenvalues = eigenvalues;
        let mut direction = vec![0.; model.graph().hidden_size];
        for (axis, coefficient) in inputs.basis.iter().zip(&coefficients) {
            for (value, component) in direction.iter_mut().zip(axis) {
                *value += coefficient * component;
            }
        }
        let radius = length(&direction);
        direction.iter_mut().for_each(|v| *v /= radius);
        let (mut frames, mut traces) = observe_pair(
            &mut model,
            &mut gpu,
            &inputs,
            &direction,
            None,
            None,
            true,
            &deadline,
            &mut report.computation.observation_decoder_steps,
        )?;
        let best = report
            .scan
            .iter()
            .min_by(|a, b| a.metric.jeffreys.total_cmp(&b.metric.jeffreys))
            .unwrap();
        let spectral = ProbabilityPair::new(&frames[0].logits, &frames[1].logits)?.metric;
        report.parameters["spectral_prediction"] =
            json!({"coefficients":coefficients,"measured":spectral});
        if spectral.jeffreys > best.metric.jeffreys {
            direction = inputs.basis[best.index].clone();
            (frames, traces) = observe_pair(
                &mut model,
                &mut gpu,
                &inputs,
                &direction,
                None,
                None,
                true,
                &deadline,
                &mut report.computation.observation_decoder_steps,
            )?;
            report.selected_by = Some("measured_basis_direction".into());
        } else {
            report.selected_by = Some("measured_spectral_combination".into());
        }
        report.baseline = Some(ProbabilityPair::new(&frames[0].logits, &frames[1].logits)?.metric);
        inputs.selected_direction = Some(direction.clone());
        inputs.protected_hidden.push(baseline.hidden);
        let mut control = inputs
            .basis
            .iter()
            .min_by(|a, b| {
                dot(a, &direction)
                    .abs()
                    .total_cmp(&dot(b, &direction).abs())
            })
            .unwrap()
            .clone();
        let projection = dot(&control, &direction);
        for (c, d) in control.iter_mut().zip(&direction) {
            *c -= projection * d;
        }
        let control_length = length(&control);
        if control_length > 1e-8 {
            control.iter_mut().for_each(|v| *v /= control_length);
            inputs.controls.extend(inputs.pair(&control));
        }
        for vector in &inputs.controls {
            deadline.check()?;
            model.rewind(inputs.prefix.len())?;
            let control = model.inject(&mut gpu, vector, None, None, None)?;
            report.computation.observation_decoder_steps += 1;
            inputs.protected_hidden.push(control.hidden);
        }
        save(&options.output.join("inputs.json"), &inputs)?;
        report.input_sha256 = Some(input_digest(&options.output.join("inputs.json"))?);
        save(
            &options.output.join("trace.json"),
            &json!({"frames":frames,"traces":traces,"pair_inputs":inputs.pair(&direction)}),
        )?;
        report.phase = "signed_transport".into();
        report.transport = Some(trace_transport(
            &model,
            &inputs,
            &direction,
            &frames,
            &traces,
            options.coordinates,
            &deadline,
        )?);
        save(&options.output.join("transport.json"), &report.transport)?;
        report.phase = "protected_rotation".into();
        repair(
            &model,
            &mut gpu,
            &files,
            &inputs,
            &frames,
            &options,
            &mut report,
            &deadline,
        )?;
        if options.intervention_fraction > 0. {
            if let Some(coordinates) = report
                .transport
                .as_ref()
                .and_then(|t| t.coordinates.as_ref())
                .filter(|c| !c.leading_negative.is_empty())
            {
                report.phase = "coordinate_intervention".into();
                for scale in [
                    -options.intervention_fraction,
                    options.intervention_fraction,
                ] {
                    deadline.check()?;
                    let entries: Vec<_> = coordinates
                        .leading_negative
                        .iter()
                        .map(|c| WeightDelta {
                            row: c.row,
                            column: c.column,
                            delta: scale * c.weight,
                        })
                        .collect();
                    let state = model.prepare_intervention(&coordinates.tensor, entries.clone())?;
                    prefill(
                        &mut model,
                        &mut gpu,
                        &inputs,
                        None,
                        Some(&state),
                        &deadline,
                        &mut report.computation.intervention_decoder_steps,
                    )?;
                    let (changed, _) = observe_pair(
                        &mut model,
                        &mut gpu,
                        &inputs,
                        &direction,
                        None,
                        Some(&state),
                        false,
                        &deadline,
                        &mut report.computation.intervention_decoder_steps,
                    )?;
                    report.interventions.push(InterventionResult {
                        tensor: coordinates.tensor.clone(),
                        relative_change: scale,
                        delta_norm_squared: entries.iter().map(|e| e.delta * e.delta).sum(),
                        entries,
                        metric: ProbabilityPair::new(&changed[0].logits, &changed[1].logits)?
                            .metric,
                        prefix_recomputed: inputs.prefix.len(),
                    });
                }
            }
        }
        Ok(())
    })();
    report.elapsed_seconds = started.elapsed().as_secs_f64();
    report.phase = match &calculation {
        Ok(()) => "complete",
        Err(e) if e.is::<BudgetExpired>() => "partial",
        Err(_) => "failed",
    }
    .into();
    if let Err(error) = &calculation {
        report.stop_reason = Some(format!("{error:#}"));
    }
    save(&options.output.join("report.json"), &report)?;
    save(
        &options.output.join("status.json"),
        &json!({"phase":report.phase,"outcome":report.outcome,"stop_reason":report.stop_reason,"elapsed_seconds":report.elapsed_seconds}),
    )?;
    calculation?;
    Ok(report)
}
