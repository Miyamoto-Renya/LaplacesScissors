//! 접두사 관측, 출력 평면 탐색, 이차부등식 해법과 실제 생성 검증.
use crate::{
    attribution::{self, Attribution, LossPoint},
    gpu::{Gpu, Tensor},
    hub::ModelFiles,
    model::{DeviceRotation, Frame, Model},
    probes::Case,
    response::{self, ObservedSpace, ReadoutEdit, ResponseRecord, SubspaceRotation, WeightMetric},
    rotation::{self, FeasibleSet, FeasibleSummary, Pressure, ProtectedSpace, ReadoutRotation},
    text::{EncodedTarget, Text},
};
use anyhow::{Context, Result, ensure};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct BudgetExpired;
impl std::fmt::Display for BudgetExpired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "진단 시간 예산을 사용했습니다.")
    }
}
impl std::error::Error for BudgetExpired {}
pub struct Deadline {
    end: Instant,
}
impl Deadline {
    pub fn after(duration: Duration) -> Result<Self> {
        Ok(Self {
            end: Instant::now()
                .checked_add(duration)
                .context("시간 예산 범위")?,
        })
    }
    pub fn check(&self) -> Result<()> {
        if Instant::now() >= self.end {
            Err(BudgetExpired.into())
        } else {
            Ok(())
        }
    }
}
pub fn save(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("partial");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
pub fn argmax(values: &[f32]) -> Result<u32> {
    ensure!(
        !values.is_empty() && values.iter().all(|x| x.is_finite()),
        "유한 vocabulary 점수가 필요합니다."
    );
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    Ok(best as u32)
}
fn other(logits: &[f32], target: u32) -> Result<usize> {
    ensure!(
        logits.len() > 1 && (target as usize) < logits.len(),
        "정답 token의 vocabulary 범위를 확인하세요."
    );
    (0..logits.len())
        .filter(|i| *i != target as usize)
        .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then(b.cmp(a)))
        .context("경쟁 token")
}
pub fn prefill(
    model: &mut Model,
    gpu: &mut Gpu,
    prompt: &[u32],
    edit: Option<&DeviceRotation>,
    deadline: &Deadline,
) -> Result<Frame> {
    ensure!(
        !prompt.is_empty() && prompt.len() <= model.max_context(),
        "입력 token 길이를 확인하세요."
    );
    model.reset();
    let mut frame = None;
    for &token in prompt {
        deadline.check()?;
        frame = Some(model.advance(gpu, token, edit)?);
    }
    frame.context("입력 접두사")
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub text: String,
    pub stopped: bool,
}
fn generate_frames(
    model: &mut Model,
    gpu: &mut Gpu,
    text: &Text,
    prompt: &[u32],
    maximum: usize,
    edit: Option<&DeviceRotation>,
    deadline: &Deadline,
) -> Result<(Generation, Vec<Frame>)> {
    ensure!(
        maximum > 0
            && prompt
                .len()
                .checked_add(maximum)
                .is_some_and(|n| n <= model.max_context()),
        "입력과 생성 길이에 맞는 context를 지정하세요."
    );
    let mut next = prefill(model, gpu, prompt, edit, deadline)?;
    let mut tokens = Vec::new();
    let mut frames = Vec::new();
    let mut stopped = false;
    for position in 0..maximum {
        deadline.check()?;
        let token = argmax(&next.logits)?;
        tokens.push(token);
        frames.push(next);
        if text.stop_tokens.contains(&token) {
            stopped = true;
            break;
        }
        if position + 1 == maximum {
            break;
        }
        next = model.advance(gpu, token, edit)?;
    }
    Ok((
        Generation {
            text: text.decode(&tokens)?,
            tokens,
            stopped,
        },
        frames,
    ))
}
pub fn generate(
    model: &mut Model,
    gpu: &mut Gpu,
    text: &Text,
    prompt: &[u32],
    maximum: usize,
    edit: Option<&DeviceRotation>,
    deadline: &Deadline,
) -> Result<Generation> {
    Ok(generate_frames(model, gpu, text, prompt, maximum, edit, deadline)?.0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Baseline {
    pub case_id: String,
    pub expected: String,
    pub generation: Generation,
    pub correct: bool,
}
pub(crate) struct ObservedPath {
    pub case: usize,
    pub prompt: Vec<u32>,
    pub tokens: Vec<u32>,
    pub frames: Vec<Frame>,
    pub protected: bool,
}
struct TargetPoint<'a> {
    target: u32,
    frame: &'a Frame,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub index: usize,
    pub method: String,
    pub rotation: ReadoutEdit,
    pub feasible: Option<FeasibleSummary>,
    pub edit_norm_squared: Option<f64>,
    pub protected_projection_residual: f64,
    pub predicted_min_margin: Option<f64>,
    pub joint_steps: Vec<JointStep>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<CompletionRecord>,
    pub verification: Option<Verification>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionRecord {
    pub observed_rank: usize,
    pub original_norm_squared: f64,
    pub completed_norm_squared: f64,
    pub maximum_angle_degrees: f64,
    pub coordinate_residual: f64,
    pub native_max_scaled_change: Option<f64>,
    pub native_argmax_preserved: Option<bool>,
    pub native_margin_conditions_preserved: Option<bool>,
    pub native_protection_satisfied: Option<bool>,
    pub applied: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointStep {
    pub response: ResponseRecord,
    pub step: f64,
    pub objective_before: f64,
    pub objective_after: f64,
    pub maximum_angle_degrees: f64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Verification {
    pub method: String,
    pub target_min_margin: f64,
    pub protected_max_logit_change: f64,
    pub protected_max_scaled_change: f64,
    pub generations: Vec<Baseline>,
    pub certified_paths: Vec<bool>,
    pub protected_sequences_preserved: bool,
    pub requested_answers_generated: bool,
    pub readout_calls: usize,
    pub decoder_steps: usize,
    pub accepted: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeparationObjective {
    pub metric: String,
    pub before: f64,
    pub after: f64,
    pub probe_input_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub version: u32,
    pub model_revision: Option<String>,
    pub weight_hashes: BTreeMap<String, String>,
    pub input_hashes: BTreeMap<String, String>,
    pub rotation: ReadoutEdit,
    pub readout_weight: String,
    pub readout_bias: Option<String>,
    pub hidden_width: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_margin: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub separation: Option<SeparationObjective>,
}
impl Proposal {
    pub fn for_separation(
        files: &ModelFiles,
        model: &Model,
        rotation: ReadoutEdit,
        separation: SeparationObjective,
    ) -> Self {
        Self {
            version: 3,
            model_revision: files.revision.clone(),
            weight_hashes: weight_hashes(files),
            input_hashes: computational_hashes(files),
            rotation,
            readout_weight: model.graph().readout.weight.clone(),
            readout_bias: model.graph().readout.bias.clone(),
            hidden_width: model.graph().hidden_size,
            target_margin: None,
            separation: Some(separation),
        }
    }
    pub fn validate(&self, files: &ModelFiles, model: &Model) -> Result<()> {
        ensure!(
            (1..=3).contains(&self.version)
                && self.hidden_width == model.graph().hidden_size
                && self.readout_weight == model.graph().readout.weight
                && self.readout_bias == model.graph().readout.bias,
            "수정안과 모델의 출력 연산 계약을 맞추세요."
        );
        ensure!(
            self.weight_hashes == weight_hashes(files)
                && !self.weight_hashes.is_empty()
                && self.input_hashes
                    == if self.version == 3 {
                        computational_hashes(files)
                    } else {
                        input_hashes(files)
                    },
            "수정안의 가중치·설정·tokenizer SHA256을 현재 입력과 맞추세요."
        );
        if self.version == 3 {
            let objective = self
                .separation
                .as_ref()
                .context("구별도 수정안의 목적함수")?;
            ensure!(
                self.target_margin.is_none()
                    && objective.metric == "jeffreys"
                    && objective.before.is_finite()
                    && objective.before >= 0.
                    && objective.after.is_finite()
                    && objective.after > objective.before
                    && objective.probe_input_sha256.len() == 64
                    && objective
                        .probe_input_sha256
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit()),
                "구별도의 유한 개선량과 주입 입력 SHA256을 확인하세요."
            );
        } else {
            ensure!(
                self.target_margin
                    .is_some_and(|margin| margin.is_finite() && margin > 0.)
                    && self.separation.is_none(),
                "수정안의 양의 margin을 확인하세요."
            );
        }
        self.rotation.validate(self.hidden_width)
    }
}
fn computational_hashes(files: &ModelFiles) -> BTreeMap<String, String> {
    input_hashes(files)
        .into_iter()
        .filter(|(name, _)| crate::hub::computational_file(name))
        .collect()
}
fn input_hashes(files: &ModelFiles) -> BTreeMap<String, String> {
    files
        .files
        .iter()
        .map(|f| (f.name.clone(), f.sha256.clone()))
        .collect()
}
fn weight_hashes(files: &ModelFiles) -> BTreeMap<String, String> {
    files
        .files
        .iter()
        .filter(|f| f.name.ends_with(".safetensors"))
        .map(|f| (f.name.clone(), f.sha256.clone()))
        .collect()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Policy {
    pub joint_steps: usize,
    pub joint_rank: usize,
    pub attribution_rows: usize,
    pub attribution_tolerance: f64,
    pub max_edit_norm: Option<f64>,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            joint_steps: 12,
            joint_rank: 64,
            attribution_rows: 6,
            attribution_tolerance: 1e-6,
            max_edit_norm: None,
        }
    }
}
impl Policy {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.joint_steps <= 128 && (2..=256).contains(&self.joint_rank),
            "공동 수정은 최대 128단계, 기저 rank는 2..=256으로 지정하세요."
        );
        ensure!(
            self.attribution_rows <= 128
                && self.attribution_tolerance.is_finite()
                && self.attribution_tolerance > 0.,
            "기여 행 수는 128 이하, 적분 허용 오차는 양수로 지정하세요."
        );
        ensure!(
            self.max_edit_norm.is_none_or(|v| v.is_finite() && v > 0.),
            "가중치 수정 노름 상한은 양수로 지정하세요."
        );
        Ok(())
    }
    fn permits(&self, cost: f64) -> bool {
        self.max_edit_norm.is_none_or(|n| cost <= n * n)
    }
}
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Computation {
    pub observation_decoder_steps: usize,
    pub verification_decoder_steps: usize,
    pub verification_readouts: usize,
    pub joint_search_readouts: usize,
    pub joint_compact_readouts: usize,
    pub joint_image_readouts: usize,
    pub joint_metric_builds: usize,
    pub joint_thin_trials: usize,
    pub joint_dense_trials: usize,
    pub joint_setup_seconds: f64,
    pub joint_search_seconds: f64,
    pub joint_completion_seconds: f64,
    pub polynomial_constraints: usize,
    pub polynomial_screened: usize,
    pub polynomial_square_roots: usize,
}
#[derive(Serialize, Deserialize)]
pub struct Report {
    pub version: u32,
    pub phase: String,
    pub outcome: String,
    pub stop_reason: Option<String>,
    pub source: ModelFiles,
    pub device: Value,
    pub score_contract: String,
    pub precision: String,
    pub parameters: Value,
    pub cases: Vec<Case>,
    pub baseline: Vec<Baseline>,
    pub observed_paths: Vec<Value>,
    pub protected_prefixes: usize,
    pub protected_rank: usize,
    pub target_prefixes: usize,
    pub candidates: Vec<Candidate>,
    pub selected: Option<Proposal>,
    pub attribution: Option<Attribution>,
    pub computation: Computation,
    pub elapsed_seconds: f64,
}
#[derive(Clone)]
pub struct Options {
    pub cases: Vec<Case>,
    pub context: usize,
    pub max_new_tokens: usize,
    pub max_planes: usize,
    pub max_angle_degrees: f64,
    pub margin: f64,
    pub tolerance: f64,
    pub time_budget: Duration,
    pub output: PathBuf,
}
impl Options {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.cases.is_empty()
                && self.max_planes > 0
                && self.max_new_tokens > 0
                && !self.time_budget.is_zero(),
            "사례와 양의 실행 범위를 지정하세요."
        );
        ensure!(
            self.max_angle_degrees.is_finite()
                && self.max_angle_degrees > 0.
                && self.max_angle_degrees < 180.
                && self.margin.is_finite()
                && self.margin > 0.
                && self.tolerance.is_finite()
                && self.tolerance > 0.,
            "회전각은 0도 초과 180도 미만, margin과 허용 오차는 양수로 지정하세요."
        );
        let mut ids = std::collections::BTreeSet::new();
        for case in &self.cases {
            case.validate()?;
            ensure!(ids.insert(&case.id), "중복 사례 id: {}", case.id);
        }
        Ok(())
    }
}
fn fetch_row(
    model: &Model,
    gpu: &mut Gpu,
    rows: &mut BTreeMap<u32, Vec<f32>>,
    token: u32,
) -> Result<()> {
    if let std::collections::btree_map::Entry::Vacant(entry) = rows.entry(token) {
        entry.insert(model.head_row(gpu, token)?);
    }
    Ok(())
}
fn difference(
    model: &Model,
    gpu: &mut Gpu,
    rows: &mut BTreeMap<u32, Vec<f32>>,
    target: u32,
    competitor: u32,
) -> Result<Vec<f64>> {
    fetch_row(model, gpu, rows, target)?;
    fetch_row(model, gpu, rows, competitor)?;
    Ok(rows[&target]
        .iter()
        .zip(&rows[&competitor])
        .map(|(a, b)| f64::from(*a) - f64::from(*b))
        .collect())
}
fn minimum_margin(scores: &[f32], target: u32) -> Result<f64> {
    Ok(f64::from(scores[target as usize]) - f64::from(scores[other(scores, target)?]))
}
fn verify_cached(
    model: &Model,
    gpu: &mut Gpu,
    text: &Text,
    edit: &ReadoutEdit,
    paths: &[ObservedPath],
    options: &Options,
    deadline: &Deadline,
) -> Result<(Verification, Vec<Vec<Vec<f32>>>)> {
    let device = DeviceRotation::upload_edit(gpu, edit, model.graph().hidden_size)?;
    let mut minimum = f64::INFINITY;
    let mut maximum = 0f64;
    let mut scaled = 0f64;
    let mut preserved = true;
    let mut repaired = true;
    let mut calls = 0;
    let mut generations = Vec::new();
    let mut certificates = Vec::new();
    let mut all_scores = Vec::new();
    for path in paths {
        let mut matched = true;
        let mut tokens = Vec::new();
        let mut stopped = false;
        let mut path_scores = Vec::new();
        for (&expected, frame) in path.tokens.iter().zip(&path.frames) {
            deadline.check()?;
            let scores = model.readout(gpu, &frame.hidden, Some(&device))?;
            calls += 1;
            if path.protected {
                for (a, b) in frame.logits.iter().zip(&scores) {
                    let change = (f64::from(*a) - f64::from(*b)).abs();
                    maximum = maximum.max(change);
                    scaled = scaled.max(change / (1. + f64::from(*a).abs()));
                }
            } else {
                minimum = minimum.min(minimum_margin(&scores, expected)?);
            }
            if matched {
                let actual = argmax(&scores)?;
                tokens.push(actual);
                stopped = text.stop_tokens.contains(&actual);
                matched = actual == expected;
            }
            path_scores.push(scores);
        }
        if path.protected {
            preserved &= matched;
        } else {
            repaired &= matched;
        }
        let case = &options.cases[path.case];
        let generation = Generation {
            text: text.decode(&tokens)?,
            tokens,
            stopped,
        };
        let correct = matched && stopped && generation.text == case.answer();
        if !path.protected {
            repaired &= correct;
        }
        generations.push(Baseline {
            case_id: case.id.clone(),
            expected: case.answer().into(),
            generation,
            correct,
        });
        certificates.push(matched);
        all_scores.push(path_scores);
    }
    let accepted =
        minimum >= options.margin && scaled <= options.tolerance && preserved && repaired;
    Ok((
        Verification {
            method: "native_readout_on_observed_prefixes".into(),
            target_min_margin: minimum,
            protected_max_logit_change: maximum,
            protected_max_scaled_change: scaled,
            generations,
            certified_paths: certificates,
            protected_sequences_preserved: preserved,
            requested_answers_generated: repaired,
            readout_calls: calls,
            decoder_steps: 0,
            accepted,
        },
        all_scores,
    ))
}
fn predicted_margin(
    points: &[TargetPoint<'_>],
    plane: &ReadoutRotation,
    wu: &[f32],
    wv: &[f32],
) -> f64 {
    let (a, b) = plane.coefficients();
    let mut minimum = f64::INFINITY;
    for point in points {
        let h: Vec<f64> = point.frame.hidden.iter().map(|&x| f64::from(x)).collect();
        let hu = rotation::dot(&plane.u, &h);
        let hv = rotation::dot(&plane.v, &h);
        let target = point.target as usize;
        for j in 0..wu.len() {
            if j != target {
                let du = f64::from(wu[target]) - f64::from(wu[j]);
                let dv = f64::from(wv[target]) - f64::from(wv[j]);
                minimum = minimum.min(
                    f64::from(point.frame.logits[target]) - f64::from(point.frame.logits[j])
                        + du * (a * hu + b * hv)
                        + dv * (a * hv - b * hu),
                );
            }
        }
    }
    minimum
}
struct CompactReadout {
    image: Tensor,
    baseline: Vec<Tensor>,
    coordinates: DMatrix<f64>,
    vocabulary: usize,
}
impl CompactReadout {
    fn new(
        gpu: &Gpu,
        images: &[Vec<f32>],
        baseline: &[Vec<f32>],
        coordinates: &[Vec<f64>],
    ) -> Result<Self> {
        let rank = images.len();
        let vocabulary = images[0].len();
        ensure!(
            baseline.len() == coordinates.len() && baseline.iter().all(|z| z.len() == vocabulary),
            "압축 출력의 기준 점수와 관측 수를 확인하세요."
        );
        let mut rows = vec![0f32; vocabulary * rank];
        for (i, row) in rows.chunks_exact_mut(rank).enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = images[j][i];
            }
        }
        Ok(Self {
            image: gpu.upload_f32(&rows)?,
            baseline: baseline
                .iter()
                .map(|z| gpu.upload_f32(z))
                .collect::<Result<_>>()?,
            coordinates: DMatrix::from_fn(rank, coordinates.len(), |i, j| coordinates[j][i]),
            vocabulary,
        })
    }
    fn evaluate(
        &self,
        gpu: &mut Gpu,
        rotation: &DMatrix<f64>,
        deadline: &Deadline,
    ) -> Result<Vec<Vec<f32>>> {
        let rank = rotation.nrows();
        let delta = (rotation - DMatrix::identity(rank, rank)) * &self.coordinates;
        let mut result = Vec::with_capacity(self.baseline.len());
        for (i, baseline) in self.baseline.iter().enumerate() {
            deadline.check()?;
            let coefficients: Vec<f32> = delta.column(i).iter().map(|&v| v as f32).collect();
            let input = gpu.upload_f32(&coefficients)?;
            let output = gpu.linear(&input, &self.image, Some(baseline), self.vocabulary, rank)?;
            let scores = gpu.read(&output)?;
            ensure!(
                scores.iter().all(|v| v.is_finite()),
                "압축 출력의 유한한 점수를 확인하세요."
            );
            result.push(scores);
        }
        Ok(result)
    }
}

struct JointCandidate {
    candidate: Candidate,
    completion: Option<SubspaceRotation>,
}

fn joint_candidate(
    model: &Model,
    gpu: &mut Gpu,
    targets: &[TargetPoint<'_>],
    protected: &[&Frame],
    policy: &Policy,
    options: &Options,
    guard: f64,
    deadline: &Deadline,
    work: &mut Computation,
) -> Result<Option<JointCandidate>> {
    let setup_started = Instant::now();
    let width = model.graph().hidden_size;
    let mut fixed = ProtectedSpace::new(width);
    for frame in protected {
        fixed.include(
            &frame
                .hidden
                .iter()
                .map(|&x| f64::from(x))
                .collect::<Vec<_>>(),
        )?;
    }
    let required = options.margin + guard;
    let mut active = Vec::new();
    for (i, point) in targets.iter().enumerate() {
        if minimum_margin(&point.frame.logits, point.target)? >= required {
            fixed.include(
                &point
                    .frame
                    .hidden
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect::<Vec<_>>(),
            )?;
        } else {
            active.push(i);
        }
    }
    if active.is_empty() {
        return Ok(None);
    }
    let mut basis = Vec::new();
    let mut rows = BTreeMap::new();
    for &i in &active {
        response::extend_basis(
            &mut basis,
            &targets[i]
                .frame
                .hidden
                .iter()
                .map(|&x| f64::from(x))
                .collect::<Vec<_>>(),
            &fixed,
            policy.joint_rank,
        )?;
    }
    let mut competitors = Vec::new();
    for &i in &active {
        let point = &targets[i];
        let mut order: Vec<usize> = (0..model.vocab_size())
            .filter(|&j| j != point.target as usize)
            .collect();
        let count = 12.min(order.len());
        if count < order.len() {
            order.select_nth_unstable_by(count, |&a, &b| {
                point.frame.logits[b].total_cmp(&point.frame.logits[a])
            });
        }
        order.truncate(count);
        order.sort_by(|&a, &b| point.frame.logits[b].total_cmp(&point.frame.logits[a]));
        competitors.push(order);
    }
    for rank in 0..12 {
        for (n, &i) in active.iter().enumerate() {
            if let Some(&competitor) = competitors[n].get(rank) {
                deadline.check()?;
                let d = difference(model, gpu, &mut rows, targets[i].target, competitor as u32)?;
                response::extend_basis(&mut basis, &d, &fixed, policy.joint_rank)?;
            }
        }
    }
    if basis.len() < 2 {
        return Ok(None);
    }
    let mut edit = SubspaceRotation::identity(basis);
    let coordinates: Vec<Vec<f64>> = active
        .iter()
        .map(|&i| {
            edit.coordinates(
                &targets[i]
                    .frame
                    .hidden
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let mut current: Vec<Vec<f32>> = active
        .iter()
        .map(|&i| targets[i].frame.logits.clone())
        .collect();
    let mut images = Vec::new();
    for axis in &edit.basis {
        deadline.check()?;
        images.push(model.head_image(gpu, axis)?);
        work.joint_image_readouts += 1;
    }
    let metric = WeightMetric::from_images(&images)?;
    work.joint_metric_builds += 1;
    let observed = ObservedSpace::new(&coordinates, edit.basis.len())?;
    let compact = CompactReadout::new(gpu, &images, &current, &coordinates)?;
    drop(images);
    work.joint_setup_seconds += setup_started.elapsed().as_secs_f64();
    let objective = |scores: &[Vec<f32>]| -> Result<f64> {
        active
            .iter()
            .enumerate()
            .map(|(n, &i)| {
                Ok((required - minimum_margin(&scores[n], targets[i].target)?)
                    .max(0.)
                    .powi(2))
            })
            .sum()
    };
    let mut history = Vec::new();
    let search_started = Instant::now();
    for _ in 0..policy.joint_steps {
        deadline.check()?;
        let before = objective(&current)?;
        if before <= 1e-16 {
            break;
        }
        let q = edit.matrix();
        let mut ds = Vec::new();
        let mut demand = Vec::new();
        for (n, &i) in active.iter().enumerate() {
            let j = other(&current[n], targets[i].target)?;
            let d = difference(model, gpu, &mut rows, targets[i].target, j as u32)?;
            let local = q.transpose() * nalgebra::DVector::from_vec(edit.coordinates(&d));
            ds.push(local.as_slice().to_vec());
            demand.push((required - minimum_margin(&current[n], targets[i].target)?).max(0.));
        }
        let (path, record) = response::response(&q, &metric, &ds, &coordinates, &demand)?;
        if record.rank == 0 || record.generator_norm < 1e-14 {
            break;
        }
        let mut chosen = None;
        for step in [1., 0.5, 0.25, 0.125, 0.0625] {
            deadline.check()?;
            let (matrix, cost) = path.trial(step)?;
            if record.solve_dimension < q.nrows() {
                work.joint_thin_trials += 1;
            } else {
                work.joint_dense_trials += 1;
            }
            let mut trial = edit.clone();
            trial.set_matrix(&matrix);
            let angle = trial.angle_degrees();
            if angle > options.max_angle_degrees || !policy.permits(cost) {
                continue;
            }
            let scores = compact.evaluate(gpu, &matrix, deadline)?;
            work.joint_search_readouts += scores.len();
            work.joint_compact_readouts += scores.len();
            let after = objective(&scores)?;
            if after < before {
                chosen = Some((
                    trial,
                    scores,
                    JointStep {
                        response: record.clone(),
                        step,
                        objective_before: before,
                        objective_after: after,
                        maximum_angle_degrees: angle,
                    },
                ));
                break;
            }
        }
        let Some((trial, scores, step)) = chosen else {
            break;
        };
        edit = trial;
        current = scores;
        history.push(step);
    }
    work.joint_search_seconds += search_started.elapsed().as_secs_f64();
    if history.is_empty() {
        return Ok(None);
    }
    let residual = protected
        .iter()
        .flat_map(|p| {
            edit.basis.iter().map(move |u| {
                rotation::dot(
                    u,
                    &p.hidden.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
                )
                .abs()
            })
        })
        .fold(0., f64::max);
    let matrix = edit.matrix();
    let cost = metric.norm_squared(&matrix);
    let completion_started = Instant::now();
    let mut completion = None;
    let mut completion_record = None;
    if observed.rank < edit.basis.len() {
        deadline.check()?;
        let completed = observed.complete(&matrix, &metric)?;
        let completed_cost = metric.norm_squared(&completed);
        let coordinate_residual = ((&completed - &matrix) * &compact.coordinates).amax();
        let mut alternative = edit.clone();
        alternative.set_matrix(&completed);
        alternative.validate(width)?;
        let angle = alternative.angle_degrees();
        completion_record = Some(CompletionRecord {
            observed_rank: observed.rank,
            original_norm_squared: cost,
            completed_norm_squared: completed_cost,
            maximum_angle_degrees: angle,
            coordinate_residual,
            native_max_scaled_change: None,
            native_argmax_preserved: None,
            native_margin_conditions_preserved: None,
            native_protection_satisfied: None,
            applied: false,
        });
        if angle <= options.max_angle_degrees
            && policy.permits(completed_cost)
            && completed_cost <= cost
        {
            completion = Some(alternative);
        }
    }
    work.joint_completion_seconds += completion_started.elapsed().as_secs_f64();
    Ok(Some(JointCandidate {
        candidate: Candidate {
            index: 0,
            method: "joint_response_cayley_compact".into(),
            rotation: ReadoutEdit::Joint(edit),
            feasible: None,
            edit_norm_squared: Some(cost),
            protected_projection_residual: residual,
            predicted_min_margin: None,
            joint_steps: history,
            completion: completion_record,
            verification: None,
        },
        completion,
    }))
}
pub fn diagnose(
    files: ModelFiles,
    cache: &Path,
    python: &Path,
    options: Options,
) -> Result<Report> {
    diagnose_with_policy(files, cache, python, options, Policy::default())
}
pub fn diagnose_with_policy(
    files: ModelFiles,
    cache: &Path,
    python: &Path,
    options: Options,
    policy: Policy,
) -> Result<Report> {
    options.validate()?;
    policy.validate()?;
    ensure!(
        !options.output.join("report.json").exists(),
        "새 결과 폴더를 지정하세요."
    );
    let started = Instant::now();
    let deadline = Deadline::after(options.time_budget)?;
    let mut report=Report{version:2,phase:"loading".into(),outcome:"pending".into(),stop_reason:None,source:files.clone(),device:Value::Null,
        score_contract:"all-vocabulary target margin including EOS; protected original paths; native cached-prefix verification".into(),
        precision:"original F32/F16/BF16 storage; F32 compact search and native verification; F64 intervals, metric, response, completion and attribution".into(),
        parameters:json!({"context":options.context,"max_new_tokens":options.max_new_tokens,"max_planes":options.max_planes,
            "max_angle_degrees":options.max_angle_degrees,"margin":options.margin,"validation_tolerance":options.tolerance,
            "time_budget_seconds":options.time_budget.as_secs_f64(),"policy":policy}),
        cases:options.cases.clone(),baseline:Vec::new(),observed_paths:Vec::new(),protected_prefixes:0,protected_rank:0,target_prefixes:0,
        candidates:Vec::new(),selected:None,attribution:None,computation:Computation::default(),elapsed_seconds:0.};
    save(&options.output.join("manifest.json"), &report)?;
    let calculation = (|| -> Result<()> {
        let text = Text::load(&files.root, false)?;
        let mut gpu = Gpu::new(cache, python)?;
        let mut model = Model::load(&files.root, &mut gpu, options.context)?;
        report.device =
            json!({"backend":gpu.backend(),"name":gpu.name(),"architecture":gpu.architecture()});
        report.parameters["chat_template"] = json!(text.template_source);
        report.parameters["stored_parameters"] = json!(model.parameter_count());
        report.parameters["weight_storage"] = json!(model.weight_storage());
        report.parameters["edit_scope"] = json!("virtual_readout");
        save(&options.output.join("manifest.json"), &report)?;
        let mut prepared = Vec::<EncodedTarget>::new();
        for case in &options.cases {
            let encoded = text.target(&case.prompt, case.answer())?;
            ensure!(
                encoded.prompt.len() + options.max_new_tokens <= options.context
                    && encoded.continuation.len() <= options.max_new_tokens
                    && encoded
                        .prompt
                        .iter()
                        .chain(&encoded.continuation)
                        .all(|&t| (t as usize) < model.vocab_size()),
                "{}: context·생성 상한·tokenizer vocabulary를 확인하세요.",
                case.id
            );
            prepared.push(encoded);
        }
        let mut paths = Vec::new();
        let mut space = ProtectedSpace::new(model.graph().hidden_size);
        report.phase = "baseline".into();
        for (case_index, (case, encoded)) in options.cases.iter().zip(&prepared).enumerate() {
            let (generation, frames) = generate_frames(
                &mut model,
                &mut gpu,
                &text,
                &encoded.prompt,
                options.max_new_tokens,
                None,
                &deadline,
            )?;
            report.computation.observation_decoder_steps += model.position();
            let correct = generation.stopped && generation.text == case.answer();
            let path = if correct {
                for frame in &frames {
                    space.include(
                        &frame
                            .hidden
                            .iter()
                            .map(|&x| f64::from(x))
                            .collect::<Vec<_>>(),
                    )?;
                }
                ObservedPath {
                    case: case_index,
                    prompt: encoded.prompt.clone(),
                    tokens: generation.tokens.clone(),
                    frames,
                    protected: true,
                }
            } else {
                let mut frame = frames[0].clone();
                model.rewind(encoded.prompt.len())?;
                let mut observed = Vec::new();
                for (index, &token) in encoded.continuation.iter().enumerate() {
                    deadline.check()?;
                    observed.push(frame);
                    if index + 1 == encoded.continuation.len() {
                        break;
                    }
                    frame = model.advance(&mut gpu, token, None)?;
                    report.computation.observation_decoder_steps += 1;
                }
                ObservedPath {
                    case: case_index,
                    prompt: encoded.prompt.clone(),
                    tokens: encoded.continuation.clone(),
                    frames: observed,
                    protected: false,
                }
            };
            report.observed_paths.push(
                json!({"case_id":case.id,"prompt_tokens":path.prompt,"path_tokens":path.tokens,
                "protected":correct,"observed_prefixes":path.frames.len()}),
            );
            if correct {
                report.protected_prefixes += path.frames.len();
            } else {
                report.target_prefixes += path.frames.len();
            }
            report.baseline.push(Baseline {
                case_id: case.id.clone(),
                expected: case.answer().into(),
                generation,
                correct,
            });
            paths.push(path);
            save(&options.output.join("baseline.json"), &report.baseline)?;
            eprintln!(
                "baseline {}/{}: {}",
                case_index + 1,
                options.cases.len(),
                case.id
            );
        }
        report.protected_rank = space.rank();
        let targets: Vec<TargetPoint<'_>> = paths
            .iter()
            .filter(|p| !p.protected)
            .flat_map(|p| {
                p.tokens
                    .iter()
                    .zip(&p.frames)
                    .map(|(&target, frame)| TargetPoint { target, frame })
            })
            .collect();
        let protected: Vec<&Frame> = paths
            .iter()
            .filter(|p| p.protected)
            .flat_map(|p| &p.frames)
            .collect();
        if targets.is_empty() {
            report.outcome = "all_target_strings_observed".into();
            return Ok(());
        }
        let guard = 8.
            * options.tolerance
            * targets
                .iter()
                .flat_map(|p| &p.frame.logits)
                .map(|&x| f64::from(x).abs())
                .fold(1., f64::max);
        report.parameters["algebraic_margin_guard"] = json!(guard);
        report.phase = "planes".into();
        let mut rows = BTreeMap::new();
        let mut pressures = Vec::new();
        for point in &targets {
            deadline.check()?;
            let margin = minimum_margin(&point.frame.logits, point.target)?;
            if margin < options.margin {
                let competitor = other(&point.frame.logits, point.target)?;
                pressures.push(Pressure {
                    hidden: point.frame.hidden.iter().map(|&x| f64::from(x)).collect(),
                    difference: difference(
                        &model,
                        &mut gpu,
                        &mut rows,
                        point.target,
                        competitor as u32,
                    )?,
                    weight: options.margin - margin,
                });
            }
        }
        let planes = rotation::planes(&space, &pressures, options.max_planes)?;
        for (index, mut plane) in planes.into_iter().enumerate() {
            deadline.check()?;
            let wu = model.head_image(&mut gpu, &plane.u)?;
            let wv = model.head_image(&mut gpu, &plane.v)?;
            let mut feasible =
                FeasibleSet::new((options.max_angle_degrees.to_radians() / 2.).tan())?;
            'points: for point in &targets {
                let h: Vec<_> = point.frame.hidden.iter().map(|&x| f64::from(x)).collect();
                let hu = rotation::dot(&plane.u, &h);
                let hv = rotation::dot(&plane.v, &h);
                let target = point.target as usize;
                for j in 0..model.vocab_size() {
                    if j != target {
                        if j % 1024 == 0 {
                            deadline.check()?;
                        }
                        let du = f64::from(wu[target]) - f64::from(wu[j]);
                        let dv = f64::from(wv[target]) - f64::from(wv[j]);
                        let alpha = du * hv - dv * hu;
                        let beta = du * hu + dv * hv;
                        let c = f64::from(point.frame.logits[target])
                            - f64::from(point.frame.logits[j])
                            - options.margin
                            - guard;
                        feasible.constrain(c - 2. * beta, 2. * alpha, c)?;
                        if feasible.is_empty() {
                            break 'points;
                        }
                    }
                }
            }
            report.computation.polynomial_constraints += feasible.work.constraints;
            report.computation.polynomial_screened += feasible.work.screened;
            report.computation.polynomial_square_roots += feasible.work.square_roots;
            let tangent = feasible.nearest();
            plane.tangent = tangent.unwrap_or(0.);
            let cost = tangent
                .map(|_| plane.norm_squared(&wu, &wv))
                .filter(|v| policy.permits(*v));
            let predicted = tangent.map(|_| predicted_margin(&targets, &plane, &wu, &wv));
            let residual = protected
                .iter()
                .map(|f| {
                    let h: Vec<_> = f.hidden.iter().map(|&x| f64::from(x)).collect();
                    rotation::dot(&plane.u, &h)
                        .abs()
                        .max(rotation::dot(&plane.v, &h).abs())
                })
                .fold(0., f64::max);
            report.candidates.push(Candidate {
                index,
                method: "plane_forbidden_intervals".into(),
                rotation: plane.into(),
                feasible: Some(feasible.summary()),
                edit_norm_squared: cost,
                protected_projection_residual: residual,
                predicted_min_margin: predicted,
                joint_steps: Vec::new(),
                completion: None,
                verification: None,
            });
        }
        let mut order: Vec<usize> = (0..report.candidates.len())
            .filter(|&i| report.candidates[i].edit_norm_squared.is_some())
            .collect();
        order.sort_by(|&a, &b| {
            report.candidates[a]
                .edit_norm_squared
                .unwrap()
                .total_cmp(&report.candidates[b].edit_norm_squared.unwrap())
        });
        let mut selected_scores = None;
        let mut selected_index = None;
        report.phase = "verification".into();
        for index in order {
            let (verification, scores) = verify_cached(
                &model,
                &mut gpu,
                &text,
                &report.candidates[index].rotation,
                &paths,
                &options,
                &deadline,
            )?;
            report.computation.verification_readouts += verification.readout_calls;
            let accepted = verification.accepted;
            report.candidates[index].verification = Some(verification);
            if accepted {
                selected_scores = Some(scores);
                selected_index = Some(index);
                break;
            }
        }
        if selected_index.is_none() && policy.joint_steps > 0 {
            report.phase = "joint_response".into();
            if let Some(JointCandidate {
                mut candidate,
                completion,
            }) = joint_candidate(
                &model,
                &mut gpu,
                &targets,
                &protected,
                &policy,
                &options,
                guard,
                &deadline,
                &mut report.computation,
            )? {
                candidate.index = report.candidates.len();
                let (mut verification, mut scores) = verify_cached(
                    &model,
                    &mut gpu,
                    &text,
                    &candidate.rotation,
                    &paths,
                    &options,
                    &deadline,
                )?;
                report.computation.verification_readouts += verification.readout_calls;
                if let Some(completed) = completion {
                    let completed = ReadoutEdit::Joint(completed);
                    let (measured, completed_scores) = verify_cached(
                        &model, &mut gpu, &text, &completed, &paths, &options, &deadline,
                    )?;
                    report.computation.verification_readouts += measured.readout_calls;
                    let record = candidate.completion.as_mut().expect("prepared completion");
                    let mut scaled = 0f64;
                    let mut tokens_preserved = true;
                    let mut margins_preserved = true;
                    for ((path, before), after) in paths.iter().zip(&scores).zip(&completed_scores)
                    {
                        for ((&target, before), after) in path.tokens.iter().zip(before).zip(after)
                        {
                            tokens_preserved &= argmax(before)? == argmax(after)?;
                            if !path.protected && minimum_margin(before, target)? >= options.margin
                            {
                                margins_preserved &=
                                    minimum_margin(after, target)? >= options.margin;
                            }
                            for (&a, &b) in before.iter().zip(after) {
                                scaled = scaled.max(
                                    (f64::from(a) - f64::from(b)).abs() / (1. + f64::from(a).abs()),
                                );
                            }
                        }
                    }
                    let protection = measured.protected_sequences_preserved
                        && measured.protected_max_scaled_change <= options.tolerance;
                    record.native_max_scaled_change = Some(scaled);
                    record.native_argmax_preserved = Some(tokens_preserved);
                    record.native_margin_conditions_preserved = Some(margins_preserved);
                    record.native_protection_satisfied = Some(protection);
                    if tokens_preserved
                        && margins_preserved
                        && scaled <= options.tolerance
                        && protection
                        && (!verification.accepted || measured.accepted)
                    {
                        record.applied = true;
                        candidate.edit_norm_squared = Some(record.completed_norm_squared);
                        candidate.rotation = completed;
                        candidate.method = "joint_response_cayley_compact_completed".into();
                        verification = measured;
                        scores = completed_scores;
                    }
                }
                if verification.accepted {
                    selected_index = Some(candidate.index);
                    selected_scores = Some(scores);
                }
                candidate.verification = Some(verification);
                report.candidates.push(candidate);
            }
        }
        save(&options.output.join("candidates.json"), &report.candidates)?;
        if let Some(index) = selected_index {
            let edit = report.candidates[index].rotation.clone();
            let proposal = Proposal {
                version: 2,
                model_revision: files.revision.clone(),
                weight_hashes: weight_hashes(&files),
                input_hashes: input_hashes(&files),
                rotation: edit.clone(),
                readout_weight: model.graph().readout.weight.clone(),
                readout_bias: model.graph().readout.bias.clone(),
                hidden_width: model.graph().hidden_size,
                target_margin: Some(options.margin),
                separation: None,
            };
            save(&options.output.join("proposal.json"), &proposal)?;
            report.selected = Some(proposal);
            report.outcome = "verified_rotation".into();
            report.phase = "attribution".into();
            let scores = selected_scores.as_ref().unwrap();
            let mut points = Vec::new();
            for (path, path_scores) in paths.iter().zip(scores) {
                if !path.protected {
                    for ((frame, &target), after) in
                        path.frames.iter().zip(&path.tokens).zip(path_scores)
                    {
                        points.push(LossPoint {
                            before: frame,
                            after,
                            target,
                            weight: 1. / targets.len() as f64,
                        });
                    }
                }
            }
            if policy.attribution_rows > 0 {
                let attribution = attribution::analyze(
                    &model,
                    &mut gpu,
                    &text,
                    &edit,
                    &points,
                    policy.attribution_rows,
                    policy.attribution_tolerance,
                    "mean_target_token_nll_including_eos",
                    &deadline,
                )?;
                save(&options.output.join("attribution.json"), &attribution)?;
                report.attribution = Some(attribution);
            }
        } else {
            report.outcome = "candidate_search_finished".into();
        }
        Ok(())
    })();
    report.elapsed_seconds = started.elapsed().as_secs_f64();
    match &calculation {
        Ok(()) => report.phase = "complete".into(),
        Err(error) => {
            report.phase = if error.is::<BudgetExpired>() {
                "partial"
            } else {
                "failed"
            }
            .into();
            report.stop_reason = Some(format!("{error:#}"));
        }
    }
    save(&options.output.join("report.json"), &report)?;
    save(
        &options.output.join("status.json"),
        &json!({"phase":report.phase,"outcome":report.outcome,"elapsed_seconds":report.elapsed_seconds,"stop_reason":report.stop_reason}),
    )?;
    calculation?;
    Ok(report)
}
