//! 고정 decoder의 내부 벡터와 별도 출력 회전 상태를 실행한다.
use crate::{
    gpu::{AttentionConfig, DType, Gpu, NormConfig, RopeConfig, Tensor},
    graph::{AttentionSpec, GraphSpec, LinearSpec, NormKind, NormSpec, TensorShape},
    response::ReadoutEdit,
    rotation::ReadoutRotation,
};
use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs::File, path::Path, sync::Arc};

struct Location {
    file: usize,
    begin: usize,
    size: usize,
    kind: Dtype,
    dimensions: Vec<usize>,
}
struct Checkpoint {
    files: Vec<Mmap>,
    entries: BTreeMap<String, Location>,
}
/// 원본 mmap을 순회하는 행렬 관측. 좌표는 저장 tensor의 행·열 순서다.
pub(crate) struct MatrixView<'a> {
    pub rows: usize,
    pub cols: usize,
    bytes: &'a [u8],
    kind: DType,
}
impl MatrixView<'_> {
    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        let stride = if self.kind == DType::F32 { 4 } else { 2 };
        self.bytes.chunks_exact(stride).map(|cell| match self.kind {
            DType::F32 => f32::from_le_bytes(cell.try_into().unwrap()) as f64,
            DType::F16 => f16::from_bits(u16::from_le_bytes(cell.try_into().unwrap())).to_f64(),
            DType::BF16 => bf16::from_bits(u16::from_le_bytes(cell.try_into().unwrap())).to_f64(),
        })
    }
    pub fn transpose(&self, vector: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            vector.len() == self.rows && vector.iter().all(|v| v.is_finite()),
            "전치 곱의 입력 차원과 유한값을 확인하세요."
        );
        let mut result = vec![0.; self.cols];
        for (index, weight) in self.values().enumerate() {
            result[index % self.cols] += weight * vector[index / self.cols];
        }
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "전치 곱의 유한값을 확인하세요."
        );
        Ok(result)
    }
}
impl Checkpoint {
    fn open(root: &Path) -> Result<Self> {
        let names = crate::hub::weight_files(root)?;
        ensure!(!names.is_empty(), "safetensors 모델 가중치가 필요합니다.");
        let mut result = Self {
            files: Vec::new(),
            entries: BTreeMap::new(),
        };
        for name in names {
            ensure!(
                Path::new(&name)
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_))),
                "shard는 모델 폴더 내부 경로를 사용하세요."
            );
            let map = unsafe { Mmap::map(&File::open(root.join(name))?)? };
            let archive = SafeTensors::deserialize(&map)?;
            for (key, tensor) in archive.tensors() {
                let value = Location {
                    file: result.files.len(),
                    begin: tensor.data().as_ptr() as usize - map.as_ptr() as usize,
                    size: tensor.data().len(),
                    kind: tensor.dtype(),
                    dimensions: tensor.shape().to_vec(),
                };
                ensure!(
                    result.entries.insert(key.clone(), value).is_none(),
                    "중복 가중치 이름: {key}"
                );
            }
            result.files.push(map);
        }
        Ok(result)
    }
    fn metadata(&self) -> BTreeMap<String, TensorShape> {
        self.entries
            .iter()
            .map(|(name, e)| {
                (
                    name.clone(),
                    TensorShape {
                        shape: e.dimensions.clone(),
                        dtype: format!("{:?}", e.kind),
                    },
                )
            })
            .collect()
    }
    fn bytes(&self, name: &str) -> Result<(&[u8], DType)> {
        let e = self
            .entries
            .get(name)
            .with_context(|| format!("가중치 {name}"))?;
        let kind = match e.kind {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            other => bail!("{name}: 저장 자료형 {other:?}에 맞는 연산이 필요합니다."),
        };
        Ok((&self.files[e.file][e.begin..e.begin + e.size], kind))
    }
    fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let (bytes, kind) = self.bytes(name)?;
        Ok(match kind {
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
            DType::F16 => bytes
                .chunks_exact(2)
                .map(|x| f16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32())
                .collect(),
            DType::BF16 => bytes
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32())
                .collect(),
        })
    }
}
struct Bank {
    values: Vec<Tensor>,
    names: BTreeMap<String, usize>,
}
impl Bank {
    fn take(&mut self, source: &Checkpoint, gpu: &Gpu, name: &str) -> Result<usize> {
        if let Some(&index) = self.names.get(name) {
            return Ok(index);
        }
        let (bytes, kind) = source.bytes(name)?;
        let index = self.values.len();
        self.values.push(gpu.upload_raw(bytes, kind)?);
        self.names.insert(name.into(), index);
        Ok(index)
    }
}
struct Projection {
    weight: usize,
    bias: Option<usize>,
    input: usize,
    output: usize,
}
impl Projection {
    fn bind(
        source: &Checkpoint,
        gpu: &Gpu,
        bank: &mut Bank,
        spec: &LinearSpec,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: bank.take(source, gpu, &spec.weight)?,
            bias: spec
                .bias
                .as_ref()
                .map(|n| bank.take(source, gpu, n))
                .transpose()?,
            input,
            output,
        })
    }
    fn run(&self, gpu: &mut Gpu, bank: &Bank, x: &Tensor, with_bias: bool) -> Result<Tensor> {
        gpu.linear(
            x,
            &bank.values[self.weight],
            if with_bias {
                self.bias.map(|i| &bank.values[i])
            } else {
                None
            },
            self.output,
            self.input,
        )
    }
    fn with_intervention(
        &self,
        gpu: &mut Gpu,
        bank: &Bank,
        x: &Tensor,
        intervention: Option<&SparseIntervention>,
    ) -> Result<Tensor> {
        let output = self.run(gpu, bank, x, true)?;
        match intervention.filter(|edit| edit.weight == self.weight) {
            Some(edit) => {
                let input = gpu.read(x)?;
                let mut delta = vec![0f64; self.output];
                for entry in &edit.entries {
                    delta[entry.row] += entry.delta * f64::from(input[entry.column]);
                }
                let device_delta =
                    gpu.upload_f32(&delta.into_iter().map(|v| v as f32).collect::<Vec<_>>())?;
                gpu.combine(&output, &device_delta, 1.)
            }
            None => Ok(output),
        }
    }
}
struct Normalization {
    gain: Tensor,
    bias: Option<usize>,
    width: usize,
    groups: usize,
    epsilon: f32,
    center: bool,
}
impl Normalization {
    fn bind(
        source: &Checkpoint,
        gpu: &Gpu,
        bank: &mut Bank,
        spec: &NormSpec,
        groups: usize,
        width: usize,
    ) -> Result<Self> {
        let values = source
            .floats(&spec.weight)?
            .into_iter()
            .map(|v| v + spec.gain_offset)
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "정규화 gain의 유한값을 확인하세요."
        );
        Ok(Self {
            gain: gpu.upload_f32(&values)?,
            bias: spec
                .bias
                .as_ref()
                .map(|n| bank.take(source, gpu, n))
                .transpose()?,
            width,
            groups,
            epsilon: spec.epsilon,
            center: spec.kind == NormKind::Layer,
        })
    }
    fn run(&self, gpu: &mut Gpu, bank: &Bank, x: &Tensor) -> Result<Tensor> {
        gpu.norm(
            x,
            &self.gain,
            self.bias.map(|i| &bank.values[i]),
            NormConfig {
                groups: self.groups,
                width: self.width,
                epsilon: self.epsilon,
                center: self.center,
            },
        )
    }
}
struct DecoderBlock {
    input: Normalization,
    post: Normalization,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Option<Normalization>,
    k_norm: Option<Normalization>,
    gate: Projection,
    up: Projection,
    down: Projection,
    keys: Tensor,
    values: Tensor,
    attention: AttentionSpec,
    activation: u32,
}
pub enum DeviceRotation {
    Plane {
        u: Tensor,
        v: Tensor,
        cosine_minus_one: f32,
        sine: f32,
    },
    Joint {
        basis: Tensor,
        transpose: Tensor,
        delta: Tensor,
        rank: usize,
        width: usize,
    },
}
impl DeviceRotation {
    pub fn upload(gpu: &Gpu, spec: &ReadoutRotation, width: usize) -> Result<Self> {
        spec.validate(width)?;
        let (a, b) = spec.coefficients();
        Ok(Self::Plane {
            u: gpu.upload_f32(&spec.u.iter().map(|x| *x as f32).collect::<Vec<_>>())?,
            v: gpu.upload_f32(&spec.v.iter().map(|x| *x as f32).collect::<Vec<_>>())?,
            cosine_minus_one: a as f32,
            sine: b as f32,
        })
    }
    pub fn upload_edit(gpu: &Gpu, spec: &ReadoutEdit, width: usize) -> Result<Self> {
        match spec {
            ReadoutEdit::Plane(p) => Self::upload(gpu, p, width),
            ReadoutEdit::Joint(q) => {
                q.validate(width)?;
                let rank = q.basis.len();
                let basis = q
                    .basis
                    .iter()
                    .flatten()
                    .map(|x| *x as f32)
                    .collect::<Vec<_>>();
                let transpose = (0..width)
                    .flat_map(|j| (0..rank).map(move |i| q.basis[i][j] as f32))
                    .collect::<Vec<_>>();
                let delta = (0..rank)
                    .flat_map(|i| {
                        (0..rank)
                            .map(move |j| (q.transform[i][j] - if i == j { 1. } else { 0. }) as f32)
                    })
                    .collect::<Vec<_>>();
                Ok(Self::Joint {
                    basis: gpu.upload_f32(&basis)?,
                    transpose: gpu.upload_f32(&transpose)?,
                    delta: gpu.upload_f32(&delta)?,
                    rank,
                    width,
                })
            }
        }
    }
    fn apply(&self, gpu: &mut Gpu, hidden: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plane {
                u,
                v,
                cosine_minus_one,
                sine,
            } => gpu.rotate(hidden, u, v, *cosine_minus_one, *sine),
            Self::Joint {
                basis,
                transpose,
                delta,
                rank,
                width,
            } => {
                let coordinates = gpu.linear(hidden, basis, None, *rank, *width)?;
                let shift = gpu.linear(&coordinates, delta, None, *rank, *rank)?;
                let lifted = gpu.linear(&shift, transpose, None, *width, *rank)?;
                gpu.combine(hidden, &lifted, 1.)
            }
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BranchFrame {
    pub layer: usize,
    pub tensor: String,
    pub head_width: Option<usize>,
    pub input: Vec<f32>,
    pub output: Vec<f32>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DecoderTrace {
    pub residual: Vec<f32>,
    pub branches: Vec<BranchFrame>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightDelta {
    pub row: usize,
    pub column: usize,
    pub delta: f64,
}
/// 준비 시 좌표를 검사한 별도 projection 개입 상태.
pub struct SparseIntervention {
    owner: Arc<()>,
    weight: usize,
    entries: Vec<WeightDelta>,
}
pub struct Model {
    identity: Arc<()>,
    source: Checkpoint,
    graph: GraphSpec,
    bank: Bank,
    embedding: usize,
    blocks: Vec<DecoderBlock>,
    final_norm: Normalization,
    head: Projection,
    cursor: usize,
    capacity: usize,
    parameters: usize,
    tensors: usize,
    storage: BTreeMap<String, usize>,
}
impl Model {
    pub fn load(root: &Path, gpu: &mut Gpu, capacity: usize) -> Result<Self> {
        let source = Checkpoint::open(root)?;
        let graph = GraphSpec::load(root, &source.metadata())?;
        ensure!(
            capacity > 0 && capacity <= graph.max_context && capacity <= i32::MAX as usize,
            "실행 context 범위를 확인하세요."
        );
        let mut bank = Bank {
            values: Vec::new(),
            names: BTreeMap::new(),
        };
        let embedding = bank.take(&source, gpu, &graph.embedding)?;
        let mut blocks = Vec::new();
        let width = graph.hidden_size;
        for block in &graph.blocks {
            let a = &block.attention;
            let mid = block.mlp.intermediate_size;
            let length = capacity
                .checked_mul(a.kv_heads)
                .and_then(|x| x.checked_mul(a.head_dim))
                .context("KV 크기")?;
            ensure!(length <= i32::MAX as usize, "KV 인덱스 범위를 확인하세요.");
            blocks.push(DecoderBlock {
                input: Normalization::bind(&source, gpu, &mut bank, &block.input_norm, 1, width)?,
                post: Normalization::bind(
                    &source,
                    gpu,
                    &mut bank,
                    &block.post_attention_norm,
                    1,
                    width,
                )?,
                q: Projection::bind(
                    &source,
                    gpu,
                    &mut bank,
                    &a.query,
                    width,
                    a.query_heads * a.head_dim,
                )?,
                k: Projection::bind(
                    &source,
                    gpu,
                    &mut bank,
                    &a.key,
                    width,
                    a.kv_heads * a.head_dim,
                )?,
                v: Projection::bind(
                    &source,
                    gpu,
                    &mut bank,
                    &a.value,
                    width,
                    a.kv_heads * a.head_dim,
                )?,
                o: Projection::bind(
                    &source,
                    gpu,
                    &mut bank,
                    &a.output,
                    a.query_heads * a.head_dim,
                    width,
                )?,
                q_norm: a
                    .query_norm
                    .as_ref()
                    .map(|s| {
                        Normalization::bind(&source, gpu, &mut bank, s, a.query_heads, a.head_dim)
                    })
                    .transpose()?,
                k_norm: a
                    .key_norm
                    .as_ref()
                    .map(|s| {
                        Normalization::bind(&source, gpu, &mut bank, s, a.kv_heads, a.head_dim)
                    })
                    .transpose()?,
                gate: Projection::bind(&source, gpu, &mut bank, &block.mlp.gate, width, mid)?,
                up: Projection::bind(&source, gpu, &mut bank, &block.mlp.up, width, mid)?,
                down: Projection::bind(&source, gpu, &mut bank, &block.mlp.down, mid, width)?,
                keys: gpu.zeros(length)?,
                values: gpu.zeros(length)?,
                attention: a.clone(),
                activation: block.mlp.activation.code(),
            });
        }
        let final_norm = Normalization::bind(&source, gpu, &mut bank, &graph.final_norm, 1, width)?;
        let head = Projection::bind(
            &source,
            gpu,
            &mut bank,
            &graph.readout,
            width,
            graph.vocab_size,
        )?;
        let mut storage = BTreeMap::new();
        for entry in source.entries.values() {
            *storage.entry(format!("{:?}", entry.kind)).or_insert(0) +=
                entry.dimensions.iter().product::<usize>();
        }
        let parameters = storage.values().sum();
        let tensors = source.entries.len();
        Ok(Self {
            identity: Arc::new(()),
            source,
            graph,
            bank,
            embedding,
            blocks,
            final_norm,
            head,
            cursor: 0,
            capacity,
            parameters,
            tensors,
            storage,
        })
    }
    pub fn graph(&self) -> &GraphSpec {
        &self.graph
    }
    pub fn position(&self) -> usize {
        self.cursor
    }
    pub fn max_context(&self) -> usize {
        self.capacity
    }
    pub fn vocab_size(&self) -> usize {
        self.graph.vocab_size
    }
    pub fn parameter_count(&self) -> usize {
        self.parameters
    }
    pub fn tensor_count(&self) -> usize {
        self.tensors
    }
    pub fn weight_storage(&self) -> &BTreeMap<String, usize> {
        &self.storage
    }
    pub(crate) fn matrix(&self, name: &str) -> Result<MatrixView<'_>> {
        let entry = self
            .source
            .entries
            .get(name)
            .with_context(|| format!("행렬 {name}"))?;
        ensure!(
            entry.dimensions.len() == 2,
            "두 축의 가중치 행렬을 지정하세요."
        );
        let (bytes, kind) = self.source.bytes(name)?;
        Ok(MatrixView {
            rows: entry.dimensions[0],
            cols: entry.dimensions[1],
            bytes,
            kind,
        })
    }
    pub fn readout_transpose(&self, vector: &[f64]) -> Result<Vec<f64>> {
        self.matrix(&self.graph.readout.weight)?.transpose(vector)
    }
    pub fn final_gain(&self) -> Result<Vec<f64>> {
        Ok(self
            .source
            .floats(&self.graph.final_norm.weight)?
            .into_iter()
            .map(|v| f64::from(v + self.graph.final_norm.gain_offset))
            .collect())
    }
    pub fn prepare_intervention(
        &self,
        name: &str,
        entries: Vec<WeightDelta>,
    ) -> Result<SparseIntervention> {
        let weight = *self
            .bank
            .names
            .get(name)
            .with_context(|| format!("적재한 행렬 {name}"))?;
        ensure!(
            weight != self.embedding && !entries.is_empty(),
            "embedding과 분리된 projection의 개입 좌표를 지정하세요."
        );
        let matrix = self.matrix(name)?;
        let mut unique = std::collections::BTreeSet::new();
        for entry in &entries {
            ensure!(
                entry.row < matrix.rows
                    && entry.column < matrix.cols
                    && entry.delta.is_finite()
                    && unique.insert((entry.row, entry.column)),
                "유한한 가중치 변화와 서로 다른 유효 좌표를 지정하세요."
            );
        }
        Ok(SparseIntervention {
            owner: Arc::clone(&self.identity),
            weight,
            entries,
        })
    }
    /// embedding scale까지 적용한 decoder 입구의 수치 벡터.
    pub fn embedding_vector(&self, gpu: &mut Gpu, token: u32) -> Result<Vec<f32>> {
        ensure!(
            (token as usize) < self.vocab_size(),
            "embedding 행 범위를 확인하세요."
        );
        let value = self.embedding_tensor(gpu, token)?;
        gpu.read(&value)
    }
    fn embedding_tensor(&self, gpu: &mut Gpu, token: u32) -> Result<Tensor> {
        let row = gpu.embedding(
            &self.bank.values[self.embedding],
            token as usize,
            self.graph.hidden_size,
        )?;
        if self.graph.embedding_scale == 1. {
            Ok(row)
        } else {
            gpu.combine(&row, &row, self.graph.embedding_scale - 1.)
        }
    }
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
    pub fn rewind(&mut self, position: usize) -> Result<()> {
        ensure!(
            position <= self.cursor,
            "KV 되감기는 관측한 접두사 범위 안에서 실행하세요."
        );
        self.cursor = position;
        Ok(())
    }
    pub fn head_image(&self, gpu: &mut Gpu, vector: &[f64]) -> Result<Vec<f32>> {
        ensure!(
            vector.len() == self.graph.hidden_size && vector.iter().all(|x| x.is_finite()),
            "출력 행렬 방향의 차원을 확인하세요."
        );
        let input = gpu.upload_f32(&vector.iter().map(|x| *x as f32).collect::<Vec<_>>())?;
        let out = self.head.run(gpu, &self.bank, &input, false)?;
        gpu.read(&out)
    }
    pub fn head_row(&self, gpu: &mut Gpu, token: u32) -> Result<Vec<f32>> {
        ensure!(
            (token as usize) < self.vocab_size(),
            "출력 token 범위를 확인하세요."
        );
        let row = gpu.embedding(
            &self.bank.values[self.head.weight],
            token as usize,
            self.graph.hidden_size,
        )?;
        gpu.read(&row)
    }
    pub fn readout(
        &self,
        gpu: &mut Gpu,
        hidden: &[f32],
        rotation: Option<&DeviceRotation>,
    ) -> Result<Vec<f32>> {
        ensure!(
            hidden.len() == self.graph.hidden_size && hidden.iter().all(|x| x.is_finite()),
            "출력 입력의 차원과 유한값을 확인하세요."
        );
        let mut input = gpu.upload_f32(hidden)?;
        if let Some(rotation) = rotation {
            input = rotation.apply(gpu, &input)?;
        }
        let out = self.head.run(gpu, &self.bank, &input, true)?;
        let scores = gpu.read(&out)?;
        ensure!(
            scores.iter().all(|x| x.is_finite()),
            "출력 점수의 유한값을 확인하세요."
        );
        Ok(scores)
    }
    pub fn step(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        rotation: Option<&DeviceRotation>,
    ) -> Result<Vec<f32>> {
        Ok(self.advance(gpu, token, rotation)?.logits)
    }
    pub fn advance(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        rotation: Option<&DeviceRotation>,
    ) -> Result<Frame> {
        ensure!(
            (token as usize) < self.vocab_size() && self.cursor < self.capacity,
            "token과 context 범위를 확인하세요."
        );
        let residual = self.embedding_tensor(gpu, token)?;
        self.decode(gpu, residual, rotation, None, None)
    }
    /// decoder 입구에 한 위치의 수치 상태를 공급한다. 개입 상태를 바꿀 때는 접두사를 다시 실행한다.
    pub fn inject(
        &mut self,
        gpu: &mut Gpu,
        vector: &[f32],
        rotation: Option<&DeviceRotation>,
        intervention: Option<&SparseIntervention>,
        trace: Option<&mut DecoderTrace>,
    ) -> Result<Frame> {
        ensure!(
            vector.len() == self.graph.hidden_size
                && vector.iter().all(|v| v.is_finite())
                && self.cursor < self.capacity,
            "주입 벡터의 폭·유한값·context를 확인하세요."
        );
        ensure!(
            intervention.is_none_or(|edit| Arc::ptr_eq(&edit.owner, &self.identity)),
            "현재 모델에서 준비한 좌표 개입을 지정하세요."
        );
        let residual = gpu.upload_f32(vector)?;
        self.decode(gpu, residual, rotation, intervention, trace)
    }
    fn decode(
        &mut self,
        gpu: &mut Gpu,
        mut residual: Tensor,
        rotation: Option<&DeviceRotation>,
        intervention: Option<&SparseIntervention>,
        mut trace: Option<&mut DecoderTrace>,
    ) -> Result<Frame> {
        if let Some(t) = trace.as_deref_mut() {
            t.branches.clear();
            t.residual.clear();
        }
        for (layer, block) in self.blocks.iter().enumerate() {
            let norm = block.input.run(gpu, &self.bank, &residual)?;
            let mut q = block
                .q
                .with_intervention(gpu, &self.bank, &norm, intervention)?;
            let mut k = block
                .k
                .with_intervention(gpu, &self.bank, &norm, intervention)?;
            let v = block
                .v
                .with_intervention(gpu, &self.bank, &norm, intervention)?;
            if let Some(norm) = &block.q_norm {
                q = norm.run(gpu, &self.bank, &q)?;
            }
            if let Some(norm) = &block.k_norm {
                k = norm.run(gpu, &self.bank, &k)?;
            }
            let a = &block.attention;
            if a.rotary_dim > 0 {
                q = gpu.rope(
                    &q,
                    self.cursor,
                    RopeConfig {
                        heads: a.query_heads,
                        dim: a.head_dim,
                        theta: a.rope_theta,
                        rotary: a.rotary_dim,
                        interleaved: a.rotary_interleaved,
                    },
                )?;
                k = gpu.rope(
                    &k,
                    self.cursor,
                    RopeConfig {
                        heads: a.kv_heads,
                        dim: a.head_dim,
                        theta: a.rope_theta,
                        rotary: a.rotary_dim,
                        interleaved: a.rotary_interleaved,
                    },
                )?;
            }
            let mixed = gpu.attention(
                &q,
                &k,
                &v,
                (&block.keys, &block.values),
                self.cursor,
                AttentionConfig {
                    qheads: a.query_heads,
                    kvheads: a.kv_heads,
                    dim: a.head_dim,
                    scale: a.score_scale,
                },
            )?;
            let projected = block
                .o
                .with_intervention(gpu, &self.bank, &mixed, intervention)?;
            if let Some(t) = trace.as_deref_mut() {
                t.branches.push(BranchFrame {
                    layer,
                    tensor: a.output.weight.clone(),
                    head_width: Some(a.head_dim),
                    input: gpu.read(&mixed)?,
                    output: gpu.read(&projected)?,
                });
            }
            residual = gpu.combine(&residual, &projected, 1.)?;
            let norm = block.post.run(gpu, &self.bank, &residual)?;
            let gate = block
                .gate
                .with_intervention(gpu, &self.bank, &norm, intervention)?;
            let up = block
                .up
                .with_intervention(gpu, &self.bank, &norm, intervention)?;
            let feature = gpu.gated(&gate, &up, block.activation)?;
            let down = block
                .down
                .with_intervention(gpu, &self.bank, &feature, intervention)?;
            if let Some(t) = trace.as_deref_mut() {
                t.branches.push(BranchFrame {
                    layer,
                    tensor: self.graph.blocks[layer].mlp.down.weight.clone(),
                    head_width: None,
                    input: gpu.read(&feature)?,
                    output: gpu.read(&down)?,
                });
            }
            residual = gpu.combine(&residual, &down, 1.)?;
        }
        if let Some(t) = trace {
            t.residual = gpu.read(&residual)?;
        }
        let hidden = self.final_norm.run(gpu, &self.bank, &residual)?;
        let snapshot = gpu.read(&hidden)?;
        let transformed = rotation.map(|r| r.apply(gpu, &hidden)).transpose()?;
        let logits = self.head.with_intervention(
            gpu,
            &self.bank,
            transformed.as_ref().unwrap_or(&hidden),
            intervention,
        )?;
        self.cursor += 1;
        let logits = gpu.read(&logits)?;
        ensure!(
            snapshot.iter().chain(&logits).all(|x| x.is_finite()),
            "모델 출력의 유한값을 확인하세요."
        );
        Ok(Frame {
            hidden: snapshot,
            logits,
        })
    }
}
