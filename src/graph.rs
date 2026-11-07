//! 적재 경계에서 decoder 연산과 tensor 형상을 확정한다.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorShape {
    pub shape: Vec<usize>,
    pub dtype: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    Rms,
    Layer,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormSpec {
    pub weight: String,
    #[serde(default)]
    pub bias: Option<String>,
    pub kind: NormKind,
    pub epsilon: f32,
    #[serde(default)]
    pub gain_offset: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearSpec {
    pub weight: String,
    #[serde(default)]
    pub bias: Option<String>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Silu,
    Gelu,
    Relu,
    GeluTanh,
}
impl Activation {
    pub fn code(self) -> u32 {
        match self {
            Self::Silu => 0,
            Self::Gelu => 1,
            Self::Relu => 2,
            Self::GeluTanh => 3,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionSpec {
    pub query: LinearSpec,
    pub key: LinearSpec,
    pub value: LinearSpec,
    pub output: LinearSpec,
    #[serde(default)]
    pub query_norm: Option<NormSpec>,
    #[serde(default)]
    pub key_norm: Option<NormSpec>,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    #[serde(default)]
    pub rotary_interleaved: bool,
    pub score_scale: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlpSpec {
    pub gate: LinearSpec,
    pub up: LinearSpec,
    pub down: LinearSpec,
    pub intermediate_size: usize,
    pub activation: Activation,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockSpec {
    pub input_norm: NormSpec,
    pub attention: AttentionSpec,
    pub post_attention_norm: NormSpec,
    pub mlp: MlpSpec,
}
fn one() -> f32 {
    1.
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphSpec {
    pub version: u32,
    pub source: String,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub embedding: String,
    #[serde(default = "one")]
    pub embedding_scale: f32,
    pub final_norm: NormSpec,
    pub readout: LinearSpec,
    pub blocks: Vec<BlockSpec>,
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
}
struct Contract<'a>(&'a BTreeMap<String, TensorShape>);
impl Contract<'_> {
    fn tensor(&self, name: &str, dimensions: &[usize]) -> Result<()> {
        let actual = self
            .0
            .get(name)
            .with_context(|| format!("연산에 필요한 tensor: {name}"))?;
        ensure!(
            actual.shape == dimensions,
            "{name}: 요구 형상 {dimensions:?}, 저장 형상 {:?}",
            actual.shape
        );
        ensure!(
            matches!(actual.dtype.as_str(), "F32" | "F16" | "BF16"),
            "{name}: F32/F16/BF16 저장 형상을 지정하세요."
        );
        let count = dimensions
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .context("tensor 크기 범위")?;
        ensure!(
            count > 0 && count <= i32::MAX as usize,
            "{name}: tensor 인덱스 범위가 1..=i32::MAX여야 합니다."
        );
        Ok(())
    }
    fn linear(&self, p: &LinearSpec, input: usize, output: usize) -> Result<()> {
        self.tensor(&p.weight, &[output, input])?;
        if let Some(b) = &p.bias {
            self.tensor(b, &[output])?;
        }
        Ok(())
    }
    fn norm(&self, n: &NormSpec, width: usize) -> Result<()> {
        ensure!(
            n.epsilon.is_finite() && n.epsilon > 0. && n.gain_offset.is_finite(),
            "정규화 epsilon은 양수, gain offset은 유한값이어야 합니다."
        );
        self.tensor(&n.weight, &[width])?;
        if let Some(b) = &n.bias {
            self.tensor(b, &[width])?;
        }
        Ok(())
    }
}
impl GraphSpec {
    pub fn load(root: &Path, tensors: &BTreeMap<String, TensorShape>) -> Result<Self> {
        let path = root.join("laplace.graph.json");
        let graph = if path.is_file() {
            serde_json::from_slice(&std::fs::read(path)?)?
        } else {
            Self::from_config(
                &serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?,
                tensors,
            )?
        };
        graph.validate(tensors)?;
        Ok(graph)
    }
    fn validate(&self, tensors: &BTreeMap<String, TensorShape>) -> Result<()> {
        ensure!(self.version == 1, "실행 그래프 version 1을 지정하세요.");
        ensure!(
            self.hidden_size > 0
                && self.vocab_size > 1
                && self.vocab_size <= u32::MAX as usize
                && self.max_context > 0
                && !self.blocks.is_empty(),
            "decoder 차원과 block 수를 확인하세요."
        );
        ensure!(
            self.embedding_scale.is_finite(),
            "embedding scale은 유한값이어야 합니다."
        );
        ensure!(
            self.eos_token_ids
                .iter()
                .all(|t| (*t as usize) < self.vocab_size),
            "EOS token의 vocabulary 범위를 확인하세요."
        );
        let contract = Contract(tensors);
        let width = self.hidden_size;
        contract.tensor(&self.embedding, &[self.vocab_size, width])?;
        contract.norm(&self.final_norm, width)?;
        contract.linear(&self.readout, width, self.vocab_size)?;
        for block in &self.blocks {
            contract.norm(&block.input_norm, width)?;
            contract.norm(&block.post_attention_norm, width)?;
            let a = &block.attention;
            ensure!(
                a.query_heads > 0
                    && a.kv_heads > 0
                    && a.head_dim > 0
                    && a.query_heads.is_multiple_of(a.kv_heads),
                "query head 수는 KV head 수의 양의 배수여야 합니다."
            );
            ensure!(
                a.rotary_dim <= a.head_dim && a.rotary_dim.is_multiple_of(2),
                "회전 차원은 head 차원 이하의 짝수여야 합니다."
            );
            ensure!(
                a.rope_theta.is_finite()
                    && a.rope_theta > 0.
                    && a.score_scale.is_finite()
                    && a.score_scale > 0.,
                "RoPE 기준과 attention scale은 유한한 양수여야 합니다."
            );
            let q = a
                .query_heads
                .checked_mul(a.head_dim)
                .context("query 크기")?;
            let kv = a.kv_heads.checked_mul(a.head_dim).context("KV 크기")?;
            contract.linear(&a.query, width, q)?;
            contract.linear(&a.key, width, kv)?;
            contract.linear(&a.value, width, kv)?;
            contract.linear(&a.output, q, width)?;
            for n in [&a.query_norm, &a.key_norm].into_iter().flatten() {
                contract.norm(n, a.head_dim)?;
            }
            let m = &block.mlp;
            ensure!(m.intermediate_size > 0, "MLP 폭은 양수여야 합니다.");
            contract.linear(&m.gate, width, m.intermediate_size)?;
            contract.linear(&m.up, width, m.intermediate_size)?;
            contract.linear(&m.down, m.intermediate_size, width)?;
        }
        Ok(())
    }
    fn from_config(c: &Value, tensors: &BTreeMap<String, TensorShape>) -> Result<Self> {
        let required = |key: &str| -> Result<usize> {
            usize::try_from(
                c[key]
                    .as_u64()
                    .with_context(|| format!("config 정수 {key}"))?,
            )
            .context("config 정수 범위")
        };
        let prefixes: Vec<_> = tensors
            .keys()
            .filter_map(|name| name.strip_suffix(".0.self_attn.q_proj.weight"))
            .collect();
        ensure!(
            prefixes.len() == 1,
            "단일 decoder의 별도 Q/K/V 가중치 또는 laplace.graph.json을 지정하세요."
        );
        let prefix = prefixes[0];
        let stem = prefix
            .strip_suffix(".layers")
            .map(|x| format!("{x}."))
            .or_else(|| (prefix == "layers").then(String::new))
            .context("decoder layers 경로")?;
        let linear = |name: &str| LinearSpec {
            weight: format!("{name}.weight"),
            bias: tensors
                .contains_key(&format!("{name}.bias"))
                .then(|| format!("{name}.bias")),
        };
        let epsilon = c["rms_norm_eps"].as_f64().context("config rms_norm_eps")? as f32;
        let offset = if c["model_type"] == "gemma" { 1. } else { 0. };
        let rms = |name: &str, offset: f32| NormSpec {
            weight: format!("{name}.weight"),
            bias: None,
            kind: NormKind::Rms,
            epsilon,
            gain_offset: offset,
        };
        let hidden = required("hidden_size")?;
        let heads = required("num_attention_heads")?;
        let kv = c["num_key_value_heads"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(heads);
        let dim = c["head_dim"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(hidden.checked_div(heads).unwrap_or(0));
        let ratio = c["partial_rotary_factor"].as_f64().unwrap_or(1.);
        let rotary = dim as f64 * ratio;
        ensure!(
            ratio.is_finite() && (0. ..=1.).contains(&ratio) && rotary.fract().abs() < 1e-8,
            "partial_rotary_factor로 얻는 회전 차원은 정수여야 합니다."
        );
        for key in [
            "rope_scaling",
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "num_local_experts",
            "num_experts",
        ] {
            ensure!(
                c[key].is_null(),
                "config {key}를 실행하려면 해당 연산을 확장해야 합니다."
            );
        }
        ensure!(
            !c["use_sliding_window"].as_bool().unwrap_or(false)
                && (c["sliding_window"].is_null() || c["use_sliding_window"] == false),
            "sliding window를 실행하려면 attention 연산을 확장해야 합니다."
        );
        ensure!(
            c["rope_parameters"]["rope_type"]
                .as_str()
                .is_none_or(|s| s == "default"),
            "설정된 RoPE 형식에 맞는 주파수 연산이 필요합니다."
        );
        let theta = c["rope_theta"]
            .as_f64()
            .or_else(|| c["rope_parameters"]["rope_theta"].as_f64())
            .context("config rope_theta")? as f32;
        let activation = match c["hidden_activation"]
            .as_str()
            .or_else(|| c["hidden_act"].as_str())
            .unwrap_or("silu")
        {
            "silu" | "swish" => Activation::Silu,
            "gelu" => Activation::Gelu,
            "relu" => Activation::Relu,
            "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" => Activation::GeluTanh,
            name => bail!("activation {name}의 연산을 정의해야 합니다."),
        };
        let mut blocks = Vec::new();
        for i in 0..required("num_hidden_layers")? {
            let layer = format!("{prefix}.{i}");
            let attn = format!("{layer}.self_attn");
            ensure!(
                [
                    "pre_feedforward_layernorm.weight",
                    "post_feedforward_layernorm.weight"
                ]
                .iter()
                .all(|s| !tensors.contains_key(&format!("{layer}.{s}"))),
                "추가 residual 정규화에 맞는 실행 그래프가 필요합니다."
            );
            let head_norm = |kind: &str| {
                let name = format!("{attn}.{kind}_norm");
                tensors
                    .contains_key(&format!("{name}.weight"))
                    .then(|| rms(&name, 0.))
            };
            blocks.push(BlockSpec {
                input_norm: rms(&format!("{layer}.input_layernorm"), offset),
                post_attention_norm: rms(&format!("{layer}.post_attention_layernorm"), offset),
                attention: AttentionSpec {
                    query: linear(&format!("{attn}.q_proj")),
                    key: linear(&format!("{attn}.k_proj")),
                    value: linear(&format!("{attn}.v_proj")),
                    output: linear(&format!("{attn}.o_proj")),
                    query_norm: head_norm("q"),
                    key_norm: head_norm("k"),
                    query_heads: heads,
                    kv_heads: kv,
                    head_dim: dim,
                    rotary_dim: rotary as usize,
                    rope_theta: theta,
                    rotary_interleaved: c["rope_interleaved"]
                        .as_bool()
                        .or_else(|| c["rotary_interleaved"].as_bool())
                        .unwrap_or(false),
                    score_scale: 1. / (dim as f32).sqrt(),
                },
                mlp: MlpSpec {
                    gate: linear(&format!("{layer}.mlp.gate_proj")),
                    up: linear(&format!("{layer}.mlp.up_proj")),
                    down: linear(&format!("{layer}.mlp.down_proj")),
                    intermediate_size: required("intermediate_size")?,
                    activation,
                },
            });
        }
        let embedding = format!("{stem}embed_tokens.weight");
        let readout = if c["tie_word_embeddings"] == true {
            LinearSpec {
                weight: embedding.clone(),
                bias: tensors
                    .contains_key("lm_head.bias")
                    .then(|| "lm_head.bias".into()),
            }
        } else {
            linear("lm_head")
        };
        let ids = match &c["eos_token_id"] {
            Value::Null => Vec::new(),
            Value::Array(v) => v.clone(),
            v => vec![v.clone()],
        };
        let eos_token_ids = ids
            .into_iter()
            .map(|v| u32::try_from(v.as_u64().context("EOS 정수 id")?).context("EOS id 범위"))
            .collect::<Result<_>>()?;
        Ok(Self {
            version: 1,
            source: "dense_decoder_tensor_contract".into(),
            hidden_size: hidden,
            vocab_size: required("vocab_size")?,
            max_context: required("max_position_embeddings")?,
            embedding,
            embedding_scale: if offset == 1. {
                (hidden as f32).sqrt()
            } else {
                1.
            },
            final_norm: rms(&format!("{stem}norm"), offset),
            readout,
            blocks,
            eos_token_ids,
        })
    }
}
