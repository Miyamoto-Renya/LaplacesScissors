use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use laplace::{
    diagnosis::{self, Deadline, Options, Proposal},
    gpu::Gpu,
    graph::{GraphSpec, TensorShape},
    hub::{self, ModelFiles},
    model::{DeviceRotation, Model},
    probes,
    text::Text,
};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::{Component, Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(
    name = "laplace",
    version,
    about = "보호한 출력을 유지하는 readout 회전 수정안을 계산합니다."
)]
struct Cli {
    #[arg(long, global = true, env = "LAPLACE_CACHE")]
    cache: Option<PathBuf>,
    #[arg(long, global = true, env = "LAPLACE_PYTHON", default_value = "python3")]
    python: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Args)]
struct Source {
    /// Hugging Face 조직/모델, URL 또는 모델 폴더.
    model: String,
    #[arg(long)]
    revision: Option<String>,
}
#[derive(Args)]
struct Search {
    #[arg(long, default_value_t = 8)]
    max_planes: usize,
    #[arg(long, default_value_t = 120.)]
    max_angle_degrees: f64,
    #[arg(long, default_value_t = 0.05)]
    margin: f64,
    #[arg(long, default_value_t = 0.00002)]
    tolerance: f64,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 2048)]
    context: usize,
    #[arg(long,default_value="20m",value_parser=duration)]
    time_budget: Duration,
    /// 공동 회전의 최대 보정 단계. 0은 평면 후보를 계산합니다.
    #[arg(long, default_value_t = 12)]
    joint_steps: usize,
    #[arg(long, default_value_t = 64)]
    joint_rank: usize,
    /// 기여를 기록할 출력 행 수. 0은 생성 수정안을 저장합니다.
    #[arg(long, default_value_t = 6)]
    attribution_rows: usize,
    /// 출력 가중치 변화의 Frobenius 노름 상한.
    #[arg(long)]
    max_edit_norm: Option<f64>,
}
impl Search {
    fn options(&self, cases: Vec<probes::Case>, output: PathBuf) -> Options {
        Options {
            cases,
            context: self.context,
            max_new_tokens: self.max_new_tokens,
            max_planes: self.max_planes,
            max_angle_degrees: self.max_angle_degrees,
            margin: self.margin,
            tolerance: self.tolerance,
            time_budget: self.time_budget,
            output,
        }
    }
    fn policy(&self) -> diagnosis::Policy {
        diagnosis::Policy {
            joint_steps: self.joint_steps,
            joint_rank: self.joint_rank,
            attribution_rows: self.attribution_rows,
            max_edit_norm: self.max_edit_norm,
            ..diagnosis::Policy::default()
        }
    }
}

#[derive(Args)]
struct ProbeSearch {
    /// 가중치에서 만들 접선 방향의 최대 수.
    #[arg(long, default_value_t = 12)]
    directions: usize,
    #[arg(long, default_value_t = 3)]
    prefix_length: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// 같은 길이의 두 상태를 만드는 구면 각도, 라디안.
    #[arg(long, default_value_t = 0.05)]
    probe_angle: f64,
    #[arg(long, default_value_t = 3.)]
    max_angle_degrees: f64,
    #[arg(long)]
    max_edit_norm: Option<f64>,
    #[arg(long, default_value_t = 0.00002)]
    tolerance: f64,
    /// 가장 큰 상쇄 분기에서 추적할 음의 기여 좌표 수.
    #[arg(long, default_value_t = 12)]
    coordinates: usize,
    /// 지목한 좌표의 양방향 개입 비율. 0은 추적과 보호 회전을 실행합니다.
    #[arg(long, default_value_t = 0.1)]
    intervention_fraction: f64,
    #[arg(long,default_value="20m",value_parser=duration)]
    time_budget: Duration,
    /// 저장한 수치 주입 입력 JSON.
    #[arg(long)]
    inputs: Option<PathBuf>,
    /// 저장한 입력과 수정안으로 전체 decoder를 다시 실행합니다.
    #[arg(long, requires = "inputs")]
    proposal: Option<PathBuf>,
}
impl ProbeSearch {
    fn options(self, output: PathBuf) -> laplace::injection::Options {
        laplace::injection::Options {
            directions: self.directions,
            prefix_length: self.prefix_length,
            seed: self.seed,
            probe_angle: self.probe_angle,
            max_angle_degrees: self.max_angle_degrees,
            max_edit_norm: self.max_edit_norm,
            tolerance: self.tolerance,
            coordinates: self.coordinates,
            intervention_fraction: self.intervention_fraction,
            time_budget: self.time_budget,
            inputs: self.inputs,
            proposal: self.proposal,
            output,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// 회전 수정안을 출력 가중치에 반영한 독립 checkpoint를 저장합니다.
    Export {
        #[command(flatten)]
        source: Source,
        #[arg(long)]
        proposal: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// 원본 모델과 revision을 외부 캐시에 준비합니다.
    Fetch {
        #[command(flatten)]
        source: Source,
    },
    /// 가중치 형상과 실행 그래프를 확인합니다.
    Inspect {
        #[command(flatten)]
        source: Source,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// 정답을 계산할 수 있는 입력 사례를 만듭니다.
    Cases {
        #[arg(long, default_value_t = 12)]
        count: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// 원본 출력 또는 저장한 회전 수정안으로 greedy 생성을 실행합니다.
    Generate {
        #[command(flatten)]
        source: Source,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        proposal: Option<PathBuf>,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 2048)]
        context: usize,
        #[arg(long)]
        thinking: bool,
        #[arg(long,default_value="20m",value_parser=duration)]
        time_budget: Duration,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// 정답 token 조건을 풀고 실제 생성으로 수정안을 검증합니다.
    Diagnose {
        #[command(flatten)]
        source: Source,
        #[arg(long)]
        cases: Option<PathBuf>,
        #[arg(long, default_value_t = 12)]
        count: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[command(flatten)]
        search: Search,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// 가중치에서 수치 신호를 만들어 구별도·상쇄 좌표·보호 회전을 계산합니다.
    Probe {
        #[command(flatten)]
        source: Source,
        #[command(flatten)]
        search: ProbeSearch,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// 고정한 기준 곡선과 수정 예산으로 checkpoint 계열의 정체를 분석합니다.
    Scale {
        #[arg(long)]
        manifest: PathBuf,
        #[command(flatten)]
        search: Search,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}
fn duration(input: &str) -> std::result::Result<Duration, String> {
    let (number, multiplier) = match input.as_bytes().last() {
        Some(b'h') => (&input[..input.len() - 1], 3600.),
        Some(b'm') => (&input[..input.len() - 1], 60.),
        Some(b's') => (&input[..input.len() - 1], 1.),
        _ => (input, 1.),
    };
    let seconds = number
        .parse::<f64>()
        .map_err(|_| "시간은 30s, 20m, 1h 형식으로 지정하세요.")?
        * multiplier;
    if !(seconds.is_finite() && seconds > 0.) {
        return Err("양의 시간 예산을 지정하세요.".into());
    }
    Duration::try_from_secs_f64(seconds).map_err(|e| e.to_string())
}
fn cache_path(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(|| {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(|v| PathBuf::from(v).join(".cache"))
                    .unwrap_or_else(std::env::temp_dir)
            })
            .join("laplaces-scissors")
    })
}
fn external(path: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(path)?;
    let mut normalized = PathBuf::new();
    for part in absolute.components() {
        match part {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            _ => normalized.push(part),
        }
    }
    let mut ancestor = normalized.as_path();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        suffix.push(ancestor.file_name().context("결과 경로")?);
        ancestor = ancestor.parent().context("결과 경로의 부모")?;
    }
    let mut resolved = ancestor.canonicalize()?;
    for part in suffix.into_iter().rev() {
        resolved.push(part);
    }
    ensure!(
        !resolved.starts_with(Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize()?),
        "캐시와 실행 결과는 프로젝트 외부 경로를 지정하세요."
    );
    Ok(resolved)
}
fn output_dir(cache: &Path, requested: Option<PathBuf>) -> Result<PathBuf> {
    external(&requested.unwrap_or_else(|| {
        cache.join("runs").join(format!(
            "rotation-{}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            std::process::id()
        ))
    }))
}
fn inspect(files: &ModelFiles) -> Result<Value> {
    let mut metadata = BTreeMap::new();
    for receipt in files
        .files
        .iter()
        .filter(|f| f.name.ends_with(".safetensors"))
    {
        let map = unsafe { Mmap::map(&File::open(files.root.join(&receipt.name))?)? };
        for (name, tensor) in SafeTensors::deserialize(&map)?.tensors() {
            let shape = TensorShape {
                shape: tensor.shape().to_vec(),
                dtype: format!("{:?}", tensor.dtype()),
            };
            ensure!(
                metadata.insert(name.clone(), shape).is_none(),
                "중복 tensor: {name}"
            );
        }
    }
    Ok(json!({"source":files,"graph":GraphSpec::load(&files.root,&metadata)?,"tensors":metadata}))
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    let cache = external(&cache_path(cli.cache))?;
    match cli.command {
        Command::Export {
            source,
            proposal,
            out,
        } => {
            let files = hub::resolve(&source.model, source.revision.as_deref(), &cache)?;
            let output = external(&out)?;
            let specification: Proposal = serde_json::from_slice(&fs::read(proposal)?)?;
            let mut gpu = Gpu::new(&cache.join("kernels"), &cli.python)?;
            let model = Model::load(&files.root, &mut gpu, 1)?;
            let mut report = laplace::materialize::export(&files, &model, &specification, &output)?;
            report["source_loading_device"] = json!({"name":gpu.name(),"backend":gpu.backend(),"architecture":gpu.architecture()});
            diagnosis::save(&output.join("materialization.json"), &report)?;
            println!("materialized checkpoint: {}", output.display());
        }
        Command::Cases { count, seed, out } => {
            ensure!(count > 0, "사례 개수는 양수로 지정하세요.");
            let cases = probes::generate(count, seed);
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            let mut content = String::new();
            for case in &cases {
                content.push_str(&serde_json::to_string(case)?);
                content.push('\n');
            }
            fs::write(&out, content)?;
            println!("{} cases: {}", cases.len(), out.display());
        }
        Command::Fetch { source } => {
            let files = hub::resolve(&source.model, source.revision.as_deref(), &cache)?;
            println!("{}", serde_json::to_string_pretty(&files)?);
        }
        Command::Inspect { source, out } => {
            let files = hub::resolve(&source.model, source.revision.as_deref(), &cache)?;
            let result = inspect(&files)?;
            if let Some(out) = out {
                diagnosis::save(&external(&out)?, &result)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        }
        Command::Generate {
            source,
            prompt,
            proposal,
            max_new_tokens,
            context,
            thinking,
            time_budget,
            out,
        } => {
            let files = hub::resolve(&source.model, source.revision.as_deref(), &cache)?;
            let output = output_dir(&cache, out)?;
            ensure!(
                !output.join("generation.json").exists(),
                "새 결과 폴더를 지정하세요."
            );
            let started = Instant::now();
            let deadline = Deadline::after(time_budget)?;
            let text = Text::load(&files.root, thinking)?;
            let mut gpu = Gpu::new(&cache.join("kernels"), &cli.python)?;
            let mut model = Model::load(&files.root, &mut gpu, context)?;
            let specification = proposal
                .map(|path| -> Result<Proposal> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
                .transpose()?;
            if let Some(p) = &specification {
                p.validate(&files, &model)?;
            }
            let edit = specification
                .as_ref()
                .map(|p| DeviceRotation::upload_edit(&gpu, &p.rotation, model.graph().hidden_size))
                .transpose()?;
            let input = text.encode(&text.render(&prompt)?)?;
            let generation = diagnosis::generate(
                &mut model,
                &mut gpu,
                &text,
                &input,
                max_new_tokens,
                edit.as_ref(),
                &deadline,
            )?;
            diagnosis::save(
                &output.join("generation.json"),
                &json!({"source":files,"device":{"name":gpu.name(),"backend":gpu.backend(),"architecture":gpu.architecture()},"precision":"original weight storage; F32 execution","weight_storage":model.weight_storage(),"prompt":prompt,"input_tokens":input,"thinking":thinking,"chat_template":text.template_source,"proposal":specification,"generation":generation,"elapsed_seconds":started.elapsed().as_secs_f64()}),
            )?;
            println!("{}", generation.text);
            eprintln!("result: {}", output.display());
        }
        Command::Diagnose {
            source,
            cases,
            count,
            seed,
            search,
            out,
        } => {
            let cases = match cases {
                Some(path) => probes::load_cases(&path)?,
                None => probes::generate(count, seed),
            };
            let output = output_dir(&cache, out)?;
            let files = hub::resolve(&source.model, source.revision.as_deref(), &cache)?;
            let report = diagnosis::diagnose_with_policy(
                files,
                &cache.join("kernels"),
                &cli.python,
                search.options(cases, output.clone()),
                search.policy(),
            )?;
            println!(
                "{}: {} candidates; protected rank {}; {}",
                report.outcome,
                report.candidates.len(),
                report.protected_rank,
                output.display()
            );
        }
        Command::Probe {
            source,
            search,
            out,
        } => {
            let output = output_dir(&cache, out)?;
            let files = hub::resolve_weights(&source.model, source.revision.as_deref(), &cache)?;
            let report = laplace::injection::run(
                files,
                &cache.join("kernels"),
                &cli.python,
                search.options(output.clone()),
            )?;
            println!(
                "{}: {} directions; {} coordinate interventions; {}",
                report.outcome,
                report.scan.len(),
                report.interventions.len(),
                output.display()
            );
        }
        Command::Scale {
            manifest,
            search,
            out,
        } => {
            let output = output_dir(&cache, out)?;
            let report = laplace::scaling::run(
                &manifest,
                &cache,
                &cli.python,
                search.options(Vec::new(), output.clone()),
                search.policy(),
            )?;
            println!(
                "{}: {} checkpoints; {} contrasts; {}",
                report.phase,
                report.checkpoints.len(),
                report.contrasts.len(),
                output.display()
            );
        }
    }
    Ok(())
}
