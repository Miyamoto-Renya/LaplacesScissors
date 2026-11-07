//! 장치 버퍼와 이름 붙은 Triton 인수를 연결하는 실행 계층.
use crate::device::{Device, Pointer};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    ffi::CString,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}
#[derive(Clone, Copy)]
pub struct NormConfig {
    pub groups: usize,
    pub width: usize,
    pub epsilon: f32,
    pub center: bool,
}
#[derive(Clone, Copy)]
pub struct RopeConfig {
    pub heads: usize,
    pub dim: usize,
    pub theta: f32,
    pub rotary: usize,
    pub interleaved: bool,
}
#[derive(Clone, Copy)]
pub struct AttentionConfig {
    pub qheads: usize,
    pub kvheads: usize,
    pub dim: usize,
    pub scale: f32,
}
impl DType {
    fn bytes(self) -> usize {
        if self == Self::F32 { 4 } else { 2 }
    }
    fn pointer(self) -> &'static str {
        match self {
            Self::F32 => "*fp32",
            Self::F16 => "*fp16",
            Self::BF16 => "*bf16",
        }
    }
}
pub struct Tensor {
    address: Pointer,
    count: usize,
    kind: DType,
    owner: Arc<Device>,
}
impl Tensor {
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn dtype(&self) -> DType {
        self.kind
    }
}
impl Drop for Tensor {
    fn drop(&mut self) {
        self.owner.free(self.address);
    }
}
struct Executable {
    module: Pointer,
    entry: Pointer,
    shared: u32,
    threads: u32,
    scratch: [(usize, usize); 2],
    owner: Arc<Device>,
}
impl Drop for Executable {
    fn drop(&mut self) {
        unsafe {
            (self.owner.unload)(self.module);
        }
    }
}
enum Cell {
    Address(Pointer),
    Index(i32),
    Scalar(f32),
}
impl Cell {
    fn location(&mut self) -> Pointer {
        match self {
            Self::Address(p) => (p as *mut Pointer).cast(),
            Self::Index(i) => (i as *mut i32).cast(),
            Self::Scalar(f) => (f as *mut f32).cast(),
        }
    }
}
struct Argument {
    name: &'static str,
    signature: &'static str,
    cell: Cell,
}
fn buffer(name: &'static str, x: &Tensor) -> Argument {
    Argument {
        name,
        signature: x.kind.pointer(),
        cell: Cell::Address(x.address),
    }
}
fn index(name: &'static str, x: usize) -> Argument {
    Argument {
        name,
        signature: "i32",
        cell: Cell::Index(x as i32),
    }
}
fn scalar(name: &'static str, x: f32) -> Argument {
    Argument {
        name,
        signature: "fp32",
        cell: Cell::Scalar(x),
    }
}

pub struct Gpu {
    device: Arc<Device>,
    process: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    cache: PathBuf,
    programs: HashMap<String, Executable>,
}
impl Gpu {
    pub fn new(cache: &Path, python: &Path) -> Result<Self> {
        std::fs::create_dir_all(cache)?;
        let source = if let Some(directory) = std::env::var_os("LAPLACE_KERNEL_DIR") {
            PathBuf::from(directory).join("compile.py")
        } else {
            let files = [
                ("ops.py", include_str!("../kernels/ops.py")),
                ("compile.py", include_str!("../kernels/compile.py")),
            ];
            let mut digest = Sha256::new();
            for (name, body) in files {
                digest.update(name);
                digest.update(body);
            }
            let root = cache.join("sources").join(hex::encode(digest.finalize()));
            std::fs::create_dir_all(&root)?;
            for (name, body) in files {
                let path = root.join(name);
                if !path.exists() {
                    std::fs::write(path, body)?;
                }
            }
            root.join("compile.py")
        };
        let device = Arc::new(Device::open(python)?);
        let mut command = Command::new(python);
        command
            .arg("-B")
            .arg(source)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("TRITON_CACHE_DIR", cache.join("triton"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(root) = &device.sdk_root {
            command.env("ROCM_PATH", root);
        }
        let mut process = command.spawn().context("Triton 연결 시작")?;
        let input = process.stdin.take().context("컴파일 입력")?;
        let output = BufReader::new(process.stdout.take().context("컴파일 출력")?);
        Ok(Self {
            device,
            process,
            input,
            output,
            cache: cache.to_owned(),
            programs: HashMap::new(),
        })
    }
    pub fn name(&self) -> &str {
        &self.device.name
    }
    pub fn backend(&self) -> &str {
        self.device.backend
    }
    pub fn architecture(&self) -> &str {
        &self.device.architecture
    }
    pub fn synchronize(&self) -> Result<()> {
        self.device
            .check(unsafe { (self.device.synchronize)() }, "장치 완료 대기")
    }
    fn reserve(&self, count: usize, kind: DType) -> Result<Tensor> {
        ensure!(
            count > 0 && count <= i32::MAX as usize,
            "버퍼 원소 수는 1..=i32::MAX 범위를 사용하세요."
        );
        let bytes = count.checked_mul(kind.bytes()).context("버퍼 크기 범위")?;
        Ok(Tensor {
            address: self.device.allocate(bytes)?,
            count,
            kind,
            owner: Arc::clone(&self.device),
        })
    }
    pub fn upload_raw(&self, data: &[u8], kind: DType) -> Result<Tensor> {
        ensure!(
            !data.is_empty() && data.len().is_multiple_of(kind.bytes()),
            "가중치 저장 자료형과 바이트 크기를 확인하세요."
        );
        let result = self.reserve(data.len() / kind.bytes(), kind)?;
        self.device
            .upload(result.address, data.as_ptr().cast(), data.len())?;
        Ok(result)
    }
    pub fn upload_f32(&self, data: &[f32]) -> Result<Tensor> {
        let result = self.reserve(data.len(), DType::F32)?;
        self.device.upload(
            result.address,
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
        )?;
        Ok(result)
    }
    pub fn zeros(&self, count: usize) -> Result<Tensor> {
        let t = self.reserve(count, DType::F32)?;
        self.device.zero(t.address, count * 4)?;
        Ok(t)
    }
    pub fn read(&self, t: &Tensor) -> Result<Vec<f32>> {
        ensure!(t.kind == DType::F32, "F32 결과 버퍼가 필요합니다.");
        let mut result = vec![0.; t.count];
        self.device
            .download(result.as_mut_ptr().cast(), t.address, t.count * 4)?;
        Ok(result)
    }
    fn dispatch(
        &mut self,
        name: &str,
        constants: Value,
        grid: usize,
        arguments: Vec<Argument>,
    ) -> Result<()> {
        let signature: serde_json::Map<String, Value> = arguments
            .iter()
            .map(|a| (a.name.to_owned(), json!(a.signature)))
            .collect();
        let request = json!({"operation":name,"signature":signature,"constants":constants,"backend":self.backend(),"architecture":self.architecture(),"warp_size":self.device.warp_size,"num_warps":4,"cache":self.cache});
        let key = request.to_string();
        if !self.programs.contains_key(&key) {
            writeln!(self.input, "{key}")?;
            self.input.flush()?;
            let mut response = String::new();
            ensure!(
                self.output.read_line(&mut response)? > 0,
                "Triton 컴파일 응답을 확인하세요."
            );
            let record: Value = serde_json::from_str(&response)?;
            ensure!(
                record.get("error").is_none(),
                "{name} 컴파일: {}",
                record["error"]
            );
            let path = CString::new(record["binary"].as_str().context("장치 코드 경로")?)?;
            let symbol = CString::new(record["name"].as_str().context("장치 함수 이름")?)?;
            let shared = record["shared"]
                .as_u64()
                .context("shared bytes")?
                .try_into()?;
            let threads = record["warps"]
                .as_u64()
                .context("warp count")?
                .checked_mul(self.device.warp_size as u64)
                .context("thread 수 범위")?
                .try_into()?;
            let mut module = std::ptr::null_mut();
            self.device.check(
                unsafe { (self.device.load)(&mut module, path.as_ptr()) },
                "장치 코드 적재",
            )?;
            let mut program = Executable {
                module,
                entry: std::ptr::null_mut(),
                shared,
                threads,
                scratch: [(0, 1); 2],
                owner: Arc::clone(&self.device),
            };
            self.device.check(
                unsafe { (self.device.function)(&mut program.entry, module, symbol.as_ptr()) },
                "장치 함수 연결",
            )?;
            self.device
                .configure_shared(program.entry, program.shared as usize)?;
            for (i, prefix) in ["global", "profile"].iter().enumerate() {
                program.scratch[i] = (
                    record[format!("{prefix}_scratch_size")]
                        .as_u64()
                        .context("scratch size")? as usize,
                    record[format!("{prefix}_scratch_align")]
                        .as_u64()
                        .context("scratch align")? as usize,
                );
            }
            self.programs.insert(key.clone(), program);
        }
        let program = &self.programs[&key];
        let mut cells: Vec<_> = arguments.into_iter().map(|a| a.cell).collect();
        let mut storage = Vec::new();
        for &(bytes, alignment) in &program.scratch {
            if bytes == 0 {
                cells.push(Cell::Address(std::ptr::null_mut()));
                continue;
            }
            ensure!(alignment.is_power_of_two(), "scratch 정렬값을 확인하세요.");
            let size = bytes
                .checked_mul(grid)
                .and_then(|n| n.checked_add(alignment))
                .context("scratch 크기")?;
            let t = self.reserve(size.div_ceil(4), DType::F32)?;
            cells.push(Cell::Address(
                (t.address as usize)
                    .div_ceil(alignment)
                    .checked_mul(alignment)
                    .context("scratch 주소")? as Pointer,
            ));
            storage.push(t);
        }
        let mut addresses: Vec<_> = cells.iter_mut().map(Cell::location).collect();
        self.device.check(
            unsafe {
                (self.device.launch)(
                    program.entry,
                    grid as u32,
                    1,
                    1,
                    program.threads,
                    1,
                    1,
                    program.shared,
                    std::ptr::null_mut(),
                    addresses.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            name,
        )
    }
    pub fn embedding(&mut self, weight: &Tensor, row: usize, width: usize) -> Result<Tensor> {
        ensure!(
            width > 0 && row < weight.len() / width,
            "embedding 행 범위를 확인하세요."
        );
        let out = self.reserve(width, DType::F32)?;
        self.dispatch(
            "take_row",
            json!({"WIDTH":width,"TILE":256}),
            width.div_ceil(256),
            vec![
                buffer("Matrix", weight),
                buffer("Output", &out),
                index("row", row),
            ],
        )?;
        Ok(out)
    }
    pub fn linear(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        ensure!(
            x.len() == cols
                && w.len() == rows.checked_mul(cols).context("행렬 크기")?
                && bias.is_none_or(|b| b.len() == rows),
            "선형 연산의 차원을 확인하세요."
        );
        let out = self.reserve(rows, DType::F32)?;
        self.dispatch(
            "matvec",
            json!({"ROWS":rows,"COLS":cols,"BIAS":bias.is_some(),"TILE_R":8,"TILE_C":256}),
            rows.div_ceil(8),
            vec![
                buffer("Matrix", w),
                buffer("Vector", x),
                buffer("Bias", bias.unwrap_or(x)),
                buffer("Output", &out),
            ],
        )?;
        Ok(out)
    }
    pub fn norm(
        &mut self,
        x: &Tensor,
        gain: &Tensor,
        bias: Option<&Tensor>,
        config: NormConfig,
    ) -> Result<Tensor> {
        let NormConfig {
            groups,
            width,
            epsilon,
            center,
        } = config;
        ensure!(
            groups.checked_mul(width) == Some(x.len())
                && width > 0
                && gain.len() == width
                && bias.is_none_or(|b| b.len() == width)
                && epsilon.is_finite()
                && epsilon > 0.,
            "정규화 차원을 확인하세요."
        );
        let out = self.reserve(x.len(), DType::F32)?;
        self.dispatch("normalization",json!({"WIDTH":width,"EPSILON":epsilon,"CENTER":center,"BIAS":bias.is_some(),"TILE":width.next_power_of_two()}),groups,vec![buffer("Input",x),buffer("Gain",gain),buffer("Bias",bias.unwrap_or(gain)),buffer("Output",&out)])?;
        Ok(out)
    }
    pub fn rope(&mut self, x: &Tensor, position: usize, config: RopeConfig) -> Result<Tensor> {
        let RopeConfig {
            heads,
            dim,
            theta,
            rotary,
            interleaved,
        } = config;
        ensure!(
            heads.checked_mul(dim) == Some(x.len())
                && rotary > 0
                && rotary <= dim
                && rotary.is_multiple_of(2)
                && position <= i32::MAX as usize
                && theta.is_finite()
                && theta > 0.,
            "RoPE 차원을 확인하세요."
        );
        let out = self.reserve(x.len(), DType::F32)?;
        self.dispatch("rotary_position",json!({"HEAD_DIM":dim,"WIDTH":x.len(),"ROTARY":rotary,"BASE":theta,"INTERLEAVED":interleaved,"TILE":256}),x.len().div_ceil(256),vec![buffer("Input",x),buffer("Output",&out),index("position",position)])?;
        Ok(out)
    }
    pub fn attention(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: (&Tensor, &Tensor),
        position: usize,
        config: AttentionConfig,
    ) -> Result<Tensor> {
        let AttentionConfig {
            qheads,
            kvheads,
            dim,
            scale,
        } = config;
        let (keys, values) = cache;
        let width = kvheads.checked_mul(dim).context("KV 차원 범위")?;
        ensure!(
            width > 0
                && qheads.checked_mul(dim) == Some(q.len())
                && k.len() == width
                && v.len() == width
                && qheads.is_multiple_of(kvheads)
                && position
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(width))
                    .is_some_and(|n| n <= keys.len())
                && keys.len() == values.len()
                && scale.is_finite()
                && scale > 0.,
            "attention 버퍼의 차원을 확인하세요."
        );
        self.dispatch(
            "cache_write",
            json!({"WIDTH":width,"TILE":256}),
            width.div_ceil(256),
            vec![
                buffer("Key", k),
                buffer("Value", v),
                buffer("Keys", keys),
                buffer("Values", values),
                index("position", position),
            ],
        )?;
        let out = self.reserve(q.len(), DType::F32)?;
        self.dispatch("causal_attention",json!({"Q_HEADS":qheads,"KV_HEADS":kvheads,"DIM":dim,"SCALE":scale,"TILE_T":32,"TILE_D":dim.next_power_of_two()}),qheads,vec![buffer("Query",q),buffer("Keys",keys),buffer("Values",values),buffer("Output",&out),index("position",position)])?;
        Ok(out)
    }
    pub fn gated(&mut self, gate: &Tensor, up: &Tensor, kind: u32) -> Result<Tensor> {
        ensure!(
            gate.len() == up.len() && kind <= 3,
            "MLP gate 조건을 확인하세요."
        );
        let out = self.reserve(gate.len(), DType::F32)?;
        self.dispatch(
            "gated_product",
            json!({"WIDTH":gate.len(),"KIND":kind,"TILE":256}),
            gate.len().div_ceil(256),
            vec![
                buffer("Gate", gate),
                buffer("Up", up),
                buffer("Output", &out),
            ],
        )?;
        Ok(out)
    }
    pub fn combine(&mut self, x: &Tensor, y: &Tensor, factor: f32) -> Result<Tensor> {
        ensure!(
            x.len() == y.len() && factor.is_finite(),
            "벡터 결합 조건을 확인하세요."
        );
        let out = self.reserve(x.len(), DType::F32)?;
        self.dispatch(
            "vector_sum",
            json!({"WIDTH":x.len(),"TILE":256}),
            x.len().div_ceil(256),
            vec![
                buffer("X", x),
                buffer("Y", y),
                buffer("Output", &out),
                scalar("factor", factor),
            ],
        )?;
        Ok(out)
    }
    pub fn rotate(&mut self, h: &Tensor, u: &Tensor, v: &Tensor, a: f32, b: f32) -> Result<Tensor> {
        ensure!(
            h.len() == u.len() && h.len() == v.len() && a.is_finite() && b.is_finite(),
            "출력 회전 조건을 확인하세요."
        );
        let out = self.reserve(h.len(), DType::F32)?;
        self.dispatch(
            "plane_rotation",
            json!({"WIDTH":h.len(),"TILE":h.len().next_power_of_two()}),
            1,
            vec![
                buffer("Hidden", h),
                buffer("U", u),
                buffer("V", v),
                buffer("Output", &out),
                scalar("cosine_minus_one", a),
                scalar("sine", b),
            ],
        )?;
        Ok(out)
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        let _ = self.synchronize();
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}
