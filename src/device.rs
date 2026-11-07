//! 드라이버 ABI와 장치 자원을 backend 경계 안에서 관리한다.
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    path::{Path, PathBuf},
    process::Command,
    ptr,
};
pub(crate) type Pointer = *mut c_void;
type Status = i32;
type Launch = unsafe extern "C" fn(
    Pointer,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Pointer,
    *mut Pointer,
    *mut Pointer,
) -> Status;
struct HipMemory {
    malloc: unsafe extern "C" fn(*mut Pointer, usize) -> Status,
    release: unsafe extern "C" fn(Pointer) -> Status,
    transfer: unsafe extern "C" fn(Pointer, *const c_void, usize, i32) -> Status,
    clear: unsafe extern "C" fn(Pointer, i32, usize) -> Status,
    message: unsafe extern "C" fn(Status) -> *const c_char,
}
struct CudaMemory {
    malloc: unsafe extern "C" fn(*mut u64, usize) -> Status,
    release: unsafe extern "C" fn(u64) -> Status,
    to_device: unsafe extern "C" fn(u64, *const c_void, usize) -> Status,
    to_host: unsafe extern "C" fn(Pointer, u64, usize) -> Status,
    clear: unsafe extern "C" fn(u64, u8, usize) -> Status,
    message: unsafe extern "C" fn(Status, *mut *const c_char) -> Status,
}
enum Allocator {
    Hip(HipMemory),
    Cuda(CudaMemory),
}
struct ContextLease {
    device: i32,
    release: unsafe extern "C" fn(i32) -> Status,
}
impl Drop for ContextLease {
    fn drop(&mut self) {
        unsafe {
            (self.release)(self.device);
        }
    }
}
pub(crate) struct Device {
    allocator: Allocator,
    pub synchronize: unsafe extern "C" fn() -> Status,
    pub load: unsafe extern "C" fn(*mut Pointer, *const c_char) -> Status,
    pub unload: unsafe extern "C" fn(Pointer) -> Status,
    pub function: unsafe extern "C" fn(*mut Pointer, Pointer, *const c_char) -> Status,
    pub launch: Launch,
    shared_attribute: Option<unsafe extern "C" fn(Pointer, i32, i32) -> Status>,
    shared_limit: usize,
    pub backend: &'static str,
    pub name: String,
    pub architecture: String,
    pub warp_size: usize,
    pub sdk_root: Option<PathBuf>,
    _lease: Option<ContextLease>,
    _driver: Library,
}
unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    Ok(unsafe { *library.get::<T>(name)? })
}
fn driver(candidates: &[PathBuf]) -> Result<Library> {
    for path in candidates {
        if let Ok(library) = unsafe { Library::new(path) } {
            return Ok(library);
        }
    }
    bail!("GPU 드라이버 라이브러리 경로를 확인하세요: {candidates:?}")
}
fn driver_paths(names: &[&str], sdk: Option<&Path>) -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("LAPLACE_DRIVER_LIBRARY") {
        return vec![path.into()];
    }
    names
        .iter()
        .flat_map(|name| {
            std::iter::once(PathBuf::from(name)).chain(sdk.map(|root| root.join("lib").join(name)))
        })
        .collect()
}
fn sdk(python: &Path) -> Option<PathBuf> {
    std::env::var_os("ROCM_PATH").or_else(||std::env::var_os("HIP_PATH")).map(PathBuf::from).or_else(||{
    let child=Command::new(python).args(["-B","-c","from importlib.metadata import distribution; print(distribution('rocm-sdk-core').locate_file('_rocm_sdk_core'))"]).output().ok()?;
    if child.status.success(){let text=String::from_utf8(child.stdout).ok()?;Some(PathBuf::from(text.trim()))}else{None}
 })
}
impl Device {
    pub fn open(python: &Path) -> Result<Self> {
        let ordinal = std::env::var("LAPLACE_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse::<i32>()?;
        ensure!(ordinal >= 0, "장치 번호는 0 이상이어야 합니다.");
        match std::env::var("LAPLACE_BACKEND")
            .as_deref()
            .unwrap_or("auto")
        {
            "hip" => Self::hip(ordinal, python),
            "cuda" => Self::cuda(ordinal),
            "auto" => Self::cuda(ordinal).or_else(|cuda| {
                Self::hip(ordinal, python)
                    .map_err(|hip| anyhow::anyhow!("장치 탐색 결과: CUDA {cuda:#}; HIP {hip:#}"))
            }),
            value => bail!("LAPLACE_BACKEND={value}: auto, hip, cuda 중 선택하세요."),
        }
    }
    fn cuda(ordinal: i32) -> Result<Self> {
        let library = driver(&driver_paths(&["libcuda.so.1", "libcuda.so"], None))?;
        unsafe {
            let init: unsafe extern "C" fn(u32) -> Status = symbol(&library, b"cuInit\0")?;
            let get: unsafe extern "C" fn(*mut i32, i32) -> Status =
                symbol(&library, b"cuDeviceGet\0")?;
            let retain: unsafe extern "C" fn(*mut Pointer, i32) -> Status =
                symbol(&library, b"cuDevicePrimaryCtxRetain\0")?;
            let release = symbol(&library, b"cuDevicePrimaryCtxRelease_v2\0")
                .or_else(|_| symbol(&library, b"cuDevicePrimaryCtxRelease\0"))?;
            let set: unsafe extern "C" fn(Pointer) -> Status =
                symbol(&library, b"cuCtxSetCurrent\0")?;
            let name_fn: unsafe extern "C" fn(*mut c_char, i32, i32) -> Status =
                symbol(&library, b"cuDeviceGetName\0")?;
            let attr: unsafe extern "C" fn(*mut i32, i32, i32) -> Status =
                symbol(&library, b"cuDeviceGetAttribute\0")?;
            ensure!(init(0) == 0, "CUDA 초기화 상태를 확인하세요.");
            let mut device = 0;
            ensure!(get(&mut device, ordinal) == 0, "CUDA 장치 번호 {ordinal}");
            let mut context = ptr::null_mut();
            ensure!(
                retain(&mut context, device) == 0,
                "CUDA primary context 획득"
            );
            let lease = ContextLease { device, release };
            ensure!(set(context) == 0, "CUDA context 선택");
            let mut name = [0 as c_char; 256];
            ensure!(
                name_fn(name.as_mut_ptr(), 256, device) == 0,
                "CUDA 장치명 조회"
            );
            let property = |key| -> Result<i32> {
                let mut value = 0;
                ensure!(attr(&mut value, key, device) == 0, "CUDA 속성 {key} 조회");
                Ok(value)
            };
            let architecture = (property(75)? * 10 + property(76)?).to_string();
            Ok(Self {
                allocator: Allocator::Cuda(CudaMemory {
                    malloc: symbol(&library, b"cuMemAlloc_v2\0")?,
                    release: symbol(&library, b"cuMemFree_v2\0")?,
                    to_device: symbol(&library, b"cuMemcpyHtoD_v2\0")?,
                    to_host: symbol(&library, b"cuMemcpyDtoH_v2\0")?,
                    clear: symbol(&library, b"cuMemsetD8_v2\0")?,
                    message: symbol(&library, b"cuGetErrorString\0")?,
                }),
                synchronize: symbol(&library, b"cuCtxSynchronize\0")?,
                load: symbol(&library, b"cuModuleLoad\0")?,
                unload: symbol(&library, b"cuModuleUnload\0")?,
                function: symbol(&library, b"cuModuleGetFunction\0")?,
                launch: symbol(&library, b"cuLaunchKernel\0")?,
                shared_attribute: Some(symbol(&library, b"cuFuncSetAttribute\0")?),
                shared_limit: property(8)? as usize,
                warp_size: property(10)? as usize,
                backend: "cuda",
                name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(),
                architecture,
                sdk_root: None,
                _lease: Some(lease),
                _driver: library,
            })
        }
    }
    fn hip(ordinal: i32, python: &Path) -> Result<Self> {
        let sdk_root = sdk(python);
        let library = driver(&driver_paths(
            &["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"],
            sdk_root.as_deref(),
        ))?;
        unsafe {
            let init: unsafe extern "C" fn(u32) -> Status = symbol(&library, b"hipInit\0")?;
            let select: unsafe extern "C" fn(i32) -> Status = symbol(&library, b"hipSetDevice\0")?;
            let get_name: unsafe extern "C" fn(*mut c_char, i32, i32) -> Status =
                symbol(&library, b"hipDeviceGetName\0")?;
            let pci_fn: unsafe extern "C" fn(*mut c_char, i32, i32) -> Status =
                symbol(&library, b"hipDeviceGetPCIBusId\0")?;
            ensure!(
                init(0) == 0 && select(ordinal) == 0,
                "HIP 장치 {ordinal} 초기화"
            );
            let mut name = [0 as c_char; 256];
            let mut pci = [0 as c_char; 32];
            ensure!(
                get_name(name.as_mut_ptr(), 256, ordinal) == 0
                    && pci_fn(pci.as_mut_ptr(), 32, ordinal) == 0,
                "HIP 장치 속성 조회"
            );
            let (architecture, warp_size) =
                agent_target(CStr::from_ptr(pci.as_ptr()).to_str()?, sdk_root.as_deref())?;
            Ok(Self {
                allocator: Allocator::Hip(HipMemory {
                    malloc: symbol(&library, b"hipMalloc\0")?,
                    release: symbol(&library, b"hipFree\0")?,
                    transfer: symbol(&library, b"hipMemcpy\0")?,
                    clear: symbol(&library, b"hipMemset\0")?,
                    message: symbol(&library, b"hipGetErrorString\0")?,
                }),
                synchronize: symbol(&library, b"hipDeviceSynchronize\0")?,
                load: symbol(&library, b"hipModuleLoad\0")?,
                unload: symbol(&library, b"hipModuleUnload\0")?,
                function: symbol(&library, b"hipModuleGetFunction\0")?,
                launch: symbol(&library, b"hipModuleLaunchKernel\0")?,
                shared_attribute: None,
                shared_limit: 0,
                backend: "hip",
                name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(),
                architecture,
                warp_size,
                sdk_root,
                _lease: None,
                _driver: library,
            })
        }
    }
    pub fn check(&self, status: Status, operation: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let description = unsafe {
            match &self.allocator {
                Allocator::Hip(api) => (api.message)(status),
                Allocator::Cuda(api) => {
                    let mut message = ptr::null();
                    (api.message)(status, &mut message);
                    message
                }
            }
        };
        let message = if description.is_null() {
            format!("status {status}")
        } else {
            unsafe { CStr::from_ptr(description) }
                .to_string_lossy()
                .into_owned()
        };
        bail!("{operation}: {message}")
    }
    pub fn allocate(&self, bytes: usize) -> Result<Pointer> {
        ensure!(bytes > 0, "GPU 할당 크기는 양수여야 합니다.");
        let mut pointer = ptr::null_mut();
        let status = unsafe {
            match &self.allocator {
                Allocator::Hip(api) => (api.malloc)(&mut pointer, bytes),
                Allocator::Cuda(api) => {
                    let mut address = 0;
                    {
                        let status = (api.malloc)(&mut address, bytes);
                        pointer = address as usize as Pointer;
                        status
                    }
                }
            }
        };
        self.check(status, "GPU 할당")?;
        Ok(pointer)
    }
    pub fn free(&self, pointer: Pointer) {
        unsafe {
            match &self.allocator {
                Allocator::Hip(api) => {
                    (api.release)(pointer);
                }
                Allocator::Cuda(api) => {
                    (api.release)(pointer as usize as u64);
                }
            }
        }
    }
    pub fn upload(&self, target: Pointer, source: *const c_void, bytes: usize) -> Result<()> {
        let status = unsafe {
            match &self.allocator {
                Allocator::Hip(api) => (api.transfer)(target, source, bytes, 1),
                Allocator::Cuda(api) => (api.to_device)(target as usize as u64, source, bytes),
            }
        };
        self.check(status, "CPU에서 GPU로 전송")
    }
    pub fn download(&self, target: Pointer, source: Pointer, bytes: usize) -> Result<()> {
        let status = unsafe {
            match &self.allocator {
                Allocator::Hip(api) => (api.transfer)(target, source, bytes, 2),
                Allocator::Cuda(api) => (api.to_host)(target, source as usize as u64, bytes),
            }
        };
        self.check(status, "GPU에서 CPU로 전송")
    }
    pub fn zero(&self, pointer: Pointer, bytes: usize) -> Result<()> {
        let status = unsafe {
            match &self.allocator {
                Allocator::Hip(api) => (api.clear)(pointer, 0, bytes),
                Allocator::Cuda(api) => (api.clear)(pointer as usize as u64, 0, bytes),
            }
        };
        self.check(status, "GPU 버퍼 초기화")
    }
    pub fn configure_shared(&self, entry: Pointer, bytes: usize) -> Result<()> {
        if let Some(attribute) = self.shared_attribute
            && bytes > self.shared_limit
        {
            self.check(
                unsafe { attribute(entry, 8, bytes.try_into()?) },
                "CUDA shared memory 크기 설정",
            )?;
        }
        Ok(())
    }
}
// HSA 공개 속성으로 PCI 주소를 맞춘 뒤 ISA와 wavefront 크기를 읽는다.
#[repr(C)]
#[derive(Clone, Copy)]
struct Agent(u64);
type Query = unsafe extern "C" fn(Agent, u32, Pointer) -> u32;
unsafe extern "C" fn append(agent: Agent, data: Pointer) -> u32 {
    unsafe {
        (*data.cast::<Vec<Agent>>()).push(agent);
    }
    0
}
fn agent_target(pci: &str, sdk: Option<&Path>) -> Result<(String, usize)> {
    let fields = pci
        .split([':', '.'])
        .map(|s| u32::from_str_radix(s, 16))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(fields.len() == 4, "장치 PCI 주소 {pci}");
    let candidates = std::iter::once(PathBuf::from("libhsa-runtime64.so.1"))
        .chain(sdk.map(|s| s.join("lib/libhsa-runtime64.so.1")))
        .collect::<Vec<_>>();
    let runtime = driver(&candidates)?;
    unsafe {
        let enumerate: unsafe extern "C" fn(
            unsafe extern "C" fn(Agent, Pointer) -> u32,
            Pointer,
        ) -> u32 = symbol(&runtime, b"hsa_iterate_agents\0")?;
        let info: Query = symbol(&runtime, b"hsa_agent_get_info\0")?;
        let list_isa: unsafe extern "C" fn(
            Agent,
            unsafe extern "C" fn(Agent, Pointer) -> u32,
            Pointer,
        ) -> u32 = symbol(&runtime, b"hsa_agent_iterate_isas\0")?;
        let isa_info: Query = symbol(&runtime, b"hsa_isa_get_info_alt\0")?;
        let mut agents = Vec::new();
        ensure!(
            enumerate(append, (&mut agents as *mut Vec<Agent>).cast()) == 0,
            "HSA 장치 열거"
        );
        for agent in agents {
            let read = |key| -> Result<u32> {
                let mut result = 0u32;
                ensure!(
                    info(agent, key, (&mut result as *mut u32).cast()) == 0,
                    "HSA 속성 {key}"
                );
                Ok(result)
            };
            if read(17)? != 1 {
                continue;
            }
            if read(0xA00F)? != fields[0]
                || read(0xA006)? != (fields[1] << 8 | fields[2] << 3 | fields[3])
            {
                continue;
            }
            let warp = read(6)?;
            let mut isas = Vec::new();
            ensure!(
                list_isa(agent, append, (&mut isas as *mut Vec<Agent>).cast()) == 0,
                "HSA ISA 열거"
            );
            let isa = *isas.first().context("장치 ISA")?;
            let mut length = 0u32;
            ensure!(
                isa_info(isa, 0, (&mut length as *mut u32).cast()) == 0
                    && length > 0
                    && length < 4096,
                "ISA 문자열 길이"
            );
            let mut bytes = vec![0u8; length as usize];
            ensure!(
                isa_info(isa, 1, bytes.as_mut_ptr().cast()) == 0,
                "ISA 문자열 조회"
            );
            let full = CStr::from_bytes_until_nul(&bytes)?.to_str()?;
            let architecture = full.rsplit("--").next().unwrap().split(':').next().unwrap();
            ensure!(
                architecture.starts_with("gfx") && warp > 0,
                "ISA 실행 대상 {full}"
            );
            return Ok((architecture.into(), warp as usize));
        }
    }
    bail!("PCI {pci}에 대응하는 HSA 장치를 조회해야 합니다.")
}
