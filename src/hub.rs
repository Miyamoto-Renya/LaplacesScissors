//! 모델 revision과 입력 파일의 내용을 해시로 고정한다.
use anyhow::{Context, Result, ensure};
use reqwest::{Url, blocking::Client};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    time::Duration,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileReceipt {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFiles {
    pub root: PathBuf,
    pub repository: Option<String>,
    pub revision: Option<String>,
    pub files: Vec<FileReceipt>,
}
const CONFIGS: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "laplace.graph.json",
];
fn relative(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && !name.contains('\\')
            && Path::new(name)
                .components()
                .all(|x| matches!(x, Component::Normal(_))),
        "모델 내부 상대 경로를 지정하세요: {name}"
    );
    Ok(())
}
fn receipt(root: &Path, name: &str) -> Result<FileReceipt> {
    relative(name)?;
    let mut stream = File::open(root.join(name))?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut chunk = vec![0u8; 1024 * 1024];
    loop {
        let size = stream.read(&mut chunk)?;
        if size == 0 {
            break;
        }
        hash.update(&chunk[..size]);
        bytes += size as u64;
    }
    Ok(FileReceipt {
        name: name.into(),
        bytes,
        sha256: hex::encode(hash.finalize()),
    })
}
fn choose_index(names: &BTreeSet<String>) -> Result<Option<String>> {
    if names.contains("model.safetensors.index.json") {
        return Ok(Some("model.safetensors.index.json".into()));
    }
    let found = names
        .iter()
        .filter(|s| s.ends_with(".safetensors.index.json"))
        .collect::<Vec<_>>();
    ensure!(
        found.len() <= 1,
        "가중치 index가 하나인 모델 폴더를 지정하세요."
    );
    Ok(found.first().map(|s| (*s).clone()))
}
fn shards(index: &Value) -> Result<BTreeSet<String>> {
    index["weight_map"]
        .as_object()
        .context("가중치 index의 weight_map")?
        .values()
        .map(|v| {
            let name = v.as_str().context("가중치 shard 파일명")?;
            relative(name)?;
            ensure!(
                name.ends_with(".safetensors"),
                "safetensors shard를 지정하세요."
            );
            Ok(name.to_owned())
        })
        .collect()
}
fn choose_single(names: &BTreeSet<String>, selected: Option<&str>) -> Result<BTreeSet<String>> {
    let chosen = selected
        .filter(|n| n.ends_with(".safetensors"))
        .or_else(|| {
            names
                .contains("model.safetensors")
                .then_some("model.safetensors")
        });
    if let Some(name) = chosen {
        ensure!(names.contains(name), "가중치 파일 {name}");
        return Ok(BTreeSet::from([name.into()]));
    }
    let candidates = names
        .iter()
        .filter(|n| n.ends_with(".safetensors") && n.as_str() != "adapter_model.safetensors")
        .cloned()
        .collect::<BTreeSet<_>>();
    ensure!(
        candidates.len() == 1,
        "전체 가중치 파일 또는 shard index를 선택하세요: {candidates:?}"
    );
    Ok(candidates)
}
pub fn weight_files(root: &Path) -> Result<BTreeSet<String>> {
    let names = fs::read_dir(root)?
        .map(|e| Ok(e?.file_name().to_string_lossy().into_owned()))
        .collect::<Result<BTreeSet<_>>>()?;
    match choose_index(&names)? {
        Some(index) => shards(&serde_json::from_slice(&fs::read(root.join(index))?)?),
        None => choose_single(&names, None),
    }
}
pub(crate) fn computational_file(name: &str) -> bool {
    matches!(name, "config.json" | "laplace.graph.json")
        || name.ends_with(".safetensors")
        || name.ends_with(".safetensors.index.json")
}
fn local(root: &Path, text_assets: bool) -> Result<ModelFiles> {
    let root = root.canonicalize()?;
    ensure!(
        root.join("config.json").is_file() || root.join("laplace.graph.json").is_file(),
        "모델 설정 파일이 있는 폴더를 지정하세요."
    );
    let mut names = weight_files(&root)?;
    for name in CONFIGS {
        if (text_assets || computational_file(name)) && root.join(name).is_file() {
            names.insert((*name).into());
        }
    }
    let listed = fs::read_dir(&root)?
        .map(|e| Ok(e?.file_name().to_string_lossy().into_owned()))
        .collect::<Result<BTreeSet<_>>>()?;
    if let Some(index) = choose_index(&listed)? {
        names.insert(index);
    }
    let files = names
        .iter()
        .map(|name| receipt(&root, name))
        .collect::<Result<Vec<_>>>()?;
    let origin_name = if text_assets {
        "laplace.source.json"
    } else {
        "laplace.weights-source.json"
    };
    let origin = fs::read(root.join(origin_name))
        .or_else(|_| fs::read(root.join("laplace.source.json")))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<ModelFiles>(&bytes).ok());
    let source = origin.filter(|old| {
        !old.files.is_empty()
            && (!text_assets || old.files.len() == files.len())
            && files.iter().all(|a| {
                old.files
                    .iter()
                    .any(|b| a.name == b.name && a.sha256 == b.sha256 && a.bytes == b.bytes)
            })
    });
    Ok(ModelFiles {
        root,
        repository: source.as_ref().and_then(|s| s.repository.clone()),
        revision: source.and_then(|s| s.revision),
        files,
    })
}
struct Remote {
    repository: String,
    revision: String,
    folder: String,
    selected: Option<String>,
}
fn unescape(input: &str) -> Result<String> {
    let mut bytes = Vec::new();
    let mut i = 0;
    while i < input.len() {
        if input.as_bytes()[i] == b'%' {
            ensure!(i + 2 < input.len(), "URL 문자 인코딩");
            bytes.push(u8::from_str_radix(&input[i + 1..i + 3], 16)?);
            i += 3;
        } else {
            bytes.push(input.as_bytes()[i]);
            i += 1;
        }
    }
    Ok(String::from_utf8(bytes)?)
}
impl Remote {
    fn parse(input: &str, override_revision: Option<&str>) -> Result<Self> {
        let mut folder = String::new();
        let mut selected = None;
        let (repository, embedded) = if input.starts_with("https://") {
            let url = Url::parse(input)?;
            ensure!(
                url.host_str() == Some("huggingface.co") && url.username().is_empty(),
                "Hugging Face 모델 URL을 지정하세요."
            );
            let parts = url
                .path_segments()
                .context("모델 URL 경로")?
                .filter(|s| !s.is_empty())
                .map(unescape)
                .collect::<Result<Vec<_>>>()?;
            ensure!(parts.len() >= 2, "모델 URL의 조직/모델 경로");
            let repo = format!("{}/{}", parts[0], parts[1]);
            let revision = if parts.len() > 2 {
                ensure!(
                    parts.len() >= 4 && matches!(parts[2].as_str(), "tree" | "blob" | "resolve"),
                    "모델 tree, blob, resolve URL을 지정하세요."
                );
                let mut remaining = parts[4..].to_vec();
                if parts[2] != "tree" {
                    selected = remaining.pop();
                    ensure!(selected.is_some(), "선택한 모델 파일명");
                }
                folder = remaining.join("/");
                Some(parts[3].clone())
            } else {
                None
            };
            (repo, revision)
        } else {
            let (repo, rev) = input
                .rsplit_once('@')
                .map_or((input, None), |(r, v)| (r, Some(v.to_owned())));
            (repo.to_owned(), rev)
        };
        let pieces = repository.split('/').collect::<Vec<_>>();
        ensure!(
            pieces.len() == 2
                && pieces.iter().all(|s| !s.is_empty()
                    && *s != "."
                    && *s != ".."
                    && s.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))),
            "Hugging Face 조직/모델 이름을 확인하세요."
        );
        if !folder.is_empty() {
            relative(&folder)?;
        }
        if let Some(name) = &selected {
            relative(name)?;
            ensure!(!name.contains('/'), "선택한 파일명을 확인하세요.");
        }
        let revision = override_revision
            .map(str::to_owned)
            .or(embedded)
            .unwrap_or_else(|| "main".into());
        ensure!(!revision.is_empty(), "모델 revision을 지정하세요.");
        Ok(Self {
            repository,
            revision,
            folder,
            selected,
        })
    }
    fn url(&self, revision: &str, file: Option<&str>) -> Result<Url> {
        let mut url = Url::parse("https://huggingface.co")?;
        {
            let mut parts = url
                .path_segments_mut()
                .map_err(|_| anyhow::anyhow!("Hub URL"))?;
            if file.is_none() {
                parts.extend(["api", "models"]);
            }
            parts.extend(self.repository.split('/'));
            parts
                .push(if file.is_some() {
                    "resolve"
                } else {
                    "revision"
                })
                .push(revision);
            if let Some(file) = file {
                parts.extend(file.split('/'));
            }
        }
        if file.is_none() {
            url.set_query(Some("blobs=true"));
        }
        Ok(url)
    }
}
fn request(client: &Client, url: Url) -> reqwest::blocking::RequestBuilder {
    let builder = client.get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder.bearer_auth(token)
    } else {
        builder
    }
}
fn obtain(
    client: &Client,
    remote: &Remote,
    commit: &str,
    root: &Path,
    metadata: &BTreeMap<String, Value>,
    name: &str,
) -> Result<FileReceipt> {
    relative(name)?;
    let meta = metadata
        .get(name)
        .with_context(|| format!("Hub 파일 {name}"))?;
    let expected_hash = meta["lfs"]["sha256"].as_str();
    let expected_bytes = meta["size"]
        .as_u64()
        .or_else(|| meta["lfs"]["size"].as_u64());
    let matches = |r: &FileReceipt| {
        expected_hash.is_none_or(|h| h == r.sha256) && expected_bytes.is_none_or(|n| n == r.bytes)
    };
    if root.join(name).is_file() {
        let cached = receipt(root, name)?;
        if matches(&cached) {
            return Ok(cached);
        }
    }
    let path = root.join(name);
    fs::create_dir_all(path.parent().context("가중치 폴더")?)?;
    let remote_name = if remote.folder.is_empty() {
        name.to_owned()
    } else {
        format!("{}/{name}", remote.folder)
    };
    eprintln!("download {}/{}", remote.repository, remote_name);
    let mut response = request(client, remote.url(commit, Some(&remote_name))?)
        .send()?
        .error_for_status()?;
    let temporary = path.with_extension(format!("{}.partial", std::process::id()));
    let result = (|| -> Result<FileReceipt> {
        let mut output = File::create(&temporary)?;
        let mut hash = Sha256::new();
        let mut bytes = 0;
        let mut chunk = vec![0u8; 1024 * 1024];
        loop {
            let size = response.read(&mut chunk)?;
            if size == 0 {
                break;
            }
            output.write_all(&chunk[..size])?;
            hash.update(&chunk[..size]);
            bytes += size as u64;
        }
        output.sync_all()?;
        let file = FileReceipt {
            name: name.into(),
            bytes,
            sha256: hex::encode(hash.finalize()),
        };
        ensure!(
            matches(&file),
            "다운로드한 {name}의 크기와 SHA256을 확인하세요."
        );
        fs::rename(&temporary, &path)?;
        Ok(file)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
pub fn resolve(input: &str, revision: Option<&str>, cache: &Path) -> Result<ModelFiles> {
    resolve_assets(input, revision, cache, true)
}
pub fn resolve_weights(input: &str, revision: Option<&str>, cache: &Path) -> Result<ModelFiles> {
    resolve_assets(input, revision, cache, false)
}
fn resolve_assets(
    input: &str,
    revision: Option<&str>,
    cache: &Path,
    text_assets: bool,
) -> Result<ModelFiles> {
    let path = Path::new(input);
    if path.exists() {
        return local(
            if path.is_dir() {
                path
            } else {
                path.parent().context("모델 폴더")?
            },
            text_assets,
        );
    }
    let remote = Remote::parse(input, revision)?;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(3600))
        .user_agent("laplaces-scissors/0.2")
        .build()?;
    let info: Value = request(&client, remote.url(&remote.revision, None)?)
        .send()?
        .error_for_status()?
        .json()?;
    let commit = info["sha"].as_str().context("Hub commit SHA")?;
    ensure!(
        matches!(commit.len(), 40 | 64) && commit.bytes().all(|b| b.is_ascii_hexdigit()),
        "Hub commit SHA 형식을 확인하세요."
    );
    let prefix = if remote.folder.is_empty() {
        String::new()
    } else {
        format!("{}/", remote.folder)
    };
    let metadata = info["siblings"]
        .as_array()
        .context("Hub 파일 목록")?
        .iter()
        .filter_map(|v| {
            v["rfilename"]
                .as_str()
                .and_then(|s| s.strip_prefix(&prefix))
                .map(|s| (s.to_owned(), v.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let names = metadata.keys().cloned().collect::<BTreeSet<_>>();
    let root = cache
        .join("models")
        .join(remote.repository.replace('/', "--"))
        .join(commit)
        .join(&remote.folder);
    fs::create_dir_all(&root)?;
    let mut files = Vec::new();
    for &name in CONFIGS {
        if (text_assets || computational_file(name)) && names.contains(name) {
            files.push(obtain(&client, &remote, commit, &root, &metadata, name)?);
        }
    }
    ensure!(
        (!text_assets || root.join("tokenizer.json").is_file())
            && (root.join("config.json").is_file() || root.join("laplace.graph.json").is_file()),
        "실행 모드에 필요한 모델 설정과 입력 자산을 포함한 경로를 지정하세요."
    );
    let weights = match choose_index(&names)? {
        Some(index) => {
            files.push(obtain(&client, &remote, commit, &root, &metadata, &index)?);
            shards(&serde_json::from_slice(&fs::read(root.join(index))?)?)?
        }
        None => choose_single(&names, remote.selected.as_deref())?,
    };
    for name in weights {
        files.push(obtain(&client, &remote, commit, &root, &metadata, &name)?);
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    let result = ModelFiles {
        root,
        repository: Some(remote.repository),
        revision: Some(commit.into()),
        files,
    };
    let target = result.root.join(if text_assets {
        "laplace.source.json"
    } else {
        "laplace.weights-source.json"
    });
    let temporary = target.with_extension("partial");
    fs::write(&temporary, serde_json::to_vec_pretty(&result)?)?;
    fs::rename(temporary, target)?;
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_source_contract() {
        let source = Remote::parse(
            "https://huggingface.co/org/model/blob/abc/sub/model.safetensors",
            None,
        )
        .unwrap();
        assert_eq!(source.repository, "org/model");
        assert_eq!(source.revision, "abc");
        assert_eq!(source.folder, "sub");
        assert_eq!(source.selected.as_deref(), Some("model.safetensors"));
        assert!(relative("../escape").is_err());
        assert!(Remote::parse("https://example.com/org/model", None).is_err());
        assert_eq!(
            Remote::parse("org/model@branch", Some("commit"))
                .unwrap()
                .revision,
            "commit"
        );
    }
    #[test]
    fn shards_keep_validated_relative_names() {
        assert!(shards(&serde_json::json!({"weight_map":{"a":"../x.safetensors"}})).is_err());
        let names = BTreeSet::from(["a.safetensors".into(), "b.safetensors".into()]);
        assert!(choose_single(&names, None).is_err());
    }
}
