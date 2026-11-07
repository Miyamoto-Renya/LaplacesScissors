//! 저장된 회전을 독립 출력 행렬로 계산해 새 checkpoint에 기록한다.
use crate::{
    diagnosis::{self, Proposal},
    graph::{GraphSpec, TensorShape},
    hub::ModelFiles,
    model::Model,
    response::ReadoutEdit,
};
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use nalgebra::DMatrix;
use safetensors::{Dtype, SafeTensors, serialize_to_file, tensor::TensorView};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    path::Path,
    time::Instant,
};

const HEAD: &str = "lm_head.weight";

fn factors(edit: &ReadoutEdit, width: usize) -> Result<(DMatrix<f64>, DMatrix<f64>)> {
    edit.validate(width)?;
    let subspace = edit.subspace();
    let rank = subspace.basis.len();
    let basis = DMatrix::from_fn(width, rank, |i, j| subspace.basis[j][i]);
    let right = (subspace.matrix() - DMatrix::identity(rank, rank)) * basis.transpose();
    Ok((basis, right))
}

fn rotate_rows(w: &DMatrix<f64>, basis: &DMatrix<f64>, right: &DMatrix<f64>) -> DMatrix<f64> {
    w + (w * basis) * right
}

fn independent_head(graph: &GraphSpec) -> bool {
    graph.embedding != HEAD
        && graph.blocks.iter().all(|b| {
            [
                &b.attention.query,
                &b.attention.key,
                &b.attention.value,
                &b.attention.output,
                &b.mlp.gate,
                &b.mlp.up,
                &b.mlp.down,
            ]
            .iter()
            .all(|p| p.weight != HEAD)
        })
}

pub fn export(
    files: &ModelFiles,
    model: &Model,
    proposal: &Proposal,
    output: &Path,
) -> Result<Value> {
    proposal.validate(files, model)?;
    ensure!(
        independent_head(model.graph()),
        "lm_head.weight는 독립 출력 행렬 이름으로 사용합니다."
    );
    ensure!(!output.exists(), "새 checkpoint 폴더를 지정하세요.");
    let parent = output.parent().context("checkpoint 부모 경로")?;
    fs::create_dir_all(parent)?;
    let name = output
        .file_name()
        .context("checkpoint 폴더 이름")?
        .to_string_lossy();
    let staging = parent.join(format!(".{name}.materializing-{}", std::process::id()));
    fs::create_dir(&staging)?;
    let started = Instant::now();
    let calculation = (|| -> Result<Value> {
        let mut graph = model.graph().clone();
        graph.readout.weight = HEAD.into();
        graph.source = "materialized_readout_rotation".into();
        let mut weight_map = BTreeMap::<String, String>::new();
        let mut shapes = BTreeMap::new();
        let mut total_bytes = 0usize;
        for (i, receipt) in files
            .files
            .iter()
            .filter(|f| f.name.ends_with(".safetensors"))
            .enumerate()
        {
            let source = files.root.join(&receipt.name);
            let map = unsafe { Mmap::map(&File::open(&source)?)? };
            let archive = SafeTensors::deserialize(&map)?;
            let tensors: Vec<_> = archive
                .tensors()
                .into_iter()
                .filter(|(name, _)| name != HEAD)
                .collect();
            if tensors.is_empty() {
                continue;
            }
            let shard = format!("model-part-{:05}.safetensors", i + 1);
            if tensors.len() == archive.len() {
                fs::copy(&source, staging.join(&shard))?;
            } else {
                serialize_to_file(
                    tensors.iter().map(|(name, tensor)| (name, tensor.clone())),
                    Some(HashMap::from([("format".into(), "pt".into())])),
                    &staging.join(&shard),
                )?;
            }
            for (name, tensor) in tensors {
                total_bytes = total_bytes
                    .checked_add(tensor.data().len())
                    .context("checkpoint 크기")?;
                shapes.insert(
                    name.clone(),
                    TensorShape {
                        shape: tensor.shape().to_vec(),
                        dtype: format!("{:?}", tensor.dtype()),
                    },
                );
                weight_map.insert(name, shard.clone());
            }
        }

        let source = model.matrix(&proposal.readout_weight)?;
        let (basis, right) = factors(&proposal.rotation, source.cols)?;
        let cells = source
            .rows
            .checked_mul(source.cols)
            .context("출력 행렬 크기")?;
        let byte_count = cells.checked_mul(4).context("FP32 출력 크기")?;
        let mut bytes = Vec::with_capacity(byte_count);
        let mut original = source.values();
        let mut squared_change = 0.;
        let mut rounding_error = 0f64;
        for first in (0..source.rows).step_by(256) {
            let rows = (source.rows - first).min(256);
            let block = DMatrix::from_row_iterator(
                rows,
                source.cols,
                original.by_ref().take(rows * source.cols),
            );
            let edited = rotate_rows(&block, &basis, &right);
            for i in 0..rows {
                for j in 0..source.cols {
                    let value = edited[(i, j)] as f32;
                    ensure!(value.is_finite(), "수정 출력 행렬의 유한값을 확인하세요.");
                    squared_change += (f64::from(value) - block[(i, j)]).powi(2);
                    rounding_error = rounding_error.max((f64::from(value) - edited[(i, j)]).abs());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        let shard = "model-readout.safetensors";
        serialize_to_file(
            [(
                HEAD,
                TensorView::new(Dtype::F32, vec![source.rows, source.cols], &bytes)?,
            )],
            Some(HashMap::from([("format".into(), "pt".into())])),
            &staging.join(shard),
        )?;
        drop(bytes);
        weight_map.insert(HEAD.into(), shard.into());
        shapes.insert(
            HEAD.into(),
            TensorShape {
                shape: vec![source.rows, source.cols],
                dtype: "F32".into(),
            },
        );
        total_bytes = total_bytes
            .checked_add(byte_count)
            .context("checkpoint 크기")?;
        for receipt in &files.files {
            if receipt.name.ends_with(".safetensors")
                || receipt.name.ends_with(".safetensors.index.json")
                || matches!(receipt.name.as_str(), "config.json" | "laplace.graph.json")
            {
                continue;
            }
            let path = staging.join(&receipt.name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(files.root.join(&receipt.name), path)?;
        }
        let mut config: Value = serde_json::from_slice(&fs::read(files.root.join("config.json"))?)?;
        config["tie_word_embeddings"] = json!(false);
        config["torch_dtype"] = json!("float32");
        config["dtype"] = json!("float32");
        diagnosis::save(&staging.join("config.json"), &config)?;
        diagnosis::save(&staging.join("laplace.graph.json"), &graph)?;
        diagnosis::save(
            &staging.join("model.safetensors.index.json"),
            &json!({"metadata":{"total_size":total_bytes},"weight_map":weight_map}),
        )?;
        GraphSpec::load(&staging, &shapes)?;
        let report = json!({"version":1,"phase":"complete","source":files,"proposal":proposal,
            "output":output,"readout_weight":HEAD,"readout_shape":[source.rows,source.cols],
            "readout_dtype":"F32","decoder_storage":"original tensor bytes",
            "application":"materialized W + (W U)(R - I) U^T; independent readout",
            "computation":"CPU F64 block matrix products, F32 storage rounding",
            "actual_weight_delta_norm":squared_change.sqrt(),"maximum_storage_rounding_error":rounding_error,
            "total_tensor_bytes":total_bytes,"elapsed_seconds":started.elapsed().as_secs_f64()});
        diagnosis::save(&staging.join("materialization.json"), &report)?;
        ensure!(!output.exists(), "새 checkpoint 폴더를 지정하세요.");
        fs::rename(&staging, output)?;
        Ok(report)
    })();
    if calculation.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    calculation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::ReadoutRotation;

    #[test]
    fn stored_rows_match_rotated_hidden() {
        let plane = ReadoutRotation {
            u: vec![0.6, 0.8, 0., 0.],
            v: vec![0., 0., 1., 0.],
            tangent: 0.37,
        };
        let edits = [
            ReadoutEdit::Plane(plane.clone()),
            ReadoutEdit::Joint(ReadoutEdit::Plane(plane).subspace()),
        ];
        let w = DMatrix::from_row_slice(
            3,
            4,
            &[
                0.25, -1.5, 2., 0.1, -0.75, 0.4, 1.2, 3., 0.9, 0.2, -0.3, 0.8,
            ],
        );
        let h = nalgebra::DVector::from_vec(vec![0.3, -0.7, 1.1, 0.4]);
        for edit in edits {
            let (basis, right) = factors(&edit, 4).unwrap();
            let stored = rotate_rows(&w, &basis, &right).map(|v| f64::from(v as f32));
            let rotated = edit.subspace().apply(h.as_slice());
            let expected = &w * nalgebra::DVector::from_vec(rotated);
            assert!((stored * &h - expected).amax() < 1e-6);
        }
    }
}
