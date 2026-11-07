//! 실제 실행에서 수집한 연산 계약을 다른 컴파일 대상으로 변환한다.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Command, Stdio},
};

#[test]
#[ignore = "먼저 GPU 검사를 실행해 LAPLACE_TEST_CACHE에 연산 계약을 수집한다"]
fn measured_operations_compile_for_cuda() -> Result<()> {
    let cache =
        PathBuf::from(std::env::var_os("LAPLACE_TEST_CACHE").context("LAPLACE_TEST_CACHE")?);
    let python = std::env::var_os("LAPLACE_PYTHON").unwrap_or_else(|| "python3".into());
    let target = cache.join("cuda-compilation");
    fs::create_dir_all(&target)?;
    fs::write(
        target.join("compile.py"),
        include_str!("../kernels/compile.py"),
    )?;
    fs::write(target.join("ops.py"), include_str!("../kernels/ops.py"))?;
    let mut requests = Vec::new();
    for entry in fs::read_dir(&cache)? {
        let path = entry?.path();
        if path.extension().and_then(|v| v.to_str()) != Some("json") {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(path)?)?;
        if value["request"]["backend"] == "hip" {
            requests.push(value["request"].clone());
        }
    }
    requests.sort_by_key(Value::to_string);
    requests.dedup();
    ensure!(
        !requests.is_empty(),
        "실제 GPU 실행의 컴파일 계약을 먼저 수집하세요."
    );
    let mut child = Command::new(python)
        .arg("-B")
        .arg(target.join("compile.py"))
        .env("TRITON_CACHE_DIR", target.join("triton"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let result = (|| -> Result<usize> {
        for request in &mut requests {
            request["backend"] = json!("cuda");
            // 이 검사의 명시적 컴파일 대상. 실제 실행은 드라이버가 조회한 capability를 쓴다.
            request["architecture"] = json!(80);
            request["warp_size"] = json!(32);
            request["cache"] = json!(target);
            writeln!(input, "{request}")?;
            input.flush()?;
            let mut line = String::new();
            ensure!(output.read_line(&mut line)? > 0, "컴파일 응답");
            let reply: Value = serde_json::from_str(&line)?;
            ensure!(
                reply.get("error").is_none(),
                "{}: {reply}",
                request["operation"]
            );
            let binary = PathBuf::from(reply["binary"].as_str().context("cubin")?);
            ensure!(
                binary.extension().and_then(|v| v.to_str()) == Some("cubin")
                    && fs::metadata(binary)?.len() > 0,
                "CUDA code object 생성"
            );
        }
        Ok(requests.len())
    })();
    drop(input);
    let status = child.wait()?;
    println!("CUDA code objects compiled: {}", result?);
    ensure!(status.success(), "컴파일러 종료 상태");
    Ok(())
}
