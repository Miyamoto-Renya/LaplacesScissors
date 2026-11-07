//! 원본 채팅 양식과 고정 prompt token 접두사에서 정답 문자열을 준비한다.
use anyhow::{Context, Result, ensure};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};
use tokenizers::Tokenizer;

pub struct EncodedTarget {
    pub prompt: Vec<u32>,
    pub continuation: Vec<u32>,
}
pub struct Text {
    tokenizer: Tokenizer,
    templates: Environment<'static>,
    special: serde_json::Map<String, Value>,
    thinking: bool,
    pub stop_tokens: BTreeSet<u32>,
    primary_stop: Option<u32>,
    pub template_source: String,
}
fn json_file(path: &Path) -> Result<Value> {
    if path.exists() {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    } else {
        Ok(json!({}))
    }
}
fn stops(value: &Value) -> Result<Vec<u32>> {
    let values = if value.is_null() {
        Vec::new()
    } else if let Some(xs) = value.as_array() {
        xs.clone()
    } else {
        vec![value.clone()]
    };
    values
        .iter()
        .map(|v| {
            u32::try_from(v.as_u64().context("EOS token의 정수 id")?).context("EOS token 범위")
        })
        .collect()
}
fn stop_configuration(generation: &Value, model: &Value, graph: &Value) -> Result<Vec<u32>> {
    for value in [
        &generation["eos_token_id"],
        &model["eos_token_id"],
        &graph["eos_token_ids"],
    ] {
        let selected = stops(value)?;
        if !selected.is_empty() {
            return Ok(selected);
        }
    }
    Ok(Vec::new())
}
fn executable_template(source: &str) -> String {
    let mut tail = source;
    let mut result = String::new();
    while let Some(start) = tail.find("{%") {
        result.push_str(&tail[..start]);
        let block = &tail[start..];
        let Some(end) = block.find("%}") else {
            result.push_str(block);
            return result;
        };
        let instruction = block[2..end].trim().trim_matches(['-', '+']).trim();
        match instruction {
            "generation" => result.push_str(&block[..end + 2].replacen("generation", "if true", 1)),
            "endgeneration" => {
                result.push_str(&block[..end + 2].replacen("endgeneration", "endif", 1))
            }
            _ => result.push_str(&block[..end + 2]),
        }
        tail = &block[end + 2..];
    }
    result.push_str(tail);
    result
}
impl Text {
    pub fn load(root: &Path, thinking: bool) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("토크나이저 적재: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("token 길이 설정: {e}"))?;
        let configuration = json_file(&root.join("tokenizer_config.json"))?;
        let separate = root.join("chat_template.jinja");
        let (body, template_source) = if separate.exists() {
            (
                std::fs::read_to_string(separate)?,
                "chat_template.jinja".into(),
            )
        } else {
            let selected = &configuration["chat_template"];
            if let Some(body) = selected.as_str() {
                (
                    body.to_owned(),
                    "tokenizer_config.json:chat_template".into(),
                )
            } else if let Some(list) = selected.as_array() {
                (
                    list.iter()
                        .find(|x| x["name"] == "default")
                        .and_then(|x| x["template"].as_str())
                        .context("default 채팅 template")?
                        .to_owned(),
                    "tokenizer_config.json:default".into(),
                )
            } else {
                ("{{ messages[0].content }}".into(), "raw completion".into())
            }
        };
        let mut templates = Environment::new();
        templates.set_trim_blocks(true);
        templates.set_lstrip_blocks(true);
        templates.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        templates.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        templates.add_template_owned("prompt", executable_template(&body))?;
        let mut special = serde_json::Map::new();
        if let Some(object) = configuration.as_object() {
            for (key, value) in object {
                if key.ends_with("_token") {
                    special.insert(key.clone(), value.get("content").unwrap_or(value).clone());
                }
            }
        }
        let generation = json_file(&root.join("generation_config.json"))?;
        let model = json_file(&root.join("config.json"))?;
        let graph = json_file(&root.join("laplace.graph.json"))?;
        let selected = stop_configuration(&generation, &model, &graph)?;
        let primary_stop = selected.first().copied();
        let stop_tokens = selected.into_iter().collect();
        Ok(Self {
            tokenizer,
            templates,
            special,
            thinking,
            stop_tokens,
            primary_stop,
            template_source,
        })
    }
    pub fn render(&self, prompt: &str) -> Result<String> {
        let mut context = self.special.clone();
        context.insert("messages".into(), json!([{"role":"user","content":prompt}]));
        context.insert("add_generation_prompt".into(), json!(true));
        context.insert("enable_thinking".into(), json!(self.thinking));
        context.insert("tools".into(), Value::Null);
        Ok(self
            .templates
            .get_template("prompt")?
            .render(Value::Object(context))?)
    }
    pub fn encode(&self, value: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(value, false)
            .map_err(|e| anyhow::anyhow!("token 인코딩: {e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("token 디코딩: {e}"))
    }
    pub fn target(&self, prompt: &str, answer: &str) -> Result<EncodedTarget> {
        let prompt = self.encode(&self.render(prompt)?)?;
        let mut continuation = self.encode(answer)?;
        ensure!(
            !prompt.is_empty() && !continuation.is_empty(),
            "입력과 정답에 token이 필요합니다."
        );
        ensure!(
            self.decode(&continuation)? == answer,
            "정답 문자열의 tokenizer 왕복 결과를 확인하세요."
        );
        ensure!(
            continuation.iter().all(|t| !self.stop_tokens.contains(t)),
            "정답 본문은 종료 token 앞의 문자열로 지정하세요."
        );
        continuation.push(
            self.primary_stop
                .context("정답 생성 계약에 EOS token이 필요합니다.")?,
        );
        Ok(EncodedTarget {
            prompt,
            continuation,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generation_markers_keep_content() {
        let mut environment = Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        for (source, expected) in [
            ("x{% generation %}y{%- endgeneration -%}z", "xyz"),
            (
                "first \n{%- generation -%} body {%- endgeneration -%}\nlast",
                "firstbodylast",
            ),
        ] {
            let result = environment
                .render_str(&executable_template(source), ())
                .unwrap();
            assert_eq!(result, expected);
        }
    }
    #[test]
    fn stop_order_and_ranges() {
        assert_eq!(stops(&json!([9, 3])).unwrap(), vec![9, 3]);
        assert!(stops(&json!(-1)).is_err());
        assert!(stops(&json!(4294967296u64)).is_err());
    }
    #[test]
    fn generation_eos_overrides_model_and_graph_defaults() {
        let model = json!({"eos_token_id":9});
        let graph = json!({"eos_token_ids":[7,8]});
        assert_eq!(
            stop_configuration(&json!({"eos_token_id":[3,4]}), &model, &graph).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            stop_configuration(&Value::Null, &model, &graph).unwrap(),
            vec![9]
        );
        assert_eq!(
            stop_configuration(&Value::Null, &Value::Null, &graph).unwrap(),
            vec![7, 8]
        );
    }
}
