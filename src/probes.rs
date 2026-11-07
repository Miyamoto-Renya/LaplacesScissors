//! 정답 문자열과 정답 근거를 가진 진단 입력.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashSet, path::Path};
fn family() -> String {
    "user".into()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    #[serde(default = "family")]
    pub family: String,
    pub prompt: String,
    pub choices: Vec<String>,
    #[serde(alias = "expected_index")]
    pub expected: usize,
    #[serde(default)]
    pub oracle: Value,
}
impl Case {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.is_empty()
                && !self.prompt.is_empty()
                && self.choices.len() >= 2
                && self.expected < self.choices.len(),
            "사례 id·입력·선택지·정답 위치를 확인하세요."
        );
        let unique: HashSet<_> = self.choices.iter().collect();
        ensure!(
            unique.len() == self.choices.len() && self.choices.iter().all(|s| !s.is_empty()),
            "선택지에는 서로 다른 문자열을 지정하세요."
        );
        Ok(())
    }
    pub fn answer(&self) -> &str {
        &self.choices[self.expected]
    }
}
pub fn load_cases(path: &Path) -> Result<Vec<Case>> {
    let content = std::fs::read_to_string(path)?;
    let mut result = Vec::new();
    let mut used = HashSet::new();
    for (line, text) in content
        .lines()
        .enumerate()
        .filter(|(_, s)| !s.trim().is_empty())
    {
        let case: Case = serde_json::from_str(text)
            .with_context(|| format!("{}:{} 사례", path.display(), line + 1))?;
        case.validate()?;
        ensure!(
            used.insert(case.id.clone()),
            "사례 id가 중복됐습니다: {}",
            case.id
        );
        result.push(case);
    }
    ensure!(!result.is_empty(), "한 개 이상의 사례를 지정하세요.");
    Ok(result)
}
struct Sequence(u64);
impl Sequence {
    fn pick(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}
pub fn generate(count: usize, seed: u64) -> Vec<Case> {
    let mut random = Sequence(seed ^ 0xd1b54a32d192ed03);
    if random.0 == 0 {
        random.0 = 1;
    }
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let (family, body, answer, mut choices, oracle) = match index % 7 {
            0 => {
                let a = 10 + random.pick(90);
                let b = 10 + random.pick(90);
                let sum = a + b;
                (
                    "arithmetic", format!("Compute {a} + {b}."), sum.to_string(),
                    vec![sum.to_string(), (sum + 1).to_string(), (sum - 1).to_string(), a.to_string()],
                    json!({"operands": [a, b], "sum": sum}),
                )
            }
            1 => {
                let names = ["Ada", "Bora", "Cleo", "Dara"];
                let p = random.pick(4);
                let (a, b, c) = (names[p], names[(p + 1) % 4], names[(p + 2) % 4]);
                (
                    "transitive_order",
                    format!("{a} is taller than {b}, and {b} is taller than {c}. Who is tallest?"),
                    a.into(), vec![a.into(), b.into(), c.into(), "Cannot be determined".into()],
                    json!({"order": [a, b, c]}),
                )
            }
            2 => (
                "insufficient_information",
                "Ada is taller than Bora. Ada is taller than Cleo. Who is taller, Bora or Cleo, using only these facts?".into(),
                "Cannot be determined".into(),
                vec!["Bora".into(), "Cleo".into(), "Cannot be determined".into()],
                json!({"edges": [["Ada", "Bora"], ["Ada", "Cleo"]], "query": ["Bora", "Cleo"]}),
            ),
            3 => {
                let n = 1000 + random.pick(8000);
                let reference = format!("REF-{n}");
                (
                    "exact_copy", format!("Copy the reference identifier {reference} exactly."),
                    reference.clone(), vec![reference.clone(), format!("REF-{}", n + 1), format!("ID-{n}")],
                    json!({"source": reference}),
                )
            }
            4 => {
                let n = 10 + random.pick(90);
                let answer = if n.is_multiple_of(2) {"silver"} else {"violet"};
                (
                    "conditional_rule",
                    format!("Answer silver for an even number and violet for an odd number. The number is {n}."),
                    answer.into(), vec!["silver".into(), "violet".into(), "amber".into()],
                    json!({"number": n, "even": "silver", "odd": "violet"}),
                )
            }
            5 => {
                let mut colors = ["red", "blue", "green"];
                let first = random.pick(3);
                let second = (first + 1) % 3;
                let third = (first + 2) % 3;
                colors.swap(first, second);
                colors.swap(second, third);
                let query = random.pick(3);
                (
                    "state_tracking",
                    format!("Boxes 1, 2, 3 hold red, blue, green. Swap boxes {} and {}, then boxes {} and {}. What color is in box {}?", first + 1, second + 1, second + 1, third + 1, query + 1),
                    colors[query].into(), vec!["red".into(), "blue".into(), "green".into()],
                    json!({"initial": ["red", "blue", "green"], "swaps": [[first, second], [second, third]], "query": query, "final": colors}),
                )
            }
            _ => {
                let n = 10 + random.pick(90);
                let answer = format!("{{\"count\":{n}}}");
                (
                    "format_recognition",
                    format!("Return a JSON object with exactly one key count and integer value {n}."),
                    answer.clone(), vec![answer, format!("{{\"count\":\"{n}\"}}"), format!("{{\"total\":{n}}}")],
                    json!({"key": "count", "integer": n}),
                )
            }
        };
        for end in (1..choices.len()).rev() {
            let j = random.pick(end + 1);
            choices.swap(end, j);
        }
        let expected = choices.iter().position(|s| s == &answer).unwrap();
        result.push(Case {
            id: format!("auto-{index:05}"),
            family: family.into(),
            prompt: format!("{body}\nReply with the exact answer only."),
            choices,
            expected,
            oracle,
        });
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_oracles_and_cases() {
        let a = generate(35, 19);
        let b = generate(35, 19);
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
        for c in a {
            c.validate().unwrap();
            if c.family == "arithmetic" {
                assert_eq!(c.answer(), c.oracle["sum"].to_string());
            }
        }
    }
}
