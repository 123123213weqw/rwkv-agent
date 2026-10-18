use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const TASK_SPEC_SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    TASK_SPEC_SCHEMA_VERSION
}

/// Stable, model-independent description of one agent task.
///
/// Execution policy remains owned by the Controller. TaskSpec describes the
/// requested outcome and checks; it never grants runtime capabilities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSpec {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub objective: String,
    #[serde(default)]
    pub acceptance_criteria: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub verification_commands: Vec<String>,
    #[serde(default)]
    pub requires_mutation: Option<bool>,
    #[serde(default)]
    pub working_directory: Option<String>,
    #[serde(default)]
    pub stages: Vec<TaskStageSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStageSpec {
    pub id: String,
    pub objective: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub acceptance_criteria: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub verification_commands: Vec<String>,
    #[serde(default)]
    pub requires_mutation: Option<bool>,
}

impl TaskSpec {
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            schema_version: TASK_SPEC_SCHEMA_VERSION,
            objective: objective.into(),
            acceptance_criteria: Vec::new(),
            constraints: Vec::new(),
            verification_commands: Vec::new(),
            requires_mutation: None,
            working_directory: None,
            stages: Vec::new(),
        }
    }

    pub fn legacy(objective: impl Into<String>, working_directory: Option<String>) -> Self {
        Self {
            working_directory,
            ..Self::new(objective)
        }
    }

    pub fn normalize(mut self) -> Result<Self, TaskSpecError> {
        self.objective = self.objective.trim().to_string();
        self.acceptance_criteria = trim_all(self.acceptance_criteria);
        self.constraints = trim_all(self.constraints);
        self.verification_commands = trim_all(self.verification_commands);
        self.working_directory = self.working_directory.map(|value| value.trim().to_string());
        for stage in &mut self.stages {
            stage.id = stage.id.trim().to_string();
            stage.objective = stage.objective.trim().to_string();
            stage.depends_on = trim_all(std::mem::take(&mut stage.depends_on));
            stage.acceptance_criteria = trim_all(std::mem::take(&mut stage.acceptance_criteria));
            stage.constraints = trim_all(std::mem::take(&mut stage.constraints));
            stage.verification_commands =
                trim_all(std::mem::take(&mut stage.verification_commands));
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), TaskSpecError> {
        if self.schema_version != TASK_SPEC_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unsupported schema_version {}; expected {}",
                self.schema_version, TASK_SPEC_SCHEMA_VERSION
            )));
        }
        validate_text("objective", &self.objective, 16_384)?;
        validate_list("acceptance_criteria", &self.acceptance_criteria, 32, 2_000)?;
        validate_list("constraints", &self.constraints, 32, 2_000)?;
        validate_list(
            "verification_commands",
            &self.verification_commands,
            16,
            4_000,
        )?;
        if let Some(directory) = &self.working_directory {
            validate_text("working_directory", directory, 4_096)?;
        }
        let stage_checks = self.stages.iter().any(|stage| {
            !stage.verification_commands.is_empty() || stage.requires_mutation.is_some()
        });
        if self.working_directory.is_none()
            && (!self.verification_commands.is_empty()
                || self.requires_mutation.is_some()
                || stage_checks)
        {
            return Err(invalid(
                "verification_commands and requires_mutation require working_directory",
            ));
        }
        validate_stages(&self.stages)?;
        Ok(())
    }

    /// Deterministic topological order. Legacy tasks become one `main` stage.
    pub fn ordered_stages(&self) -> Result<Vec<TaskStageSpec>, TaskSpecError> {
        self.validate()?;
        if self.stages.is_empty() {
            return Ok(vec![TaskStageSpec {
                id: "main".into(),
                objective: self.objective.clone(),
                depends_on: Vec::new(),
                acceptance_criteria: self.acceptance_criteria.clone(),
                constraints: self.constraints.clone(),
                verification_commands: self.verification_commands.clone(),
                requires_mutation: self.requires_mutation,
            }]);
        }
        topological_order(&self.stages)
    }

    /// Materialize one stage as an ordinary single-stage TaskSpec so the
    /// existing AgentLoop stays the only execution engine.
    pub fn task_for_stage(
        &self,
        stage: &TaskStageSpec,
        final_stage: bool,
    ) -> Result<Self, TaskSpecError> {
        if self.stages.is_empty() {
            return Self {
                schema_version: self.schema_version,
                objective: stage.objective.clone(),
                acceptance_criteria: stage.acceptance_criteria.clone(),
                constraints: stage.constraints.clone(),
                verification_commands: stage.verification_commands.clone(),
                requires_mutation: stage.requires_mutation,
                working_directory: self.working_directory.clone(),
                stages: Vec::new(),
            }
            .normalize();
        }
        let mut acceptance_criteria = stage.acceptance_criteria.clone();
        let mut constraints = self.constraints.clone();
        constraints.extend(stage.constraints.clone());
        let mut verification_commands = stage.verification_commands.clone();
        let mut requires_mutation = stage.requires_mutation;
        if final_stage {
            acceptance_criteria.extend(self.acceptance_criteria.clone());
            if verification_commands.is_empty() {
                verification_commands = self.verification_commands.clone();
            }
            if requires_mutation.is_none() {
                requires_mutation = self.requires_mutation;
            }
        }
        Self {
            schema_version: self.schema_version,
            objective: stage.objective.clone(),
            acceptance_criteria,
            constraints,
            verification_commands,
            requires_mutation,
            working_directory: self.working_directory.clone(),
            stages: Vec::new(),
        }
        .normalize()
    }
}

fn validate_stages(stages: &[TaskStageSpec]) -> Result<(), TaskSpecError> {
    if stages.len() > 32 {
        return Err(invalid("stages must contain at most 32 items"));
    }
    let mut ids = HashMap::new();
    for (index, stage) in stages.iter().enumerate() {
        validate_stage_id(&stage.id, index)?;
        if ids.insert(stage.id.as_str(), index).is_some() {
            return Err(invalid(format!("duplicate stage id `{}`", stage.id)));
        }
        validate_text(
            &format!("stages[{index}].objective"),
            &stage.objective,
            16_384,
        )?;
        validate_list(
            &format!("stages[{index}].depends_on"),
            &stage.depends_on,
            32,
            64,
        )?;
        validate_list(
            &format!("stages[{index}].acceptance_criteria"),
            &stage.acceptance_criteria,
            32,
            2_000,
        )?;
        validate_list(
            &format!("stages[{index}].constraints"),
            &stage.constraints,
            32,
            2_000,
        )?;
        validate_list(
            &format!("stages[{index}].verification_commands"),
            &stage.verification_commands,
            16,
            4_000,
        )?;
    }
    for stage in stages {
        for dependency in &stage.depends_on {
            if dependency == &stage.id {
                return Err(invalid(format!(
                    "stage `{}` cannot depend on itself",
                    stage.id
                )));
            }
            if !ids.contains_key(dependency.as_str()) {
                return Err(invalid(format!(
                    "stage `{}` depends on unknown stage `{dependency}`",
                    stage.id
                )));
            }
        }
    }
    if !stages.is_empty() {
        topological_order(stages)?;
    }
    Ok(())
}

fn topological_order(stages: &[TaskStageSpec]) -> Result<Vec<TaskStageSpec>, TaskSpecError> {
    let mut remaining = stages.iter().collect::<Vec<_>>();
    let mut completed = HashSet::new();
    let mut ordered = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let Some(index) = remaining
            .iter()
            .position(|stage| stage.depends_on.iter().all(|id| completed.contains(id)))
        else {
            return Err(invalid("stages contain a dependency cycle"));
        };
        let stage = remaining.remove(index);
        completed.insert(stage.id.clone());
        ordered.push(stage.clone());
    }
    Ok(ordered)
}

fn validate_stage_id(value: &str, index: usize) -> Result<(), TaskSpecError> {
    if value.is_empty()
        || value.chars().count() > 64
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._-".contains(character))
    {
        return Err(invalid(format!(
            "stages[{index}].id must be 1-64 ASCII letters, digits, `.`, `_`, or `-`"
        )));
    }
    Ok(())
}

fn trim_all(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| value.trim().to_string())
        .collect()
}

fn validate_list(
    field: &str,
    values: &[String],
    max_items: usize,
    max_chars: usize,
) -> Result<(), TaskSpecError> {
    if values.len() > max_items {
        return Err(invalid(format!(
            "{field} must contain at most {max_items} items"
        )));
    }
    for (index, value) in values.iter().enumerate() {
        validate_text(&format!("{field}[{index}]"), value, max_chars)?;
    }
    Ok(())
}

fn validate_text(field: &str, value: &str, max_chars: usize) -> Result<(), TaskSpecError> {
    if value.trim().is_empty() {
        return Err(invalid(format!("{field} must not be empty")));
    }
    if value.chars().count() > max_chars {
        return Err(invalid(format!(
            "{field} must contain at most {max_chars} characters"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> TaskSpecError {
    TaskSpecError::Invalid(message.into())
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TaskSpecError {
    #[error("invalid TaskSpec: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(id: &str, depends_on: &[&str]) -> TaskStageSpec {
        TaskStageSpec {
            id: id.into(),
            objective: format!("execute {id}"),
            depends_on: depends_on.iter().map(|value| (*value).into()).collect(),
            acceptance_criteria: Vec::new(),
            constraints: Vec::new(),
            verification_commands: Vec::new(),
            requires_mutation: None,
        }
    }

    #[test]
    fn legacy_task_normalizes_to_v1() {
        let task = TaskSpec::legacy("  fix calc.py  ", Some("  /repo  ".into()))
            .normalize()
            .unwrap();
        assert_eq!(task.schema_version, 1);
        assert_eq!(task.objective, "fix calc.py");
        assert_eq!(task.working_directory.as_deref(), Some("/repo"));
        assert_eq!(task.ordered_stages().unwrap()[0].id, "main");
    }

    #[test]
    fn structured_workspace_task_round_trips() {
        let raw = serde_json::json!({
            "objective": "Fix calc.py",
            "acceptance_criteria": ["2 + 2 returns 4"],
            "constraints": ["Do not change the public API"],
            "verification_commands": ["python3 test_calc.py"],
            "requires_mutation": true,
            "working_directory": "/repo"
        });
        let task: TaskSpec = serde_json::from_value(raw).unwrap();
        task.validate().unwrap();
        assert_eq!(serde_json::to_value(&task).unwrap()["schema_version"], 1);
    }

    #[test]
    fn stage_dag_is_sorted_and_materialized() {
        let mut task = TaskSpec::legacy("Finish feature", Some("/repo".into()));
        task.constraints = vec!["Keep API".into()];
        task.stages = vec![stage("verify", &["fix"]), stage("fix", &[])];
        task.stages[0].verification_commands = vec!["python3 test.py".into()];
        task.stages[0].requires_mutation = Some(false);
        let task = task.normalize().unwrap();
        let stages = task.ordered_stages().unwrap();
        assert_eq!(
            stages.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            ["fix", "verify"]
        );
        let final_task = task.task_for_stage(&stages[1], true).unwrap();
        assert_eq!(final_task.constraints, ["Keep API"]);
        assert_eq!(final_task.verification_commands, ["python3 test.py"]);
    }

    #[test]
    fn stage_cycles_and_unknown_dependencies_are_rejected() {
        let mut task = TaskSpec::new("cycle");
        task.stages = vec![stage("a", &["b"]), stage("b", &["a"])];
        assert!(task.validate().unwrap_err().to_string().contains("cycle"));
        task.stages = vec![stage("a", &["missing"])];
        assert!(task.validate().unwrap_err().to_string().contains("unknown"));
    }

    #[test]
    fn execution_checks_require_a_workspace() {
        let mut task = TaskSpec::new("Run a check");
        task.verification_commands.push("python3 test.py".into());
        assert!(
            task.validate()
                .unwrap_err()
                .to_string()
                .contains("require working_directory")
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let error = serde_json::from_value::<TaskSpec>(serde_json::json!({
            "objective": "hello",
            "unexpected": true
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }
}
