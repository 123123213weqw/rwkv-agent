use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::ToolCall;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonKind {
    String,
    Number,
    Boolean,
    Object,
    Array,
}

impl JsonKind {
    fn accepts(self, value: &Value) -> bool {
        match self {
            Self::String => value.is_string(),
            Self::Number => value.is_number(),
            Self::Boolean => value.is_boolean(),
            Self::Object => value.is_object(),
            Self::Array => value.is_array(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArgumentSpec {
    pub name: String,
    pub kind: JsonKind,
    pub required: bool,
}

impl ArgumentSpec {
    pub fn required_string(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: JsonKind::String,
            required: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub arguments: Vec<ArgumentSpec>,
    pub allow_extra_arguments: bool,
}

impl ToolDefinition {
    pub fn one_string(
        name: impl Into<String>,
        description: impl Into<String>,
        argument: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            arguments: vec![ArgumentSpec::required_string(argument)],
            allow_extra_arguments: false,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RegistryError {
    #[error("invalid tool name: {0}")]
    InvalidToolName(String),
    #[error("duplicate tool: {0}")]
    DuplicateTool(String),
    #[error("invalid argument name for {tool}: {argument}")]
    InvalidArgumentName { tool: String, argument: String },
    #[error("duplicate argument for {tool}: {argument}")]
    DuplicateArgument { tool: String, argument: String },
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("missing argument for {tool}: {argument}")]
    MissingArgument { tool: String, argument: String },
    #[error("unexpected argument for {tool}: {argument}")]
    UnexpectedArgument { tool: String, argument: String },
    #[error("invalid argument type for {tool}.{argument}: expected {expected:?}")]
    InvalidArgumentType {
        tool: String,
        argument: String,
        expected: JsonKind,
    },
}

#[derive(Clone, Debug, Default)]
pub struct ToolRegistry {
    definitions: BTreeMap<String, ToolDefinition>,
}

impl ToolRegistry {
    pub fn register(&mut self, definition: ToolDefinition) -> Result<(), RegistryError> {
        if !valid_identifier(&definition.name) {
            return Err(RegistryError::InvalidToolName(definition.name));
        }
        if self.definitions.contains_key(&definition.name) {
            return Err(RegistryError::DuplicateTool(definition.name));
        }
        let mut seen = BTreeSet::new();
        for argument in &definition.arguments {
            if !valid_identifier(&argument.name) {
                return Err(RegistryError::InvalidArgumentName {
                    tool: definition.name.clone(),
                    argument: argument.name.clone(),
                });
            }
            if !seen.insert(argument.name.clone()) {
                return Err(RegistryError::DuplicateArgument {
                    tool: definition.name.clone(),
                    argument: argument.name.clone(),
                });
            }
        }
        self.definitions.insert(definition.name.clone(), definition);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.definitions.get(name)
    }

    pub fn definitions(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.definitions.values()
    }

    pub fn validate(&self, call: &ToolCall) -> Result<(), RegistryError> {
        let definition = self
            .definitions
            .get(&call.name)
            .ok_or_else(|| RegistryError::UnknownTool(call.name.clone()))?;
        let specs = definition
            .arguments
            .iter()
            .map(|argument| (argument.name.as_str(), argument))
            .collect::<BTreeMap<_, _>>();
        for argument in &definition.arguments {
            match call.arguments.get(&argument.name) {
                None if argument.required => {
                    return Err(RegistryError::MissingArgument {
                        tool: call.name.clone(),
                        argument: argument.name.clone(),
                    });
                }
                Some(value) if !argument.kind.accepts(value) => {
                    return Err(RegistryError::InvalidArgumentType {
                        tool: call.name.clone(),
                        argument: argument.name.clone(),
                        expected: argument.kind,
                    });
                }
                _ => {}
            }
        }
        if !definition.allow_extra_arguments {
            for name in call.arguments.keys() {
                if !specs.contains_key(name.as_str()) {
                    return Err(RegistryError::UnexpectedArgument {
                        tool: call.name.clone(),
                        argument: name.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

fn valid_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};

    use super::*;

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments: arguments.as_object().cloned().unwrap_or_else(Map::new),
        }
    }

    #[test]
    fn registry_accepts_exact_arguments() {
        let mut registry = ToolRegistry::default();
        registry
            .register(ToolDefinition::one_string(
                "run_command",
                "Run a command",
                "command",
            ))
            .unwrap();
        assert!(
            registry
                .validate(&call("run_command", json!({"command": "pwd"})))
                .is_ok()
        );
    }

    #[test]
    fn registry_rejects_unknown_missing_extra_and_wrong_type() {
        let mut registry = ToolRegistry::default();
        registry
            .register(ToolDefinition::one_string("web_search", "Search", "query"))
            .unwrap();
        assert!(matches!(
            registry.validate(&call("missing", json!({}))),
            Err(RegistryError::UnknownTool(_))
        ));
        assert!(matches!(
            registry.validate(&call("web_search", json!({}))),
            Err(RegistryError::MissingArgument { .. })
        ));
        assert!(matches!(
            registry.validate(&call("web_search", json!({"query": "x", "n": 1}))),
            Err(RegistryError::UnexpectedArgument { .. })
        ));
        assert!(matches!(
            registry.validate(&call("web_search", json!({"query": 1}))),
            Err(RegistryError::InvalidArgumentType { .. })
        ));
    }
}
