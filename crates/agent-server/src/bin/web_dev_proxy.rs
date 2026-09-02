use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use clap::Parser;
use reqwest::Url;

const MAX_REQUEST_BODY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Parser, Debug)]
#[command(name = "rwkv-agent-web-dev-proxy-rs")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 5173)]
    port: u16,
    #[arg(long, default_value = "http://127.0.0.1:8122")]
    target: Url,
    #[arg(long, default_value = "web")]
    web_root: PathBuf,
}

#[derive(Clone)]
struct ProxyState {
    client: reqwest::Client,
    target: Url,
    web_root: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();
    let address: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|error| format!("invalid listen address: {error}"))?;
    if !address.ip().is_loopback() {
        return Err("the frontend development proxy must listen on loopback".into());
    }
    if !target_is_loopback(&args.target) {
        return Err("the frontend development proxy target must be loopback".into());
    }
    validate_web_root(&args.web_root).await?;
    let state = ProxyState {
        client: reqwest::Client::new(),
        target: args.target,
        web_root: args.web_root,
    };
    let app = Router::new().fallback(any(dispatch)).with_state(state);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| format!("bind {address}: {error}"))?;
    println!("RWKV frontend development proxy listening on http://{address}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| format!("serve frontend development proxy: {error}"))
}

async fn dispatch(State(state): State<ProxyState>, request: Request<Body>) -> Response {
    let path = request.uri().path();
    if let Some(asset) = static_asset(path) {
        return serve_asset(&state.web_root, asset).await;
    }
    if canonical_route(request.method(), path) {
        return proxy_request(state, request).await;
    }
    (StatusCode::NOT_FOUND, "not found").into_response()
}

fn static_asset(path: &str) -> Option<(&'static str, &'static str)> {
    match path {
        "/" | "/tasks" | "/status" => Some(("index.html", "text/html; charset=utf-8")),
        "/assets/app.css" => Some(("app.css", "text/css; charset=utf-8")),
        "/assets/app.js" => Some(("dist/app.js", "text/javascript; charset=utf-8")),
        "/assets/api-client.js" => Some(("dist/api-client.js", "text/javascript; charset=utf-8")),
        _ => None,
    }
}

fn canonical_route(method: &Method, path: &str) -> bool {
    if method == Method::GET {
        return matches!(
            path,
            "/live" | "/ready" | "/v1/openapi.json" | "/v1/schema.json" | "/v1/tasks"
        ) || task_detail_path(path);
    }
    if method == Method::POST {
        return matches!(path, "/v1/tasks" | "/v1/tasks/stream" | "/v1/research")
            || task_control_path(path);
    }
    false
}

fn task_detail_path(path: &str) -> bool {
    let Some(task_id) = path.strip_prefix("/v1/tasks/") else {
        return false;
    };
    !task_id.is_empty() && !task_id.contains('/')
}

fn task_control_path(path: &str) -> bool {
    let Some(rest) = path.strip_prefix("/v1/tasks/") else {
        return false;
    };
    let Some((task_id, action)) = rest.split_once('/') else {
        return false;
    };
    !task_id.is_empty() && matches!(action, "cancel" | "resume")
}

async fn serve_asset(root: &Path, asset: (&'static str, &'static str)) -> Response {
    match tokio::fs::read(root.join(asset.0)).await {
        Ok(bytes) => secured_asset(bytes, asset.1),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("frontend asset unavailable: {error}"),
        )
            .into_response(),
    }
}

fn secured_asset(bytes: Vec<u8>, content_type: &'static str) -> Response {
    let mut response = Body::from(bytes).into_response();
    let headers = response.headers_mut();
    headers.insert("content-type", HeaderValue::from_static(content_type));
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    headers.insert(
        "content-security-policy",
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        ),
    );
    response
}

async fn proxy_request(state: ProxyState, request: Request<Body>) -> Response {
    let (parts, body) = request.into_parts();
    let target = match target_url(&state.target, &parts.uri) {
        Ok(target) => target,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };
    let body = match to_bytes(body, MAX_REQUEST_BODY_BYTES).await {
        Ok(body) => body,
        Err(error) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("request body rejected: {error}"),
            )
                .into_response();
        }
    };
    let mut outgoing = state.client.request(parts.method, target).body(body);
    outgoing = forward_request_headers(outgoing, &parts.headers);
    let upstream = match outgoing.send().await {
        Ok(response) => response,
        Err(error) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("canonical controller unavailable: {error}"),
            )
                .into_response();
        }
    };
    let status = upstream.status();
    let headers = upstream.headers().clone();
    let mut response = Response::builder().status(status);
    for name in [
        "content-type",
        "cache-control",
        "x-accel-buffering",
        "x-content-type-options",
    ] {
        if let Some(value) = headers.get(name) {
            response = response.header(name, value);
        }
    }
    response
        .body(Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|error| {
            (
                StatusCode::BAD_GATEWAY,
                format!("build controller response: {error}"),
            )
                .into_response()
        })
}

fn forward_request_headers(
    mut request: reqwest::RequestBuilder,
    headers: &HeaderMap,
) -> reqwest::RequestBuilder {
    for name in ["accept", "content-type"] {
        if let (Ok(name), Some(value)) =
            (HeaderName::from_bytes(name.as_bytes()), headers.get(name))
        {
            request = request.header(name, value);
        }
    }
    request
}

fn target_url(base: &Url, uri: &Uri) -> Result<Url, String> {
    let mut target = base.clone();
    target.set_path(uri.path());
    target.set_query(uri.query());
    Ok(target)
}

fn target_is_loopback(target: &Url) -> bool {
    match target.host_str() {
        Some("localhost") => true,
        Some(host) => host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback()),
        None => false,
    }
}

async fn validate_web_root(root: &Path) -> Result<(), String> {
    for path in ["index.html", "app.css", "dist/app.js", "dist/api-client.js"] {
        if !root.join(path).is_file() {
            return Err(format!("frontend development asset is missing: {path}"));
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn development_proxy_allows_only_the_frontend_contract() {
        for (method, path) in [
            (Method::GET, "/live"),
            (Method::GET, "/ready"),
            (Method::GET, "/v1/tasks"),
            (Method::GET, "/v1/tasks/task-1"),
            (Method::POST, "/v1/tasks"),
            (Method::POST, "/v1/tasks/stream"),
            (Method::POST, "/v1/tasks/task-1/cancel"),
            (Method::POST, "/v1/tasks/task-1/resume"),
            (Method::POST, "/v1/research"),
        ] {
            assert!(canonical_route(&method, path), "{method} {path}");
        }
        for (method, path) in [
            (Method::GET, "/health"),
            (Method::GET, "/v1/debug/traces"),
            (Method::GET, "/v1/task-ledger"),
            (Method::POST, "/v1/agent/run"),
            (Method::POST, "/v1/tools/call"),
            (Method::POST, "/v1/tasks/task-1/unknown"),
        ] {
            assert!(!canonical_route(&method, path), "{method} {path}");
        }
    }

    #[test]
    fn development_proxy_is_loopback_only() {
        assert!(target_is_loopback(
            &Url::parse("http://127.0.0.1:8122").unwrap()
        ));
        assert!(target_is_loopback(
            &Url::parse("http://[::1]:8122").unwrap()
        ));
        assert!(!target_is_loopback(
            &Url::parse("https://example.test").unwrap()
        ));
    }
}
