//! Helpers for tests that talk to a mock HTTP server on a local port.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;

/// Serve one canned response on a free local port.
///
/// Returns the server's base URL (for example `http://127.0.0.1:41234`) and a channel that
/// yields the raw text of the request that arrived, so a test can check which path and which
/// body the client really sent.
///
/// # Panics
///
/// Panics if no local port can be bound, which fails the test.
pub async fn capture_once(
    content_type: &'static str,
    body: &'static str,
) -> (String, oneshot::Receiver<String>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buf = vec![0u8; 16 * 1024];
        // Read until the client has been quiet for a moment: the whole request has arrived.
        while let Ok(Ok(n)) =
            tokio::time::timeout(std::time::Duration::from_millis(200), socket.read(&mut buf)).await
        {
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
        }
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = socket.write_all(head.as_bytes()).await;
        let _ = socket.write_all(body.as_bytes()).await;
        let _ = tx.send(String::from_utf8_lossy(&request).to_string());
    });
    (base, rx)
}
