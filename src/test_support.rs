//! Helpers for tests that talk to a mock HTTP server on a local port.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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
    body: impl Into<String>,
) -> (String, oneshot::Receiver<String>) {
    let body: String = body.into();
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

/// How [`serve_file`] behaves.
#[derive(Clone, Copy, Debug, Default)]
pub struct Behaviour {
    /// Ignore `Range` headers and always send the whole file, like a server without range
    /// support.
    pub ignore_range: bool,
    /// On the first request only, send this many body bytes and then drop the connection,
    /// like a download that is interrupted.
    pub cut_first_after: Option<usize>,
    /// Answer every request with this status and no body.
    pub status: Option<u16>,
}

/// A local server holding one file, which remembers the requests it received.
pub struct FileServer {
    /// The address of the file.
    pub url: String,
    heads: Arc<Mutex<Vec<String>>>,
}

impl FileServer {
    /// How many requests arrived.
    ///
    /// # Panics
    ///
    /// Panics if the record is poisoned, which fails the test.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.heads.lock().unwrap().len()
    }

    /// The first byte asked for by each request: `None` for a request without a `Range`.
    ///
    /// # Panics
    ///
    /// Panics if the record is poisoned, which fails the test.
    #[must_use]
    pub fn range_starts(&self) -> Vec<Option<u64>> {
        self.heads
            .lock()
            .unwrap()
            .iter()
            .map(|h| range_start(h))
            .collect()
    }
}

/// The start of `Range: bytes=START-` in a request head.
fn range_start(head: &str) -> Option<u64> {
    head.lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("range: bytes=")
                .map(str::to_string)
        })
        .and_then(|v| v.split('-').next()?.trim().parse().ok())
}

/// Serve `body` at a free local port, honouring `Range` requests like a real file server.
///
/// # Panics
///
/// Panics if no local port can be bound, which fails the test.
pub async fn serve_file(body: Vec<u8>, behaviour: Behaviour) -> FileServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/models/file.bin", listener.local_addr().unwrap());
    let heads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::new(AtomicUsize::new(0));
    let recorded = Arc::clone(&heads);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let (body, heads, seen) = (body.clone(), Arc::clone(&recorded), Arc::clone(&seen));
            tokio::spawn(async move {
                let mut head = Vec::new();
                let mut buf = [0u8; 4096];
                while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => head.extend_from_slice(&buf[..n]),
                    }
                }
                let head = String::from_utf8_lossy(&head).to_string();
                let first = seen.fetch_add(1, Ordering::SeqCst) == 0;
                let start = range_start(&head).filter(|_| !behaviour.ignore_range);
                heads.lock().unwrap().push(head);
                let len = body.len();
                let (status, extra, slice): (String, String, &[u8]) =
                    match (behaviour.status, start) {
                        (Some(code), _) => (format!("{code} Error"), String::new(), &[]),
                        (None, Some(s)) if usize::try_from(s).is_ok_and(|s| s >= len) => (
                            "416 Range Not Satisfiable".into(),
                            format!("Content-Range: bytes */{len}\r\n"),
                            &[],
                        ),
                        (None, Some(s)) => {
                            let s = usize::try_from(s).unwrap();
                            (
                                "206 Partial Content".into(),
                                format!("Content-Range: bytes {s}-{}/{len}\r\n", len - 1),
                                &body[s..],
                            )
                        }
                        (None, None) => ("200 OK".into(), String::new(), &body[..]),
                    };
                let reply = format!(
                    "HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n",
                    slice.len()
                );
                let _ = socket.write_all(reply.as_bytes()).await;
                let cut = behaviour
                    .cut_first_after
                    .filter(|_| first && behaviour.status.is_none());
                let send = cut.map_or(slice.len(), |c| c.min(slice.len()));
                let _ = socket.write_all(&slice[..send]).await;
                let _ = socket.flush().await;
            });
        }
    });
    FileServer { url, heads }
}
