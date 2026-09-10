package openai

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// Count complete requests at the upstream: one provider call can contain several
// fasthttp attempts, even when core's retry budget is zero.
func TestPassthroughRetryBudget(t *testing.T) {
	for _, retries := range []int{0, 2} {
		for _, method := range []string{http.MethodPost, http.MethodGet, http.MethodPut, http.MethodDelete} {
			for _, fault := range []string{"eof", "reset", "headers", "body"} {
				for _, warm := range []bool{false, true} {
					t.Run(fmt.Sprintf("retries=%d/%s/%s/warm=%t", retries, method, fault, warm), func(t *testing.T) {
						const body = `{"model":"test-model","messages":[{"role":"user","content":"disconnect"}]}`
						var mu sync.Mutex
						var addresses []string
						var warmAddress string
						server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							if r.URL.Path == "/v1/warm" {
								mu.Lock()
								warmAddress = r.RemoteAddr
								mu.Unlock()
								w.Header().Set("Content-Length", "2")
								_, _ = io.WriteString(w, "ok")
								return
							}
							got, err := io.ReadAll(r.Body)
							if err != nil || string(got) != body || r.Method != method {
								t.Errorf("upstream request: method=%s body=%q err=%v", r.Method, got, err)
							}
							mu.Lock()
							addresses = append(addresses, r.RemoteAddr)
							mu.Unlock()
							conn, _, err := w.(http.Hijacker).Hijack()
							if err != nil {
								t.Errorf("hijack: %v", err)
								return
							}
							defer conn.Close()
							switch fault {
							case "reset":
								if err := conn.(*net.TCPConn).SetLinger(0); err != nil {
									t.Errorf("set linger: %v", err)
								}
							case "headers":
								if _, err := io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Length:"); err != nil {
									t.Errorf("write partial headers: %v", err)
								}
							case "body":
								if _, err := io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial"); err != nil {
									t.Errorf("write partial body: %v", err)
								}
							}
						}))
						defer server.Close()
						provider := NewOpenAIProvider(&schemas.ProviderConfig{
							NetworkConfig: schemas.NetworkConfig{
								BaseURL: server.URL, MaxRetries: retries, DefaultRequestTimeoutInSeconds: 2,
							},
							CustomProviderConfig: &schemas.CustomProviderConfig{
								BaseProviderType: schemas.OpenAI, CustomProviderKey: "retry-test",
							},
						}, passthroughTestLogger{})
						defer provider.client.CloseIdleConnections()
						defer provider.streamingClient.CloseIdleConnections()
						ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
						if warm {
							resp, err := provider.Passthrough(ctx, schemas.Key{}, &schemas.BifrostPassthroughRequest{
								Method: http.MethodGet, Path: "/warm",
							})
							if err != nil || resp == nil || string(resp.Body) != "ok" {
								t.Fatalf("warm-up failed: response=%v error=%v", resp, err)
							}
						}
						resp, err := provider.Passthrough(ctx, schemas.Key{}, &schemas.BifrostPassthroughRequest{
							Method: method, Path: "/chat/completions", Body: []byte(body),
						})
						if resp != nil || err == nil || err.StatusCode == nil || *err.StatusCode != http.StatusBadGateway {
							t.Fatalf("want connection failure, got response=%v error=%v", resp, err)
						}
						// Passthrough waits for client.Do to finish, including all transport
						// attempts, before returning. No retry can arrive after this snapshot.
						mu.Lock()
						defer mu.Unlock()
						want := 1
						if retries > 0 {
							// Preserve the existing three transport retries per provider call.
							want = 4
						}
						if len(addresses) != want {
							t.Errorf("upstream received %d complete requests, want %d; connections=%v", len(addresses), want, addresses)
						}
						if warm && (len(addresses) == 0 || addresses[0] != warmAddress) {
							t.Errorf("first fault request did not reuse warm connection %s: %v", warmAddress, addresses)
						}
					})
				}
			}
		}
	}
}

func TestPassthroughStreamRetryBudget(t *testing.T) {
	for _, retries := range []int{0, 2} {
		t.Run(fmt.Sprintf("retries=%d", retries), func(t *testing.T) {
			var received atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil || string(body) != `{"stream":true}` {
					t.Errorf("upstream body=%q err=%v", body, err)
				}
				received.Add(1)
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Errorf("hijack: %v", err)
					return
				}
				_ = conn.Close()
			}))
			defer server.Close()
			provider := NewOpenAIProvider(&schemas.ProviderConfig{
				NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, MaxRetries: retries},
			}, passthroughTestLogger{})
			defer provider.streamingClient.CloseIdleConnections()
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			stream, err := provider.PassthroughStream(ctx, nil, nil, schemas.Key{}, &schemas.BifrostPassthroughRequest{
				Method: http.MethodPost, Path: "/chat/completions", Body: []byte(`{"stream":true}`),
			})
			if stream != nil || err == nil {
				t.Fatalf("want stream setup failure, got stream=%v error=%v", stream, err)
			}
			want := int32(1)
			if retries > 0 {
				want = 4
			}
			if got := received.Load(); got != want {
				t.Fatalf("upstream received %d complete requests, want %d", got, want)
			}
		})
	}
}

func TestPassthroughRetryPreservesResponse(t *testing.T) {
	for _, status := range []int{200, 400, 401, 402, 422, 429, 500, 503} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			const body = `{"usage":{"prompt_cache_hit_tokens":5,"prompt_cache_miss_tokens":8,"completion_tokens":4},"system_fingerprint":"test","choices":[{"finish_reason":"stop"}]}`
			var received atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				received.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "2")
				w.WriteHeader(status)
				_, _ = io.WriteString(w, body)
			}))
			defer server.Close()
			provider := NewOpenAIProvider(&schemas.ProviderConfig{
				NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL},
			}, passthroughTestLogger{})
			defer provider.client.CloseIdleConnections()
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			resp, err := provider.Passthrough(ctx, schemas.Key{}, &schemas.BifrostPassthroughRequest{
				Method: http.MethodPost, Path: "/chat/completions", Body: []byte(`{"model":"test-model"}`),
			})
			if err != nil || resp == nil {
				t.Fatalf("passthrough failed: response=%v error=%v", resp, err)
			}
			if resp.StatusCode != status || string(resp.Body) != body || resp.Headers["Retry-After"] != "2" || received.Load() != 1 {
				t.Fatalf("response changed: status=%d body=%q headers=%v requests=%d", resp.StatusCode, resp.Body, resp.Headers, received.Load())
			}
		})
	}
}

// The streaming client must inherit the limit, while nonzero configurations
// retain both fasthttp's default attempt limit and the shared retry callback.
func TestOpenAIRetryClientConfig(t *testing.T) {
	for _, retries := range []int{0, 1, 2} {
		t.Run(fmt.Sprintf("retries=%d", retries), func(t *testing.T) {
			provider := NewOpenAIProvider(&schemas.ProviderConfig{
				NetworkConfig: schemas.NetworkConfig{MaxRetries: retries},
			}, passthroughTestLogger{})
			want := 0
			if retries == 0 {
				want = 1
			}
			if provider.client.MaxIdemponentCallAttempts != want || provider.streamingClient.MaxIdemponentCallAttempts != want {
				t.Fatalf("unary/streaming attempt limits = %d/%d, want %d", provider.client.MaxIdemponentCallAttempts, provider.streamingClient.MaxIdemponentCallAttempts, want)
			}
			if provider.client.RetryIfErr == nil || provider.streamingClient.RetryIfErr == nil {
				t.Fatal("shared retry callback was removed")
			}
			if retries > 0 {
				for _, client := range []*fasthttp.Client{provider.client, provider.streamingClient} {
					if reset, retry := client.RetryIfErr(nil, 1, io.EOF); !reset || !retry {
						t.Fatal("nonzero retry policy changed")
					}
				}
			}
		})
	}
}
