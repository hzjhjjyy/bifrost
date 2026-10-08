package bifrost

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// Provider names and wire compatibility do not determine which HTTP client is
// used: custom providers select their implementation through BaseProviderType.
func TestProviderTransportRetryIsolation(t *testing.T) {
	for _, tc := range []struct {
		name     schemas.ModelProvider
		base     schemas.ModelProvider
		attempts int32
	}{
		{schemas.OpenAI, "", 1},
		{"durian-deepseek", schemas.OpenAI, 1},
		{schemas.DeepSeek, "", 2},
		{schemas.Groq, "", 2},
	} {
		t.Run(string(tc.name), func(t *testing.T) {
			var received atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil || len(body) == 0 || r.Method != http.MethodPost {
					t.Errorf("expected complete chat POST: method=%s body=%q err=%v", r.Method, body, err)
				}
				if strings.Contains(string(body), `"warm"`) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"warm","object":"chat.completion","model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
					return
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
			config := &schemas.ProviderConfig{
				NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, MaxRetries: 0, DefaultRequestTimeoutInSeconds: 2},
			}
			if tc.base != "" {
				config.CustomProviderConfig = &schemas.CustomProviderConfig{BaseProviderType: tc.base}
			}
			client := &Bifrost{logger: NewDefaultLogger(schemas.LogLevelError)}
			provider, err := client.createBaseProvider(tc.name, config)
			if err != nil {
				t.Fatal(err)
			}
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			request := &schemas.BifrostChatRequest{
				Provider: tc.name,
				Model:    "test-model",
				Input: []schemas.ChatMessage{{
					Role:    schemas.ChatMessageRoleUser,
					Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("warm")},
				}},
			}
			// Upstream only retries reused connections. Warm each provider's client
			// so this still distinguishes the custom zero-retry limit from its peers.
			if response, err := provider.ChatCompletion(ctx, schemas.Key{}, request); response == nil || err != nil {
				t.Fatalf("warm-up failed: response=%v error=%v", response, err)
			}
			request.Input[0].Content.ContentStr = schemas.Ptr("disconnect")
			response, bifrostErr := provider.ChatCompletion(ctx, schemas.Key{}, request)
			if response != nil || bifrostErr == nil {
				t.Fatalf("expected connection failure: response=%v error=%v", response, bifrostErr)
			}
			if got := received.Load(); got != tc.attempts {
				t.Fatalf("upstream received %d complete requests, want %d", got, tc.attempts)
			}
		})
	}
}
