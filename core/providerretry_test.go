package bifrost

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
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
		{schemas.DeepSeek, "", 4},
		{schemas.Groq, "", 4},
	} {
		t.Run(string(tc.name), func(t *testing.T) {
			var received atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil || len(body) == 0 || r.Method != http.MethodPost {
					t.Errorf("expected complete chat POST: method=%s body=%q err=%v", r.Method, body, err)
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
			response, bifrostErr := provider.ChatCompletion(ctx, schemas.Key{}, &schemas.BifrostChatRequest{
				Provider: tc.name,
				Model:    "test-model",
				Input: []schemas.ChatMessage{{
					Role:    schemas.ChatMessageRoleUser,
					Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("disconnect")},
				}},
			})
			if response != nil || bifrostErr == nil {
				t.Fatalf("expected connection failure: response=%v error=%v", response, bifrostErr)
			}
			if got := received.Load(); got != tc.attempts {
				t.Fatalf("upstream received %d complete requests, want %d", got, tc.attempts)
			}
		})
	}
}
