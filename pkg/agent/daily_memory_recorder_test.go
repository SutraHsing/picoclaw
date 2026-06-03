package agent

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

type dailyMemoryTestProvider struct {
	mu       sync.Mutex
	response string
	err      error
	calls    int
}

func (p *dailyMemoryTestProvider) Chat(
	ctx context.Context,
	messages []providers.Message,
	tools []providers.ToolDefinition,
	model string,
	options map[string]any,
) (*providers.LLMResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	if p.err != nil {
		return nil, p.err
	}
	return &providers.LLMResponse{Content: p.response}, nil
}

func (p *dailyMemoryTestProvider) GetDefaultModel() string {
	return "test-model"
}

func (p *dailyMemoryTestProvider) Calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

func newDailyMemoryTestLoop(
	t *testing.T,
	provider *dailyMemoryTestProvider,
) (*AgentLoop, *AgentInstance, string) {
	t.Helper()
	workspace := t.TempDir()
	cfg := &config.Config{
		Agents: config.AgentsConfig{
			Defaults: config.AgentDefaults{
				Workspace:         workspace,
				ModelName:         "test-model",
				MaxTokens:         4096,
				MaxToolIterations: 10,
			},
		},
	}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	agent := al.GetRegistry().GetDefaultAgent()
	if agent == nil {
		t.Fatal("expected default agent")
	}
	return al, agent, workspace
}

func setDailyMemoryTestHistory(agent *AgentInstance, sessionKey string) {
	agent.Sessions.SetHistory(sessionKey, []providers.Message{
		{Role: "user", Content: "我决定先实现 /clear 前的 daily notes flush。"},
		{Role: "assistant", Content: "好的，先做最小可行版本。"},
	})
}

func runDailyMemoryCommand(
	t *testing.T,
	al *AgentLoop,
	agent *AgentInstance,
	sessionKey string,
	content string,
) (string, bool) {
	t.Helper()
	opts := processOptions{
		Dispatch:   DispatchRequest{SessionKey: sessionKey},
		SessionKey: sessionKey,
	}
	return al.handleCommand(
		context.Background(),
		bus.InboundMessage{
			Context: bus.InboundContext{
				Channel:  "pico",
				ChatID:   "daily-memory-test",
				SenderID: "user",
			},
			Content: content,
		},
		agent,
		&opts,
	)
}

func readDailyMemoryFiles(t *testing.T, workspace string) string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(workspace, "memory", "*", "*.md"))
	if err != nil {
		t.Fatalf("glob daily memory files: %v", err)
	}
	var sb strings.Builder
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read daily memory file %s: %v", path, err)
		}
		sb.Write(data)
	}
	return sb.String()
}

func TestClearCommandFlushesDailyMemoryBeforeClear(t *testing.T) {
	provider := &dailyMemoryTestProvider{
		response: `{"should_write":true,"entries":["用户决定先实现 /clear 前的 daily notes flush。"]}`,
	}
	al, agent, workspace := newDailyMemoryTestLoop(t, provider)
	sessionKey := "daily-memory-clear"
	setDailyMemoryTestHistory(agent, sessionKey)

	reply, handled := runDailyMemoryCommand(t, al, agent, sessionKey, "/clear")
	if !handled {
		t.Fatal("/clear was not handled")
	}
	if reply != "Chat history cleared!" {
		t.Fatalf("/clear reply = %q, want Chat history cleared!", reply)
	}
	if history := agent.Sessions.GetHistory(sessionKey); len(history) != 0 {
		t.Fatalf("history len after /clear = %d, want 0", len(history))
	}
	content := readDailyMemoryFiles(t, workspace)
	if !strings.Contains(content, "## ") || !strings.Contains(content, "· clear") {
		t.Fatalf("daily note missing clear heading:\n%s", content)
	}
	if !strings.Contains(content, "用户决定先实现 /clear 前的 daily notes flush。") {
		t.Fatalf("daily note missing extracted entry:\n%s", content)
	}
	if provider.Calls() != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.Calls())
	}
}

func TestClearCommandDailyMemorySkipOrFailureStillClearsHistory(t *testing.T) {
	tests := []struct {
		name     string
		response string
		err      error
	}{
		{
			name:     "should_write_false",
			response: `{"should_write":false,"entries":[]}`,
		},
		{
			name:     "malformed_json",
			response: `not json`,
		},
		{
			name: "provider_error",
			err:  errors.New("provider unavailable"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &dailyMemoryTestProvider{
				response: tt.response,
				err:      tt.err,
			}
			al, agent, workspace := newDailyMemoryTestLoop(t, provider)
			sessionKey := "daily-memory-clear-" + tt.name
			setDailyMemoryTestHistory(agent, sessionKey)

			reply, handled := runDailyMemoryCommand(t, al, agent, sessionKey, "/clear")
			if !handled {
				t.Fatal("/clear was not handled")
			}
			if reply != "Chat history cleared!" {
				t.Fatalf("/clear reply = %q, want Chat history cleared!", reply)
			}
			if history := agent.Sessions.GetHistory(sessionKey); len(history) != 0 {
				t.Fatalf("history len after /clear = %d, want 0", len(history))
			}
			if content := readDailyMemoryFiles(t, workspace); content != "" {
				t.Fatalf("daily note content = %q, want empty", content)
			}
			if provider.Calls() != 1 {
				t.Fatalf("provider calls = %d, want 1", provider.Calls())
			}
		})
	}
}

func TestStartCommandDoesNotFlushDailyMemory(t *testing.T) {
	provider := &dailyMemoryTestProvider{
		response: `{"should_write":true,"entries":["should not be written"]}`,
	}
	al, agent, workspace := newDailyMemoryTestLoop(t, provider)
	sessionKey := "daily-memory-start"
	setDailyMemoryTestHistory(agent, sessionKey)

	reply, handled := runDailyMemoryCommand(t, al, agent, sessionKey, "/start")
	if !handled {
		t.Fatal("/start was not handled")
	}
	if reply != "Hello! I am PicoClaw 🦞" {
		t.Fatalf("/start reply = %q, want hello", reply)
	}
	if provider.Calls() != 0 {
		t.Fatalf("provider calls = %d, want 0", provider.Calls())
	}
	if content := readDailyMemoryFiles(t, workspace); content != "" {
		t.Fatalf("daily note content = %q, want empty", content)
	}
	if history := agent.Sessions.GetHistory(sessionKey); len(history) == 0 {
		t.Fatal("/start should not clear history")
	}
}

func TestDailyMemoryRecorderConcurrentAppendDoesNotDropEntries(t *testing.T) {
	provider := &dailyMemoryTestProvider{
		response: `{"should_write":true,"entries":["并发记录"]}`,
	}
	_, agent, workspace := newDailyMemoryTestLoop(t, provider)
	sessionKey := "daily-memory-concurrent"
	setDailyMemoryTestHistory(agent, sessionKey)
	recorder := NewDailyMemoryRecorder()

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- recorder.FlushSession(context.Background(), agent, sessionKey, "clear")
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("FlushSession error: %v", err)
		}
	}

	content := readDailyMemoryFiles(t, workspace)
	if got := strings.Count(content, "并发记录"); got != 2 {
		t.Fatalf("entry count = %d, want 2; content:\n%s", got, content)
	}
}
