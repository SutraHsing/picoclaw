package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/utils"
)

const (
	dailyMemoryFlushTimeout      = 45 * time.Second
	dailyMemoryFlushMaxMessages  = 20
	dailyMemoryFlushMaxChars     = 12_000
	dailyMemoryFlushMaxTokens    = 1024
	dailyMemoryFlushMaxEntries   = 5
	dailyMemoryFlushMaxEntrySize = 600
)

// DailyMemoryRecorder extracts high-signal session details and appends them
// to the workspace daily note. It is best-effort: callers should not let it
// block primary user actions such as /clear.
type DailyMemoryRecorder struct {
	mu  sync.Mutex
	now func() time.Time
}

type dailyMemoryFlushResponse struct {
	ShouldWrite bool     `json:"should_write"`
	Entries     []string `json:"entries"`
}

// NewDailyMemoryRecorder creates a recorder for daily note writes.
func NewDailyMemoryRecorder() *DailyMemoryRecorder {
	return &DailyMemoryRecorder{
		now: time.Now,
	}
}

// FlushSession asks the current model whether the session contains high-signal
// details worth writing to today's daily note. It only writes when the model
// returns explicit entries.
func (r *DailyMemoryRecorder) FlushSession(
	ctx context.Context,
	agent *AgentInstance,
	sessionKey string,
	reason string,
) error {
	if r == nil || agent == nil || agent.Provider == nil ||
		agent.Sessions == nil || agent.ContextBuilder == nil ||
		agent.ContextBuilder.memory == nil {
		return nil
	}

	summary := strings.TrimSpace(agent.Sessions.GetSummary(sessionKey))
	messages := dailyMemoryRelevantMessages(agent.Sessions.GetHistory(sessionKey))
	if summary == "" && len(messages) < 2 {
		return nil
	}

	prompt := r.buildFlushPrompt(summary, messages, reason)
	callCtx, cancel := context.WithTimeout(ctx, dailyMemoryFlushTimeout)
	defer cancel()

	resp, err := agent.Provider.Chat(
		callCtx,
		[]providers.Message{{Role: "user", Content: prompt}},
		nil,
		agent.Model,
		map[string]any{
			"max_tokens":       dailyMemoryMaxTokens(agent),
			"temperature":      0,
			"prompt_cache_key": agent.ID + ":daily-memory",
		},
	)
	if err != nil {
		return fmt.Errorf("daily memory flush: llm: %w", err)
	}
	if resp == nil || strings.TrimSpace(resp.Content) == "" {
		return fmt.Errorf("daily memory flush: empty llm response")
	}

	parsed, err := parseDailyMemoryFlushResponse(resp.Content)
	if err != nil {
		return err
	}
	entries := cleanDailyMemoryEntries(parsed.Entries)
	if !parsed.ShouldWrite || len(entries) == 0 {
		return nil
	}

	content := r.formatDailyMemoryEntry(reason, entries)
	r.mu.Lock()
	defer r.mu.Unlock()
	return agent.ContextBuilder.memory.AppendToday(content)
}

func (r *DailyMemoryRecorder) buildFlushPrompt(
	summary string,
	messages []providers.Message,
	reason string,
) string {
	now := r.now()
	var sb strings.Builder
	sb.WriteString("You are PicoClaw's daily memory extractor.\n")
	sb.WriteString("Decide whether the conversation contains high-signal information worth writing to today's daily note.\n\n")
	sb.WriteString("Write only durable facts: explicit user preferences, important decisions, plans, and meaningful work, health, finance, or life status changes.\n")
	sb.WriteString("Do not write ordinary Q&A, greetings, transient debugging details, tool output details, or low-value context.\n")
	sb.WriteString("Do not copy the conversation verbatim. Return at most 5 concise bullet-style entries.\n")
	sb.WriteString("If nothing is worth recording, return should_write=false.\n\n")
	sb.WriteString("Return ONLY JSON in this exact shape:\n")
	sb.WriteString(`{"should_write":true,"entries":["entry 1","entry 2"]}`)
	sb.WriteString("\n\n")
	fmt.Fprintf(&sb, "Flush reason: %s\n", cleanDailyMemoryReason(reason))
	fmt.Fprintf(&sb, "Current date: %s\n", now.Format("2006-01-02"))
	if summary != "" {
		sb.WriteString("\nExisting conversation summary:\n")
		sb.WriteString(utils.Truncate(summary, dailyMemoryFlushMaxChars/3))
		sb.WriteString("\n")
	}
	sb.WriteString("\nRecent conversation messages:\n")
	sb.WriteString(formatDailyMemoryMessages(messages, dailyMemoryFlushMaxChars))
	return sb.String()
}

func (r *DailyMemoryRecorder) formatDailyMemoryEntry(reason string, entries []string) string {
	reason = cleanDailyMemoryReason(reason)
	var sb strings.Builder
	fmt.Fprintf(&sb, "## %s · %s\n\n", r.now().Format("15:04"), reason)
	for _, entry := range entries {
		fmt.Fprintf(&sb, "- %s\n", entry)
	}
	return strings.TrimRight(sb.String(), "\n") + "\n"
}

func dailyMemoryMaxTokens(agent *AgentInstance) int {
	if agent == nil || agent.MaxTokens <= 0 {
		return dailyMemoryFlushMaxTokens
	}
	if agent.MaxTokens < dailyMemoryFlushMaxTokens {
		return agent.MaxTokens
	}
	return dailyMemoryFlushMaxTokens
}

func dailyMemoryRelevantMessages(history []providers.Message) []providers.Message {
	if len(history) == 0 {
		return nil
	}
	filtered := make([]providers.Message, 0, min(len(history), dailyMemoryFlushMaxMessages))
	for i := len(history) - 1; i >= 0 && len(filtered) < dailyMemoryFlushMaxMessages; i-- {
		msg := history[i]
		if msg.Role != "user" && msg.Role != "assistant" {
			continue
		}
		msg.Content = strings.TrimSpace(msg.Content)
		if msg.Content == "" {
			continue
		}
		filtered = append(filtered, msg)
	}
	for i, j := 0, len(filtered)-1; i < j; i, j = i+1, j-1 {
		filtered[i], filtered[j] = filtered[j], filtered[i]
	}
	return filtered
}

func formatDailyMemoryMessages(messages []providers.Message, maxChars int) string {
	var sb strings.Builder
	for _, msg := range messages {
		line := fmt.Sprintf("%s: %s\n", msg.Role, strings.TrimSpace(msg.Content))
		if sb.Len()+len(line) > maxChars {
			remaining := maxChars - sb.Len()
			if remaining > 0 {
				sb.WriteString(utils.Truncate(line, remaining))
			}
			break
		}
		sb.WriteString(line)
	}
	return sb.String()
}

func parseDailyMemoryFlushResponse(content string) (dailyMemoryFlushResponse, error) {
	content = strings.TrimSpace(content)
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return dailyMemoryFlushResponse{}, fmt.Errorf("daily memory flush: response is not JSON")
	}

	var parsed dailyMemoryFlushResponse
	if err := json.Unmarshal([]byte(content[start:end+1]), &parsed); err != nil {
		return dailyMemoryFlushResponse{}, fmt.Errorf("daily memory flush: parse JSON: %w", err)
	}
	return parsed, nil
}

func cleanDailyMemoryEntries(entries []string) []string {
	if len(entries) == 0 {
		return nil
	}
	cleaned := make([]string, 0, min(len(entries), dailyMemoryFlushMaxEntries))
	for _, entry := range entries {
		entry = strings.TrimSpace(entry)
		entry = strings.TrimPrefix(entry, "-")
		entry = strings.TrimPrefix(entry, "•")
		entry = strings.TrimSpace(entry)
		entry = strings.Join(strings.Fields(entry), " ")
		if entry == "" {
			continue
		}
		entry = utils.Truncate(entry, dailyMemoryFlushMaxEntrySize)
		cleaned = append(cleaned, entry)
		if len(cleaned) >= dailyMemoryFlushMaxEntries {
			break
		}
	}
	return cleaned
}

func cleanDailyMemoryReason(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return "clear"
	}
	reason = strings.Join(strings.Fields(reason), "-")
	return strings.ToLower(reason)
}

func logDailyMemoryFlushError(reason string, err error) {
	if err == nil {
		return
	}
	logger.WarnCF("agent", "Daily memory flush skipped", map[string]any{
		"reason": cleanDailyMemoryReason(reason),
		"error":  err.Error(),
	})
}
