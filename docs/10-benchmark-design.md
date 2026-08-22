# Benchmark Design

The project uses four user scenarios as acceptance tests.

## Case 1: Backchannel

Input:

Assistant speaking.

User:

```
嗯嗯
```

Expected:

```
continue_generation=true
stop_tts=false
```

## Case 2: Interrupt Revision

Input:

```
等等，我想问上海
```

Expected:

```
stop_current_response=true
intent_updated=true
regenerate=true
```

## Case 3: Translation

Metrics:

- first audio latency
- translation quality
- stability

## Case 4: Task Resume

Input:

```
continue
```

Expected:

Task state restored and execution continues.

## Metrics

- end-to-end latency
- interruption latency
- recovery accuracy
- task completion accuracy
