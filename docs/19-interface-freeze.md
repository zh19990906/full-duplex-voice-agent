# Interface Freeze Specification

## Purpose

This document freezes the core engineering contracts before implementation begins.

The goal is to allow Codex agents and developers to work in parallel without breaking module boundaries.

## Frozen Principles

1. Models are replaceable components.
2. All realtime decisions go through Conversation Controller.
3. Modules communicate through explicit events and interfaces.
4. Business logic must not be embedded inside model adapters.

## Core Interfaces

### Event Bus

All realtime interactions use the event protocol defined in `docs/16-realtime-event-protocol.md`.

### Model Adapters

Required adapters:

- TurnAdapter
- ASRAdapter
- LLMAdapter
- TTSAdapter
- TranslationAdapter

Each adapter must hide backend implementation details.

## Controller Contract

Conversation Controller:

Input:
- realtime events
- current conversation state
- task state

Output:
- actions
- state transitions
- generation commands

## Implementation Rule

Before changing a public interface, create an Architecture Decision Record.
