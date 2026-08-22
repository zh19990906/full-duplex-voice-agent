# Project Overview

## Vision

Build a GPT Live-like realtime full duplex voice agent through modular models and engineering orchestration.

## Non Goals

The initial phase does not attempt to train an end-to-end speech-to-speech foundation model.

## Target Cases

### Case 1: Backchannel

User feedback such as "嗯嗯" should not stop assistant generation.

### Case 2: Interrupt and Revision

User interruption should stop current output, update intent and regenerate.

### Case 3: Translation

Support streaming speech translation.

### Case 4: Resume Task

Long running tasks must maintain explicit state and resume after interruption.

## Core Components

- Speech frontend
- Turn manager
- Conversation controller
- Generation manager
- Task state manager
- Model adapters

The controller layer is the core differentiator.
