# Realtime Event Protocol V1

## Scope

V1 freezes the typed contracts used by the single-machine realtime runtime.
New internal domain events and browser transport events use these contracts;
the legacy `BaseEvent` types remain available for existing callers.

## Event envelope

Every V1 domain or transport event is a `RealtimeEnvelope` with
`protocol_version` set to `1`. Its stable fields are `event`, `event_id`,
`session_id`, `sequence`, `capture_timestamp`, `server_timestamp`,
`response_id`, `generation_epoch`, `segment_id`, and `payload`. `response_id`
and `segment_id` may be `null` before a response or segmented output exists.
Consumers must reject another protocol version rather than guessing its
meaning.

`session_id`, `response_id`, `generation_epoch`, `segment_id`, `sequence`,
and capture and server timestamps make ordering, cancellation, and
stale-output rejection explicit at every V1 event boundary.

## Browser PCM frame

Browser-to-server audio is a binary header followed by a PCM16 payload. The
header is one network-byte-order `struct.Struct("!BIdIB")` layout:

| Field | Encoding |
| --- | --- |
| protocol version | unsigned 8-bit integer |
| sequence | unsigned 32-bit integer |
| capture timestamp | IEEE-754 64-bit float |
| sample rate | unsigned 32-bit integer |
| channels | unsigned 8-bit integer |

The payload is signed PCM16 and V1 requires exactly one 20 ms frame: 320
samples, or 640 PCM bytes. Input is fixed to 16 kHz mono. Encoders and
decoders reject truncated headers, unsupported protocol versions, non-16-kHz
or non-mono frames, odd-length payloads, and payloads other than 640 bytes
with `ValueError`. A variable frame duration requires a new protocol version.
