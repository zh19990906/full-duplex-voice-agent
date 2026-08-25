# Realtime Event Protocol V1

## Scope

V1 freezes the typed contracts used by the single-machine realtime runtime.
New internal domain events and browser transport events use these contracts;
the legacy `BaseEvent` types remain available for existing callers.

## Event envelope

Every V1 domain or transport event is a `RealtimeEnvelope` with
`protocol_version` set to `1`. Its stable fields are `event`, `event_id`,
`session_id`, `sequence`, `capture_timestamp`, `server_timestamp`,
`response_id`, `generation_epoch`, and `payload`. `response_id` may be
`null` before a response exists. Consumers must reject another protocol
version rather than guessing its meaning.

`session_id`, `response_id`, `generation_epoch`, `sequence`, and capture and
server timestamps make ordering, cancellation, and stale-output rejection
explicit at every V1 event boundary.

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

The payload is signed PCM16, so it must have an even byte length. Input is
fixed to 16 kHz mono. Browser capture normally sends 20 ms frames (320
samples, 640 PCM bytes); this codec preserves an even-length payload and the
ingress layer enforces sequencing and frame timing. Decoders reject truncated
headers, unsupported protocol versions, non-16-kHz or non-mono frames, and
odd-length payloads with `ValueError`.
