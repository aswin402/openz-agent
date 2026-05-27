# Cryptographic Merkle Audit Trails

ZeroClaw execution logs and tool receipts are protected against tampering using sequential SHA-256 hash chaining and HMAC-SHA256 signatures.

---

## Technical Architecture

### 1. Hash-Chaining
Every log event written to the append-only JSONL log file (`runtime-trace.jsonl`) is chained to the previous log event's hash.

For event $i$:
$$\text{Hash}_i = \text{SHA256}(\text{Hash}_{i-1} \mathbin{\Vert} \text{SerializedEventWithoutHashOrSignature}_i)$$

This creates a Merkle-like dependency chain. If an operator or external actor modifies a message, timestamp, outcome, or attribute of a log line:
1. The modified line's calculated hash will mismatch its recorded `hash` field.
2. Every subsequent line in the log file will fail validation, since the hash chain is broken.

### 2. Cryptographic Signatures
To prevent an attacker from modifying a log line and recalculating the entire hash chain downstream, each hash is cryptographically signed using HMAC-SHA256.

$$\text{Signature}_i = \text{HMAC-SHA256}(\text{Key}, \text{Hash}_i)$$

- The key is a secure, 32-byte workspace key generated dynamically and saved to `.audit_key` in the workspace root.
- The `.audit_key` file is initialized with strict `0600` permissions, ensuring only the daemon or owner can read or write to it.

---

## Log Verification

You can verify the cryptographic integrity of the log file at any time using the `openz` CLI.

### CLI Command

```bash
openz logs --verify
```

### Verification Process
When run, the CLI:
1. Locates the `.audit_key` in the log file's parent or grandparent directory.
2. Sequentially reads the log file from the first line.
3. Decodes each event, computes the expected hash based on the previous line's hash, and matches it with the recorded `hash`.
4. If `.audit_key` is available, verifies the `signature` using HMAC-SHA256. If a mismatch is detected, verification halts immediately and exits with status code `1`.
5. If `.audit_key` is missing, skips signature verification and validates the hash-chain integrity.

---

## Failure Scenarios

Log verification will fail if any of the following occur:
- **Attribute Modification**: If an attacker alters the message, severity, timestamp, or tool execution outcomes of a past log line.
- **Line Deletion/Insertion**: If lines are removed or inserted in the middle of the log file.
- **Signature Deletion**: If the signature field is removed from a log entry while `.audit_key` is present.
- **Key Modification**: If the `.audit_key` file is modified or overwritten with a different key.
