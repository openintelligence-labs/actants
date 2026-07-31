# Security policy

## Reporting a vulnerability

If you find a security issue in actants — the agent runtime, the tool-calling layer, the MCP/A2A servers, the provider adapters, or the caching layer — please **do not open a public issue**.

File a private report through GitHub Security Advisories:

> **<https://github.com/openintelligence-labs/actants/security/advisories/new>**

Include:

1. The component affected (e.g. "MCP server", "tool dispatch", "Anthropic provider adapter").
2. A reproduction case — minimal code that demonstrates the issue.
3. The impact you've observed or believe is possible.
4. Whether you've already disclosed the issue elsewhere.

We aim to acknowledge within 48 hours and to publish a fix (or a detailed mitigation) within 30 days. For high-severity issues we'll request a coordinated disclosure window — typically 90 days from first report.

## Supported versions

| Version | Status |
|---|---|
| 0.5.x (current) | Supported. Fixes land in the latest patch. |
| < 0.5 | Unsupported. Please upgrade. |

actants is the shared backbone of every Open Intelligence Labs project, so a security fix here is cut as a release immediately rather than batched.

## Scope

**In scope:**

- Arbitrary code execution through tool dispatch, the tool-schema parser, or deserialization of provider responses.
- Prompt-injection paths that let untrusted tool output escalate into unintended tool calls without the caller opting in.
- Credential leakage — API keys appearing in logs, traces, exception messages, cached artifacts, or OTel spans.
- Cache poisoning: one caller reading or influencing another caller's cached completions.
- MCP or A2A server flaws allowing unauthorized tool invocation, path traversal, or filesystem mutation beyond documented behaviour.
- SSRF via user-controllable `base_url` handling in provider adapters where the library should have constrained it.

**Out of scope** (working as designed):

- An agent calling a tool you registered. Tool registration is an explicit grant of capability; actants does not sandbox tool bodies. Sandboxing is the caller's responsibility.
- LLM output being wrong, biased, or unsafe. Model behaviour is not a library vulnerability.
- Pointing a provider at an arbitrary `base_url` you configured yourself.
- Secrets you place in a prompt reaching the provider you configured. That's the documented data flow.

## Data flow

actants is local-by-default and ships zero telemetry. It makes exactly one class of outbound request: to the LLM/embedding provider you configure. Default is a local Ollama endpoint, which makes no network egress off-device. Nothing is phoned home, ever — if you observe an outbound request to any host you did not configure, that is a security bug and we want to hear about it immediately.
