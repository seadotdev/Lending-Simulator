# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, use [GitHub's private vulnerability reporting](https://github.com/seadotdev/Lending-Simulator/security/advisories/new) to submit a report. Include:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Scope

This project is a simulation and benchmarking tool. Key areas of concern include:

- **API key handling** — keys should never be logged, committed, or exposed in output
- **LLM prompt injection** — malicious borrower data that could manipulate model behavior
- **Dependency vulnerabilities** — outdated packages with known CVEs

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest on `main` | Yes |
| Older branches | No |
