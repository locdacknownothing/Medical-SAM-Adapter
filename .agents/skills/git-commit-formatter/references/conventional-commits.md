# Conventional Commits v1.0.0 Reference

The Conventional Commits specification is a lightweight convention on top of commit messages. It provides an easy set of rules for creating an explicit commit history, which makes it easier to write automated tools on top of.

---

## 1. Syntax Specification

The commit message should be structured as follows:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Components

1. **`type`** (Mandatory):
   - A noun describing the kind of change made.
   - Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
   - `feat` MUST be used when a commit adds a new feature.
   - `fix` MUST be used when a commit represents a bug fix.

2. **`scope`** (Optional):
   - A noun describing a section of the codebase surrounded by parenthesis, e.g., `fix(parser): ...`
   - Should be concise and represent the subsystem or module affected.

3. **`!` (Breaking Change Marker)** (Optional):
   - A `!` MAY be appended directly after type/scope, e.g., `feat(api)!: change response schema` or `feat!: remove deprecated flags`.

4. **`description`** (Mandatory):
   - A short summary of code changes immediately following the colon and space.
   - Written in imperative mood ("change", not "changed" or "changes").
   - Starts lowercase, no trailing period.

5. **`body`** (Optional):
   - Free-form text explaining the motivation for the change and contrasting this with previous behavior.
   - Separated from header by a single blank line.
   - Should wrap around 72 characters.

6. **`footer`** (Optional):
   - One or more footers, each separated by a blank line from the body.
   - Each footer MUST consist of a word token followed by either `:<space>` or `<space>#`, e.g.:
     - `BREAKING CHANGE: environment variable DATA_DIR is deprecated; use DATASET_ROOT instead.`
     - `Closes #123`
     - `Reviewed-by: Z`

---

## 2. Breaking Changes

A breaking change can be signaled in two ways:
1. In the type/scope prefix with an exclamation mark (`!`):
   ```text
   feat(runtime)!: drop support for Python 3.8
   ```
2. In the footer with `BREAKING CHANGE:` uppercase followed by space and explanation:
   ```text
   refactor(api): change method signature of load_weights

   BREAKING CHANGE: load_weights now requires keyword-only arguments.
   ```

---

## 3. Best Practice Checkpoints

- **Atomic Commits**: Keep each commit focused on one cohesive task or bugfix.
- **Tone**: Use imperative mood ("refactor", "add", "fix", "update").
- **Clarity**: State *why* the change was made in the body if the reasoning is not trivial.
- **Referencing Issues**: Reference relevant issue tracker tickets in footers.
