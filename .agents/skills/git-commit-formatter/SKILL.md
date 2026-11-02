---
name: git-commit-formatter
description: Formats, generates, and validates Git commit messages adhering to Conventional Commits specifications and repository standards. Use this skill whenever writing, reviewing, formatting, or creating git commit messages.
---

# Git Commit Formatter Skill

This skill provides standard operating procedures, formatting rules, and validation tools for crafting clean, consistent, and informative Git commit messages following the [Conventional Commits v1.0.0](references/conventional-commits.md) specification.

---

## 1. Commit Message Structure

Every commit message must follow this structure:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Key Formatting Rules

- **Header line**: Maximum 72 characters (recommended <= 50 characters).
- **Type**: Must be lowercase (e.g., `feat`, `fix`, `docs`).
- **Scope**: Optional, lowercase, enclosed in parentheses (e.g., `feat(dataset): ...`).
- **Separator**: Must have a colon and space `: ` after type/scope.
- **Description**:
  - Use the **imperative mood** ("add", not "added" or "adds"; "fix", not "fixed").
  - Start with a lowercase letter (unless a proper noun or acronym).
  - Do **not** end with a period (`.`).
- **Blank lines**: Must separate header, body, and footers with a single empty line.
- **Body & Footer lines**: Wrap at 72 characters.

---

## 2. Commit Types

| Type | Purpose | Example |
| :--- | :--- | :--- |
| `feat` | Introduces a new feature or functionality | `feat(auth): add OAuth2 login support` |
| `fix` | Fixes a bug | `fix(loader): resolve tensor dimension mismatch in STARE` |
| `docs` | Documentation-only changes | `docs(readme): update installation requirements` |
| `style` | Formatting, whitespace, missing semicolons (no code logic change) | `style(dataset): format imports and line breaks` |
| `refactor` | Code change that neither fixes a bug nor adds a feature | `refactor(models): simplify forward pass computation` |
| `perf` | Code change that improves performance | `perf(dataloader): optimize image prefetching` |
| `test` | Adding or updating tests | `test(eval): add unit test for dice score metric` |
| `build` | Changes affecting build system or external dependencies | `build(deps): bump torchvision to 0.15.2` |
| `ci` | Changes to CI configuration files and scripts | `ci(actions): add GPU runner workflow` |
| `chore` | Routine tasks, maintenance, configuration tweaks | `chore(gitignore): ignore generated checkpoints` |
| `revert` | Reverts a previous commit | `revert: revert "feat(auth): add OAuth2 login support"` |

---

## 3. Workflow for Writing a Commit

### Step 1: Inspect Changes
Always inspect what is modified before writing the message:
```bash
# Check status of modified and untracked files
git status -s

# View diff of staged changes
git diff --staged
```

### Step 2: Ensure Atomic Commits
- Commit only one logical change per commit.
- If changes address multiple unrelated things, split them into separate commits.

### Step 3: Choose Type and Scope
- Select the `type` from the table above.
- Identify the `scope` from the module, file, or component affected (e.g., `dataset`, `adapter`, `losses`, `cli`).

### Step 4: Draft the Message
- Write a concise imperative subject.
- If the reason for the change is not obvious from the diff, add a body explaining:
  - **Why** the change is needed.
  - **What** approach was chosen.
  - Any side effects or migration steps.

### Step 5: Handle Breaking Changes & Issue Tracking
- **Breaking changes**:
  - Add `!` before the colon: `feat(api)!: drop legacy python 3.8 support`
  - Or add a footer starting with `BREAKING CHANGE: <explanation>`
- **Issue references**:
  - Close issues using `Fixes #123`, `Closes #456`, or `Resolves #789` in the footer.

### Step 6: Validate the Commit Message
Run the validation script on the draft message:
```bash
bash .agents/skills/git-commit-formatter/scripts/validate-commit-msg.sh "feat(dataset): add data augmentations"
```

### Step 7: Create the Commit
Use standard Git commands:
```bash
# For single-line header:
git commit -m "feat(dataset): add data augmentations"

# For header with body:
git commit -m "feat(dataset): add data augmentations" -m "Include random horizontal flip and color jitter transforms."
```

---

## 4. Good vs. Bad Examples

### Good Commit Messages
```text
feat(loss): add focal dice loss for class imbalance
```
```text
fix(dataset): revert DataLoader num_workers to default

Multi-process loading caused shared memory crashes in containerized
environments with low shm limits. Revert to single worker per process.

Fixes #42
```
```text
refactor(sam): decompose prompt encoder into modular layers

BREAKING CHANGE: PromptEncoder constructor now requires explicit mask_size argument.
```

### Bad Commit Messages
- `fix bug` *(Too vague, missing type/scope/context)*
- `Added STARE dataset loaders.` *(Past tense, capitalized, ends with period, missing type)*
- `WIP` *(Never commit unfinished, uninformative commit messages)*
- `feat: update everything and fix training and update docs` *(Not atomic; should be multiple commits)*

---

## 5. Supplementary Resources

- **Template**: [.agents/skills/git-commit-formatter/resources/commit-template.txt](resources/commit-template.txt)
- **Specification**: [.agents/skills/git-commit-formatter/references/conventional-commits.md](references/conventional-commits.md)
- **Validator Script**: [.agents/skills/git-commit-formatter/scripts/validate-commit-msg.sh](scripts/validate-commit-msg.sh)
