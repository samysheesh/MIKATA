# MIKATA — the agent core: a rebuild starting from the LLM call

October 13, 2026 · approved the same evening (decisions in section 6) · implemented in phases

This document proposes rebuilding on sound foundations everything that happens **after
transcription**: understanding the request, acting on the PC, checking, learning. Audio capture,
the wake word, transcription, the voice and the interface (phases 0 to 11) do not change.

Once approved, it replaces sections 6 and 7 of `docs/ARCHITECTURE.md` and the
`AGENT_V2.md` plan (which only exists on the `agent-v2-plan` branch).

## 1. Measured findings

Measurements taken on October 13, 2026 on the PC, from `data/logs/mikata.log` and `data/agent2/traces/`.

| | Tool loop (phases 4 to 12) | Agent v2 (phase 13) |
| --- | --- | --- |
| Commands measured | 230, varied | 15, 13 of them the same request |
| Settled in a single LLM call | 61%, median first sound 2.1 to 2.4 s | none |
| Multi-step tasks | 31 commands, 15.5 s median | 41.7 s median (25 to 81 s), 8.1 calls |
| Success | not verified (no check of the result) | 9 out of 15 |
| Learning | local replay seen once in the logs | 0 skills (8 empty compilations, 1 refusal) |

Other established facts:

- **Generated code runs in the main process.** `agent2/sandbox.py` runs it through
  `exec`; a `while True: pass` freezes MIKATA despite `timeout_s=1` (verified).
- **The v2 prompts are written for a single request** (the `youtube.com/watch` address,
  "le dernier, pas un film" ("the latest one, not a movie")).
- **The 16/20 test suite runs on a simulator** that was tuned at the same time as the cases.
- **CI has been red on `main`** since October 11 (28 pushes), including two tests of the guard.
- **56 tools, about 5,600 tokens of schemas** are sent with every LLM call.

Conclusion: the original loop is fast on the common case but verifies nothing and
learns almost nothing; v2 has the right ideas (code as action, verification,
skills) in a structure that multiplies calls and has not proven itself.

**Benchmark baseline (phase 14a).** On the bench's 49 tasks, with a simulated PC and the
real LLM, the original loop succeeds on 47 tasks (96%), with a 2.2 s median and 2 LLM calls;
long chains take 4.6 s and 2 to 5 calls. Both failures came from the bench itself.
So the new core is not justified by those requests, which it merely must not
degrade, but by what the original loop does not guarantee: security by construction
(it resisted booby-trapped content only because the model was careful), verification
of the result, learning, and holding up on the real web and real interfaces.

## 2. What the research says

Each principle adopted rests on a published result. References in section 8.

| # | Principle | Result it rests on |
| --- | --- | --- |
| 1 | **A single loop, driven by the model** | Planning accounts for 50 to 75% of latency and agents take 2.7 to 4.3 times too many steps (OSWorld-Human). "Look for the simplest solution; add complexity only when it improves results" (Anthropic). |
| 2 | **Code as action, for anything that chains steps** | Up to 20 points more success and 30% fewer actions than JSON calls (CodeAct). Model calls 10.2 → 1.8 and latency 237 → 118 s (ActionEngine). Not for a single action (Anthropic). |
| 3 | **A small, carefully designed action space** | Few high-level tools, actionable errors, call examples: 72 → 90% on complex parameters. Load 3 to 5 tools and defer the rest: 49 → 74% (Anthropic). |
| 4 | **API first, UI as a fallback** | A unified API + UI layer: +6 to +8 points. 62% of failures come from control detection. Several validated actions per call: up to 51% fewer steps (UFO²). |
| 5 | **Verify with cheap postconditions** | Deterministic checks after each action, costly reflection only after a failure (Efficient GUI Agents survey). Grade the final state, not the path (Anthropic). |
| 6 | **Skills = verified programs, in the action space** | Callable verified program 40.1% versus unverified text 32.6% (ASI). A package revisable after failure: +10.5 points on OSWorld; without an isolated critic −8.6 points (EvoSkill-GUI). |
| 7 | **Security by construction** | "Once untrusted content has been read, it must be impossible for it to trigger a consequential action" (Design Patterns). 0 successful attacks out of 949, 77% of tasks solved versus 84% without a defense (CaMeL). |
| 8 | **Fast and slow in parallel** | Fast responder + slow reasoner (Talker-Reasoner). A filler phrase hides the delay but talks over the user in 48% of cases if it is systematic (Full-Duplex-Bench-v3). |
| 9 | **Context is a scarce resource** | Stable cached prefix, just-in-time retrieval, canonical examples rather than rules (Anthropic). |
| 10 | **Measure before optimizing** | 20 to 50 real tasks are enough to start; deterministic graders; `pass^k` for consistency (Anthropic). Report cost alongside success (survey). |
| 11 | **Learn at several levels, from safest to riskiest** | A lesson playbook updated in small increments: 42.4 → 59.4% on AppWorld, with a DeepSeek model without reasoning (ACE). A code library reused on new tasks (Voyager). Tools created on the fly: 75% on GAIA (Alita). |
| 12 | **Bound self-evolution** | Memory that accumulates without safeguards drops the refusal rate from 99.4 to 54.4%; 56 to 76% of tools freely created or reused are unsafe (*Your Agent May Misevolve*). |

Two caveats drawn from the same sources:

- **CaMeL has a cost**: 2.7 to 2.8 times more tokens than native tool calling, and the
  planner cannot decide based on content it is not allowed to read.
- **Skills can hurt**: negative transfer on short, direct tasks, and
  no verifier prevents a revision from degrading a skill (EvoSkill-GUI). Only
  15.6% of induction attempts pass verification (ASI).

## 3. Target architecture

### 3.1 Overview

```text
command (transcribed text)
   │
   ├─ 0. local replay of an active skill ────────────────────► actions        0 LLM calls
   │
   └─ 1. LLM call (fast model, streamed)
          ├─ text ───────────────────────────────────────────► speech         1 call
          ├─ direct tool call ───────────────────────────────► action         1 call
          └─ script ─► interpreter ─► SDK ─► policies ─► tools ─► PC          1 call + extractions
                            │                    │
                       provenance            confirmation or refusal
   │
   └─ trace ─► background learning (induction, verification, revision)
```

The boundary with the rest of MIKATA does not move: `Agent.respond(command, speech, on_phase,
listen, follow_up) -> TurnResult`. `app.py`, the audio and the interface see no
difference.

### 3.2 The single loop

A turn has three possible outcomes, chosen by the model in one call:

1. **Text**: a question, an explanation, small talk. Spoken as a stream, as today.
2. **Direct call**: a simple action (open, volume, pause, timer). The tool runs,
   "C'est fait" ("Done") comes from the cache. This is the path of 61% of current commands: it must
   not get slower.
3. **Script**: as soon as the request chains steps, depends on information to be found, or loops.
   The model writes a short program against the SDK; the interpreter runs it in one go.

There are no longer separate "intent", "unknowns", "channel" and "plan" stages: they are
lines of the script. Example, for "lance le dernier opening de One Piece" ("play the latest One Piece opening"):

```python
resultats = web_search("dernier opening One Piece titre")
info = extract(resultats, {"titre": str},
               "Quel est le titre du dernier opening de One Piece ?")
expect(info.titre, "un titre a été trouvé")
browser_open("https://www.youtube.com/results?search_query=" + info.titre)
browser_click(info.titre)
return f"C'est lancé : {info.titre}."
```

Two LLM calls (the script, the extraction) instead of eight. If the script fails, the error
goes back to the model along with what has already been done, and it corrects course; after two failures
the reasoning level goes up, after three the turn is closed to tools and MIKATA explains.

What the script returns decides what happens next: a short sentence is spoken as is (a single
LLM call); data, or whatever it output with `print`, goes back to the model, which phrases the answer.

### 3.3 The action space

- **The existing tools remain the implementation.** One tool = one `@register_tool` class
  (project rule unchanged). The SDK is a typed facade generated from the registry.
- **Each tool declares its effect**: `lire` (read), `agir` (act, reversible), `conséquent`
  (consequential, i.e. outgoing or hard to undo: shell, writing, sending, keyboard input),
  `destructif` (destructive, forbidden).
- **Each tool returns a structured result** (`text` for the model, `data` for the
  script): the script chains on values, not on text to be re-read. This replaces the
  `publish_data` side channel.
- **In a program, a tool is called by its tool name**, with the same arguments:
  `set_volume(level=30)`, `move_file(source, destination)`. The model has nothing new to
  learn, and the program guide lists only the tools that return an object with
  fields. The SDK names (`audio.volume`, `files.move`) remain available: they are the ones
  skills use (phase E).

  Deviation measured in phase 14d from the first version of this document, which planned
  to replace the JSON schemas with an SDK reference: with two names for the same
  action, the model mixed up arguments and its programs failed. Keeping the native
  schemas and the same names removed those failures; reducing the number of schemas sent
  remains a possible optimization, to be done under the bench's control.
- **Channel order**, from most to least reliable: dedicated tool, browser (DOM),
  accessibility tree, keyboard and mouse, then vision (later, as a fallback).
- **Filtered observations**: actionable controls only, bounded, and differences
  since the last observation rather than the whole screen.

### 3.4 The script interpreter

Generated code is no longer passed to `exec`. An in-house interpreter evaluates a subset of
Python (assignments, conditions, loops, comprehensions, SDK calls), node by node:

- **execution budget** in number of operations and in time: an infinite loop stops;
- **no access** outside the SDK functions: no import, no hidden attribute;
- **provenance** attached to every value and propagated by every operation (section 3.5);
- **pause and resume** at each SDK call: this is where spoken confirmations happen;
- **redacted errors**: a message that depends on untrusted content is not sent back to the
  model as is.

Alternative considered: Monty (Pydantic's Python interpreter written in Rust, February 2026). It is
fast and isolates well, but it is young, has no known independent audit, and does not track
value provenance. Worth keeping as a possible backup engine.

### 3.5 Security: provenance and policies

Three classes of sources:

| Class | Examples |
| --- | --- |
| Trusted | The user's speech, the config, the facts they dictated |
| Local | System state: windows, volume, file names from the index |
| Untrusted | Web pages, search results, file contents, clipboard, text read in an app |

Rules:

1. **The planner does not need to read untrusted content.** The script hands it to
   `extract(...)`: a quarantined LLM call, with no tools, that returns an object with a fixed schema.
   The extracted value stays marked as untrusted.
2. **Policies are checked when the tool is called**, against the tool's effect and the
   provenance of its arguments. Examples: a shell command built from an
   untrusted value is refused; an untrusted address only opens over `http(s)`; a
   message whose recipient comes from a page requires a "oui" ("yes").
3. **If the model has to read untrusted content to decide** (exploring an unknown
   interface, for example), the turn becomes "tainted": consequential actions then require
   spoken confirmation.
4. **Destructive actions remain forbidden**, always (recycle bin only).
5. **One action per read target, in a program** (phase 17f): a program acts only
   once with an argument taken from untrusted content (a result row, a link, a
   control that was read); the next one is stopped before it acts, and the model is told why.
   Several read targets are handled one at a time, through the direct tool, each one seen by the
   model. Seen on October 15: for "lance Jet Lag de Tiakola" ("play Jet Lag by Tiakola"), a program clicked "Play"
   on each of the 18 results that mentioned the title (the music restarted 18 times in 10 s).
   An instruction in the program guide ("une chose demandée, une action", "one thing asked, one action") made
   other bench tasks fail ("ouvre la page et dis-moi…" ("open the page and tell me…") stopped after opening,
   0/3): the rule lives in the code only.

`agent/guard.py` remains the rule library (blacklist, allowed folders,
recycle bin, backups) and its phase 5 tests remain the criterion. It becomes a safety net
behind the policies, no longer the only barrier.

What this design does not stop: a summary skewed by a page (nothing is executed, but
what is said may be false), and phishing aimed at the user themselves.

### 3.6 Skills

A skill is a **readable package** in `data/skills/<name>/`:

| File | Role |
| --- | --- |
| `SKILL.md` | Metadata: intent, app, platforms, parameters, keywords, status, version, counters |
| `skill.py` | A `run(sdk, **params)` function in the interpreter's subset |
| `recover.md` | Known recovery rules (pop-up window, moved control) |
| `failures/` | At most three failure cases, for revision |

- **Induction**: only from a trace whose postconditions succeeded. Values
  that came from the request become parameters.
- **Verification before adoption**: the original request is replayed with the skill; the
  task must succeed, the skill must be called, and every call must have an
  effect. For a skill with consequential effects (sending something), the replay runs on the
  recorded results, and the first real use counts as a supervised trial.
- **In the action space**: `skills.<name>(...)` can be called from a script or directly.
- **Retrieval by metadata**, with a threshold; nothing is suggested for a single action.
- **Local replay**: if the phrase matches an active skill's pattern and its parameters
  bind, it runs without an LLM call (the idea from phase 12e, kept).
- **Revision**: after a failure, an isolated critic (it sees only the request, the
  observations and the actions) makes a diagnosis; the revision produces a new version,
  and the old one is kept. Lifecycle: proposed, on trial, active, demoted.
- **Agent Skills–compatible format** (`SKILL.md`, an open standard from December 2025) for
  sharing. An imported skill arrives as "proposed" and is reviewed before use: 341 of the
  2,857 skills audited on the OpenClaw marketplace were malicious.

### 3.7 Memory

Plain files, no external framework: memory frameworks' scores are measured by
their own vendors, and a 2026 comparison reports, for the same system, 94%
claimed versus 49% measured by a third party (gap not verified at the source).

- **Facts** dictated by the user (the existing `data/memory.json`).
- **Preferences** learned, with a confidence level ("music → Spotify").
- **Episodes**: summaries of recent traces.
- Every entry carries its **provenance**: nothing that comes from untrusted content
  becomes a fact without going through the user.

### 3.8 Latency: four paths, one budget each

Latency is the project's number 1 priority. Each request takes the shortest path that
is enough, and each path has a budget measured by the benchmark.

| Path | When | LLM calls | Target budget |
| --- | --- | --- | --- |
| **0. Local replay** | The phrase matches an active skill, or the router recognizes a learned skill (phase 17f) | 0 (Jev: ~0.4 s, no generation) | under 1 s, plus the gestures themselves |
| **1. Direct** | Question, small talk, simple action | 1 | first sound under 2 s |
| **2. Script** | The request chains steps or depends on information | 1 + extractions | under 10 s median |
| **3. Exploration** | Unknown interface, to be observed step by step | 8 steps at most | under 30 s, in the background |

What keeps these budgets:

- **A single call to choose the path**: no understanding stage before acting.
- **A script groups the actions**: one call instead of one per step, and independent reads
  run in parallel (`parallel(...)`).
- **Deterministic postconditions**: no verification by a second LLM call on the
  normal path. The cost of any verification counts against the budget.
- **In exploration, several validated actions per step** (up to 51% fewer steps in
  UFO²).
- **Stable context prefix** for the prompt cache, filtered observations, extraction
  handed to the fast model.
- **All learning off the critical path**: induction, verification and revision
  run in the background, never during the response.
- **What exists is kept**: connection prewarmed on the wake word, speculative
  transcription, cached phrases, thinking sound.

Long tasks (paths 2 and 3 beyond a few seconds):

- they become **background tasks** with a state: you can ask how far along they are or
  cancel them while another command runs;
- **the acknowledgment is only spoken if the expected wait exceeds a threshold**, and it goes out at
  the same time as the action, never before (a systematic filler phrase talks over the
  user in 48% of cases, Full-Duplex-Bench-v3). Each tool declares its typical
  duration.

Non-regression rule: a phase that slows down path 1 is not merged.

Idea kept for later: executing the most likely action ahead of time, with the fast model
(up to 20% less latency, *Speculative Actions*).

### 3.9 Models and context

- **Two configurable roles**, `llm.fast` and `llm.strong`, on the existing OpenAI-compatible
  client. Fast for the short path and extraction; strong for a new script after a
  failure and for induction. Both can point to the same provider.
- **Stable prefix** first (system prompt, direct tools, SDK reference), then memory,
  candidate skills, PC state, history, command.
- **Prompts at the right level**: principles and a few canonical examples, no instruction
  written for one specific case.

### 3.10 Evaluation

A benchmark lives in the repository (`mikata/eval/`, tasks in `tests/eval/`):

- **A task**: the request, the starting state, deterministic **final-state** graders
  (volume at 30, file present, tab on a given address), tags.
- **Two modes**: dry run on a simulated registry (CI, non-regression of the plumbing) and
  live on the PC (the real measurement).
- **Metrics**: success over three trials (`pass^3`), first sound, duration, LLM calls, tokens,
  cost.
- **Balanced set**: cases where MIKATA must act, and cases where it must not (booby-trapped
  pages served locally: zero consequential actions expected).
- **Starting point**: 30 to 40 tasks drawn from real commands and from the target requests.

### 3.11 Self-improvement: four levels, from safest to riskiest

Principle kept from `AGENT_V2.md`: **stable core, learned periphery**. The code in `mikata/`
is written and reviewed by humans; what MIKATA learns lives in `data/`, readable,
versioned, and can be switched off.

| Level | What improves | Method | Place in MIKATA |
| --- | --- | --- | --- |
| **1. Lesson playbook** | Short notes per app or domain ("in Discord, the mic button is called 'Rendre muet'") | ACE: small entries with "helpful / harmful" counters, added and edited one at a time, never rewritten wholesale | `data/playbook/`, phase E |
| **2. Skills** | Verified programs, reused and revised | ASI, EvoSkill-GUI, Voyager (section 3.6) | `data/skills/`, phase E |
| **3. New capabilities** | A missing tool | Alita: the agent writes the tool itself | A skill composed of SDK primitives is enough in most cases; a truly new primitive (code with system access, an MCP server) stays disabled until validated, phase F |
| **4. The core code** | The prompts and code of `mikata/` | Darwin Gödel Machine: 20 → 50% on SWE-bench in 80 iterations, sandboxed and under human supervision | Never live. Offline only: the benchmark acts as judge, and the result is a proposal reviewed by a human |

Why the playbook is rewritten in small increments: in ACE, a full rewrite shrank a
context from 18,282 to 122 tokens in one step, and pushed success below the level with no
learning.

Self-evolution safeguards, drawn from *Your Agent May Misevolve*:

1. **What is learned never touches the policies or the guard.** They live in the
   code, outside the learned periphery.
2. **A skill runs in the interpreter, under the same policies** as a script: it
   cannot do anything the primitives do not allow.
3. **The success signal comes from a postcondition or from a correction by the user**,
   never from the agent grading itself. With an unreliable signal, learning degrades
   (−3.4 points in ACE). Phase 17f: for actions in an app, the postcondition is
   generic — did the window change? (`mikata/uia/effects.py`, read again while MIKATA
   is speaking); a skill whose action changed nothing fails and gets revised without the
   user having to say so.
4. **Lessons enter the context as references, not as rules** (this change alone
   reduces unsafe behavior in the study).
5. **Nothing is promoted to "active" without passing the benchmark again**; everything is versioned and
   reversible.
6. **Any code from outside is reviewed before use**: faced with malicious tools, the
   models tested refuse only 0.3 to 7% of them.

### 3.12 Tasks never seen before

No architecture guarantees "every task": coverage depends first on the model
and on what it perceives. What research shows to be most effective, in the order MIKATA
applies it:

1. **General primitives and code** rather than a catalog of cases. Shell, files,
   browser, accessibility tree, keyboard and mouse, web search: anything that can be done
   by command, by API or through the interface can be written with these building blocks (CodeAct; Alita:
   "minimal predefinition").
2. **Look up knowledge at the moment it is needed.** Facing an unknown app, the script first
   searches the web for "how to do X in Y", then acts; the lesson goes into the playbook
   (UFO² likewise relies on the applications' help).
3. **Explore by observing** when there is no recipe: observe, act, check, within
   a step limit (path 3).
4. **Scale up**: after a failure, the turn moves to the strong model. The gains from code
   as action grow with model capability, and on OSWorld the gap between a small
   model (24 to 47%) and the best ones (about 85%) exceeds anything the structure adds.
5. **Vision as a fallback** for apps without accessibility (62% of UFO²'s failures
   come from undetected controls), phase F.
6. **Ask or say no**: a short question if the request is ambiguous, and an honest
   refusal rather than a random action.
7. **Learn from success**: the new task becomes a skill and lessons, so
   the next one of the same kind takes the short path (OSExpert: about 80% of the
   efficiency gap with a human closed by learned skills).

### 3.12 Learning an app on its own (phase 17)

User request (October 14): whatever had to be hand-coded for Spotify, MIKATA
must be able to learn by itself, for any app. Supporting work:

- **UFO²** (Microsoft, 2025, arXiv 2504.14603): on Windows, the agent goes first through the
  app's native interfaces, with the screen as a fallback, and works without getting in the user's way.
- **WALT** (Salesforce, ICLR 2026, arXiv 2510.01524): a site's features (search,
  filter...) are reverse-engineered into reusable tools (`search(requête)`), replacing
  click sequences with parameterized URLs; a demonstrate, generate, validate loop.
- **AppAgentX** (2025, arXiv 2503.02268): repeated action sequences become
  high-level shortcuts drawn from the agent's history.
- **ASI** (CMU, COLM 2025, arXiv 2504.06821): a successful, verified program becomes a
  skill; +23.5 points of success on WebArena, 10 to 15% fewer steps.

What MIKATA was given, none of it specific to one app:

1. **App links** (`app_links`, `open_app_link`): the protocols registered with
   Windows by installed apps (51 links for 30 apps on the user's PC),
   never a system component nor a risky protocol (`ms-msdt`, `search-ms`...);
   opened in the background: the app is minimized as soon as it comes to the front and the user's
   window comes back (verified: Spotify stays controllable while minimized); the view is ready
   before it is read (the link's text appears).
2. **List items** (`uia_items`): each row with what it displays and its
   buttons; **clicking within a row** (`uia_click` with `item`); **waiting** for a text
   (`uia_wait`).
3. **Learning**: the sequence of calls that succeeded becomes a skill. The writer
   and the reviser receive the reference sheet of the program functions (the shape of their results); a
   value chosen after reading the screen stays masked, unless the user said it;
   a skill that copies `...`, or in which a `pick` does not mention what it is looking for, is
   refused; buttons are chosen by asking a question (their order changes).

Measurements. Simulated PC, real model, the same kind of request five times with different
titles: discovery in 4 calls (6 to 8 s), skill learned from the 1st turn, called
on turns 2 to 4 (2 calls, 2.4 to 3.8 s), active after 3 successes and replayed without the model
on the 5th. Real Spotify (dry-run click): the right row targeted, the user's window
stayed in front, the skill reused for another title. Bench: 56/56 then 55/56;
new tasks on other apps (a setting through `ms-settings:`, a Discord channel). The
`spotify_play` tool from phase 16g is removed: MIKATA learns what it used to do.
Phase 17d: an app that is minimized or hidden behind others can be read, but Spotify
ignores "Play" there; for the duration of the click, the minimized window is shown without being activated,
then minimized again. Toggles report their state (a muted mic is no longer muted a second time).

Phase 17e: **a real background mode for Chromium apps**. Spotify, Discord, VS Code...
are web pages running in Chromium (Electron, CEF). Launched with `--remote-debugging-port`,
they can be driven like the working browser, with the window hidden (SW_HIDE). Supporting work:

- **Chrome DevTools Protocol**, `Accessibility` domain: `getFullAXTree` returns the
  accessibility tree computed by Chromium, the very one it exposes to UIA (same roles and
  names); each node carries its `backendDOMNodeId`, which `DOM.resolveNode` and then
  `Runtime.callFunctionOn` act on. This is the route taken by Puppeteer (`page.accessibility`) and
  by Playwright for Electron (`_electron.launch`, which launches with the debugging port).
- **UFO²** (above): go through the app's interfaces rather than the screen, without
  getting in the user's way.
- **Security**: since Chrome 111, a WebSocket connection carrying an Origin header is
  refused without `--remote-allow-origins`; MIKATA sends none and never sets that
  flag: a web page cannot reach the port. A local program, however, can:
  hence the explicit `uia.devtools_apps` list (empty by default), and the port picked at random and
  bound to 127.0.0.1.

This is not a per-app driver: `DevtoolsBackend` is a third engine behind the same
`uia_*` tools (after UIA and AX), chosen window by window; all of
`UiController`'s logic (list items, clicking within a row, "près de" ("near")) and the learned
skills stay the same. Measured on the real Spotify (sound muted), with the real
`UiController`: hidden relaunch 4.8 s (once), search and Enter 0.6 s, results
read in 1.6 s, click on the right row's "Play" 0.2 s, playback confirmed (audio session
active), no Spotify window visible (sampled every 10 ms); the tree is read in
~0.1 s versus several seconds with UIA. Two findings led to two general gestures:
an app is only ready when its page stops changing (two identical summaries 0.5 s apart:
at +0.5 s Spotify had text but not its playback bar); typing into a search
field often only opens suggestions, and the search starts with Enter
(`uia_set_text` with `submit`). An app link always brings Spotify to the front (verified):
an app kept hidden is driven through its interface, never through a link.

## 4. What goes, what stays

| Item | Decision | Reason |
| --- | --- | --- |
| `mikata/agent2/` (23 files, 3,483 lines), its 20 test files, `tests/agent2_eval/`, `scripts/eval_agent2.py`, the `agent2` config section, the background pass | **Deleted** | Section 1 |
| `mikata/skills/` (497 lines) and `data/skills/skills.json` | **Replaced** by the packages of section 3.6 | Duplicates, degraded procedures learned, replay rarely used |
| `mikata/agent/loop.py` (884 lines, 1,483 as of October 14) | **Rewritten** as modules (phase 16a: `turn`, `model`, `actions`, `skill_path`, `learning`, `history`, `narration`; `loop.py` assembles a turn, 443 lines); validated behaviors kept (follow-up, barge-in, reasoning keywords, summary) | One class mixed seven responsibilities |
| `mikata/llm/prompts.py` | **Rewritten** | New contract (text, direct, script) |
| `mikata/agent/guard.py` | **Kept**, placed behind the policies | Phase 5 criterion |
| `mikata/llm/client.py`, `usage.py`, `agent/memory.py` | **Kept** | Sound; the two model roles added |
| `mikata/tools/` (56 tools, 3,968 lines) | **Kept**, enriched (effect, structured result, duration) | Tested implementation |
| `mikata/browser/`, `uia/`, `files/` (3,041 lines) | **To be audited** in phase A | Partly written during the fragile period; not reviewed yet |
| Audio, wake word, STT, TTS, interface | **Unchanged** | Out of scope |

Deleted files are removed with `git rm` on a branch; the history keeps them.

## 5. Phases and criteria

One phase per session, on its own branch, validated on the PC before the next one.

| Phase | Content | Acceptance criterion |
| --- | --- | --- |
| **A. Foundation** | Removal of v2, green CI, audit of `browser/` `uia/` `files/`, benchmark, measurement of the current loop | Green CI on both OSes; 30 real tasks that can be replayed; baseline figures published in the README |
| **B. Action space** | Effects and structured results on the tools, generated SDK reference, filtered observations | Every tool has a declared effect; no existing test broken; context tokens measured before and after |
| **C. Interpreter and policies** | Python subset, budget, provenance, pause; quarantined extraction; suite of booby-trapped pages | Infinite loop stopped in under 1 s; zero consequential actions on the injection suite; phase 5 tests green |
| **D. Single loop** | Text, direct, script, bounded exploration; two model roles; background tasks and acknowledgments | Section 3.8 budgets met on the bench; the direct path does not get slower |
| **E. Learning** | Skills (packages, verified induction, local replay, isolated critic, lifecycle) and lesson playbook | A request close to a learned task goes through local replay in under 1 s; a broken skill is demoted; the bench does not regress after learning |
| **F. Opening up** | MCP client, `SKILL.md` import and export, validated new primitives, vision as a fallback | An external tool usable from a script; an imported skill stays inactive until reviewed |

Proposed targets, to be confirmed once the phase A baseline is known: the budgets of
section 3.8, at most three LLM calls at the median on multi-step tasks, and 80%
success over three trials.

## 5 bis. Progress as of October 14, 2026

| Phase | Status | Measurement |
| --- | --- | --- |
| A. Foundation | done (`phase-14a-socle`) | Green CI; 50-task bench; baseline 47/49 on one trial |
| B. Action space | done (`phase-14b-action`) | 56 tools with a declared effect and trust level |
| C. Interpreter and policies | done (`phase-14c-interpreteur`) | infinite loop stopped in under 1 s; injection suite green |
| D. Single loop | partly done (`phase-14d-boucle`) | 48/49 over three trials (47/49 without programs); long chains 2.8 s versus 3.7 s |
| E. Learning | done (`phase-14e-apprentissage`, `phase-15e-competences`) | lesson playbook and traces; one lesson takes a task from 2 calls to 1; a skill learned on the 1st turn, called by the model on turns 2 to 4, is replayed on the 5th without an LLM call (real model, simulated PC) |
| F. Opening up | partly done (`phase-16f-ouverture`) | the real model picks an MCP tool on its own (the official `mcp-server-time` server) and answers correctly in 4 to 5.5 s; an imported skill stays proposed until activation. Still to do: validated new primitives and vision as a fallback |

Done for memory (phase 16e, §3.7): preferences learned from the user's corrections (`learn/preferences.py`), episodes drawn from traces for a request that refers to the past (`learn/episodes.py`).

Done in D (phase 16a): loop split into components, system prompt rewritten as
principles and canonical examples (knowledge specific to one tool moved into that tool's
description); two-trial bench: 98/100 and 164 model calls, versus 99/100 and 169.
Done in D (phase 16c): "tu en es où ?" ("how far along are you?") answered without the model, from the turns in
progress; an acknowledgment when an action starts and its expected duration (declared by
each tool) exceeds `agent.ack_after_s`, never before, nor for a tool that announces
itself; the model no longer says what it is about to do, only what it found. Requests
already ran in parallel and "annule" ("cancel") stopped the last one in progress.
Done in D (phase 16d): bounded exploration (`agent/explore.py`, `explore` tool):
observe, ask the model for up to three actions with a context reduced to the goal and the
state, execute under the policies, at most 8 steps and 30 s. Phase D is complete.
Done in D (phase 16g, user feedback): an exploration whose actions
complete the goal (`"fait"`, "done", returned with the action) stops without re-reading or an extra step;
a failed re-read (window being redrawn) is retried once. A
correction ("non, sur Spotify", "no, on Spotify") redoes the request all the way through: a step that only
undoes (tool property `undoes`) does not conclude the turn, nor does the turn end until the correction
has made as many constructive actions as the corrected request. Preferences
come only from the user's own words (their request and their correction), even after a
web page. The bench replays corrections (`history`: the exchanges that come before).
A step that accomplishes something concludes the correction; only a step that prepares (property
`prepares`: opening an app, bringing a window to the front) continues. The
`spotify_play` tool from this phase is replaced in phase 17 by learned generic gestures
(§3.12).
Done in E (phase 15e): skills (`mikata/learn/skills.py`), which replace
`mikata/skills/`. Deviations from section 3.6: verification before adoption is done in
code, without replaying the request (replaying an action on the PC would perform it twice); the
first use supervised by the model serves as the trial, and three are needed before
local replay (decision of October 14). Revisited in phase 17f, at the user's request ("une
tâche apprise doit être quasi instantanée", "a learned task must be near-instant") and based on their traces from October 14: the model called
the skill on trial without checking anything further (it only sees what the skill returns), for
2 extra seconds every time, and its failures (two versions of "couper le micro", "mute the mic") happened
the same way with or without it. So while on trial, the skill is replayed without the model as soon as
the router recognizes it; only matching by its pattern still waits for the active state ("lance {titre} de
{artiste}" would catch "lance la vidéo de Squeezie" ("play the Squeezie video"), which Jev classifies as "aucune" ("none") at 0.97).
The control chosen by `pick()` is remembered after a success (0.5 s and one wrong choice
fewer on Discord, where Jev hesitated between 200 controls). When two skills share
the request (Jev: Spotify 0.77, YouTube 0.21), the router abstains and the
model chooses: the small LLM call was deciding at random, without knowing the user's
habits. Done in E (phase 16b): revision by an isolated critic when a skill fails and
the request then succeeds another way right after; the new version is verified in code,
the old one archived (`versions/`), the last three failure cases kept (`failures.json`,
rather than a folder: nothing gets deleted). Not done yet: `recover.md`, importing
skills. The audit planned in phase A is done for `browser/` and `files/` (branch
`phase-14f-audit`: two fixes, and a known limitation of the index's elevated helper);
`uia/` and the USN journal decoding remain to be reviewed.

## 6. Decisions made (October 13, 2026)

1. **In-house interpreter**, with an execution budget and provenance. Monty remains a possible
   backup engine.
2. **Adjustable autonomy**: an "Accès complet" ("full access") option in the settings
   (`guard.full_access`). When off, MIKATA asks for a "oui" ("yes") before a consequential action
   when the turn has read untrusted content; when on, it acts without asking. In both
   cases, destructive actions remain forbidden and a shell command built from
   untrusted content is refused.
3. **Two model roles**, `llm.fast` and `llm.strong`.
4. **`mikata/tools/` remains the implementation**, enriched with a declared effect, a
   trust level for its output and a structured result.

## 7. Limitations

- Generality comes first from the model: with a fast model without reasoning, new
  tasks will stay more fragile than with a frontier model.
- The targets in section 5 are proposals, not measurements.
- The reading of the papers relies on their abstracts and main sections; the
  figures cited are the authors' own, on their benchmarks, not on MIKATA.
- Vision (apps without accessibility) is deferred to phase F.

## 8. References

- CodeAct, *Executable Code Actions Elicit Better LLM Agents*: <https://arxiv.org/pdf/2402.01030>
- CoAct-1: <https://arxiv.org/html/2508.03923v3>
- OSWorld-Human: <https://arxiv.org/abs/2506.16042>
- *Efficient GUI Agents: A Systems Survey* (Sept. 2026): <https://arxiv.org/pdf/2609.02309>
- UFO², *The Desktop AgentOS*: <https://arxiv.org/html/2504.14603>
- ASI, *Inducing Programmatic Skills for Agentic Tasks*: <https://arxiv.org/html/2504.06821v2>
- Agent Workflow Memory: <https://proceedings.mlr.press/v267/wang25bx.html>
- EvoSkill-GUI, *Reflect, Revise, Reuse* (Sept. 2026): <https://arxiv.org/html/2609.17653>
- CaMeL, *Defeating Prompt Injections by Design*: <https://arxiv.org/html/2503.18813>
- *Design Patterns for Securing LLM Agents against Prompt Injections*: <https://arxiv.org/pdf/2506.08837>
- ACE, *Agentic Context Engineering*: <https://arxiv.org/html/2510.04618v2>
- Voyager: <https://arxiv.org/abs/2305.16291>
- Alita: <https://arxiv.org/abs/2505.20286>; Alita-G: <https://arxiv.org/html/2510.23601v1>
- Darwin Gödel Machine: <https://arxiv.org/html/2505.22954v3>
- *Your Agent May Misevolve*: <https://arxiv.org/html/2509.26354>
- SEAgent: <https://arxiv.org/html/2508.04700v1>; OSExpert: <https://arxiv.org/abs/2603.07978>
- *Speculative Actions*: <https://arxiv.org/pdf/2510.04371>
- Talker-Reasoner: <https://arxiv.org/pdf/2410.08328>
- Full-Duplex-Bench-v3: <https://arxiv.org/pdf/2604.04847>
- Anthropic, *Building effective agents*: <https://www.anthropic.com/engineering/building-effective-agents>
- Anthropic, *Advanced tool use*: <https://www.anthropic.com/engineering/advanced-tool-use>
- Anthropic, *Writing effective tools for agents*: <https://www.anthropic.com/engineering/writing-tools-for-agents>
- Anthropic, *Effective context engineering for AI agents*: <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
- Anthropic, *Demystifying evals for AI agents*: <https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents>
- Pydantic Monty: <https://pydantic.dev/articles/pydantic-monty>
- Agent Skills, open standard: <https://agentskills.io>
- OpenClaw security (malicious skills): <https://www.bitdoze.com/openclaw-security-guide/>
- Agent memory in 2026, claimed scores: <https://mem0.ai/blog/state-of-ai-agent-memory-2026>;
  framework comparison: <https://atlan.com/know/best-ai-agent-memory-frameworks-2026/>
