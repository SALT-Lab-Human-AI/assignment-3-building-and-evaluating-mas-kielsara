A3 - Building and evaluating a Multi-Agent System

1) Overview
You will design, implement, and evaluate a multi‑agent system for deep research on an HCI‑relevant topic. Your system should:

Orchestrate agents using AutoGen or LangGraph (your choice, or both). (Preferably use Groq API for model inference, but feel free to use your own models/APIs as well)

Provide a simple user interface (CLI or web) for interactive querying.

Include safety guardrails to handle unsafe/inappropriate requests (e.g., Guardrails, NeMo Guardrails, or equivalent policy filtering).

Evaluate generated outputs using LLM‑as‑a‑Judge scoring and report findings.

The deliverables are a technical report (3-4  page single-column and single-space write‑up) and a GitHub Classroom repo containing your code and documentation.



2) System Requirements
A. Agents & Orchestration (AutoGen or LangGraph)
Minimum 3 agents with clear roles (e.g., Planner, Retriever/Researcher, Critic/Verifier, optional Safety/Sanitizer, Writer).

Recommended workflow: task planning → evidence gathering → synthesis → critique/revision → final answer.

Include tool use (e.g., web search/S2 API, citation extraction, PDF parsing) where appropriate. You may mock tools if needed, but explain trade‑offs.

Some helpful web search services:
https://www.tavily.com/Links to an external site. (You can get a student free quota)
https://brave.com/search/api/Links to an external site.
For paper search: https://www.semanticscholar.org/product/apiLinks to an external site.
B. Safety & Guardrails
Integrate at least one safety framework, e.g., GuardrailsLinks to an external site., NeMo Guardrails, or a custom policy filter.

System must detect and handle unsafe inputs and potential unsafe outputs (refuse, route to safe alternative, or redact).

Detail your guardrail policy or documented list of prohibited categories and response strategies in your write-up

Log safety events (what was blocked/redacted and why), this should be communicated in both logs and user interfaces (see below)

C. User Interface
Provide a minimal but usable interface:

CLI with clear prompts or a web UI (e.g., Streamlit, Gradio, minimal React/Flask/FastAPI).

Display agents' output traces

Show citations/evidence collected by the system.

Indicate when a response was refused or sanitized due to safety policies.

D. Evaluation: LLM‑as‑a‑Judge
Define task prompts and ground‑truth/expectation criteria for your topic.

Some topics you can consider:
Literature reconnaissance on an HCI concept (e.g., explainable AI for novices, ethical AI in education, AR usability).

Comparative review of design patterns from research papers, blogs, and docs.

Synthesis of best practices for UI components from mixed sources (docs, repos, articles).

Trend analysis and critique of a recent HCI subarea (e.g., agentic UX, AI‑driven prototyping)

Use at least 2 independent judging prompts (e.g., different rubrics or perspectives) to score system outputs on. Below are some examples you can use, but you are encouraged to create your own metrics:

Relevance & coverage of the query

Evidence use & citation quality

Factual accuracy/consistency

Safety compliance (no unsafe content)

Clarity & organization

Report a comprehensive evaluation of your multi-agent system in your write-up


3) Report write-up structure
Technical Report 3-4 pages, single-column and single-space

Abstract (~150 words, summarize what you did)

System Design and implementation (agents, tools, control flow, models)

Safety Design (policies, guardrails)

Evaluation Setup and results (datasets/queries, judge prompts, metrics, error analysis)

Discussion & Limitations (Summarize your insights and learnings, what are the limitations of this work, and future work)

References (APA style, NOT counted towards page count)





4) Grading Rubric (100 pts)

### System Architecture & Orchestration (20 pts)
- **Agents (10 pts)**: Minimum 3 agents with distinct roles; must include planner and researcher; agents must coordinate
- **Workflow (5 pts)**: Clear and well-designed multi-agent workflow
- **Tools (3 pts)**: Web/paper search tools (or other tools) integration
- **Error Handling (2 pts)**: Graceful handling of API failures and invalid inputs

### User Interface & UX (15 pts)
- **Functionality (6 pts)**: Working CLI or web interface that accepts queries and displays results
- **Transparency (6 pts)**: Display agent traces, citations/sources, and which agent is active
- **Safety Communication (3 pts)**: Show when content is refused/sanitized

### Safety & Guardrails (15 pts)
- **Implementation (5 pts)**: Integrated safety framework with both input and output guardrails
- **Policies (5 pts)**: Documented safety policies (≥3 categories) integrated into code
- **Behavior & Logging (5 pts)**: System refuses/sanitizes unsafe content and logs events with context

### Evaluation (LLM-as-a-Judge) (20 pts)
- **Implementation (6 pts)**: Working judge with ≥2 independent evaluation prompts
- **Design (6 pts)**: ≥3 measurable metrics with clear scoring scales
- **Analysis (8 pts)**: Report evaluation results with interpretation and error analysis. Use more than 5 diverse test queries.

### Reproducibility & Engineering Quality (10 pts)
Complete README with explanation on how to reproduce the results reported in the write-up.

### Report Quality (20 pts)
- **Structure (8 pts)**: 3-4 pages; all required sections; ~150-word abstract; APA references
- **Content (12 pts)**: Clear system design (4 pts), evaluation results (4 pts), discussion of limitations/insights/ethics (4 pts)


**Bonus (up to +10 pts)**
Notable innovation (e.g., toolformer‑style augmentation, novel guardrail design, or human eval triangulation).
