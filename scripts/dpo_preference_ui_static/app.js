const state = {
  units: [],
  currentUnit: null,
  currentAuditCase: null,
  auditCases: [],
  currentSuggestions: [],
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatMetric(value) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

async function loadAll() {
  const summary = await api("/api/summary");
  const gateRate = summary.kubernetes_domain_gate_pass_rate === null || summary.kubernetes_domain_gate_pass_rate === undefined
    ? "n/a"
    : `${(summary.kubernetes_domain_gate_pass_rate * 100).toFixed(1)}%`;
  $("summary").textContent = `${summary.unit_count} prompts, ${summary.candidate_count} candidates, KDV gate pass ${gateRate}, ${summary.approved_pair_count} approved pairs`;
  await loadGuide();
  await loadUnits();
  await loadAuditCases();
}

async function loadGuide() {
  const guide = await api("/api/labeling-guide");
  $("guideContent").innerHTML = `
    <div class="guide-summary">
      <strong>${escapeHtml(guide.title)}</strong>
      <span>${escapeHtml(guide.version)}</span>
    </div>
    ${guideSection("Objective", guide.objective)}
    ${guideSection("Decision Rules", guide.decision_rules)}
    ${guideSection("Review Order", guide.review_order)}
    ${guideSection("Pair Types", guide.pair_types)}
    ${guideSection("Prompt F1 Rules", guide.prompt_f1_rules)}
    ${guideSection("Tie / Skip", guide.tie_skip_rules)}
    ${guideSection("Metric Flags", guide.recommended_metric_flags)}
    <p class="final-criterion">${escapeHtml(guide.final_criterion)}</p>
  `;
}

function guideSection(title, items) {
  return `
    <section class="guide-section">
      <h4>${escapeHtml(title)}</h4>
      <ul>
        ${(items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </section>
  `;
}

async function loadUnits() {
  const status = $("statusFilter").value;
  const query = status ? `?limit=500&status=${encodeURIComponent(status)}` : "?limit=500";
  const payload = await api(`/api/units${query}`);
  state.units = payload.units;
  renderUnitList();
}

function renderUnitList() {
  $("unitList").innerHTML = state.units.map((unit) => `
    <button class="unit-item ${state.currentUnit?.unit_id === unit.unit_id ? "active" : ""}" data-unit="${escapeHtml(unit.unit_id)}">
      <strong>${escapeHtml(unit.sample_id)} / ${escapeHtml(unit.prompt_variant)}</strong>
      <span>${unit.candidate_count} candidates - score ${formatMetric(unit.best_score)} - pairs ${unit.approved_pair_count || 0} - reqs ${unit.prompt_requirement_count}</span>
      <span class="status ${escapeHtml(unit.status)}">${escapeHtml(unit.status)}</span>
    </button>
  `).join("");
  document.querySelectorAll(".unit-item").forEach((button) => {
    button.addEventListener("click", () => selectUnit(button.dataset.unit));
  });
}

async function selectUnit(unitId) {
  const unit = await api(`/api/units/${encodeURIComponent(unitId)}`);
  state.currentUnit = unit;
  $("emptyState").classList.add("hidden");
  $("unitPanel").classList.remove("hidden");
  $("unitTitle").textContent = `${unit.sample_id} / ${unit.prompt_variant}`;
  $("unitMeta").textContent = `${unit.split} - ${unit.candidates.length} candidates`;
  $("promptText").textContent = unit.prompt_text || "";
  $("referenceYaml").textContent = unit.reference_yaml || "";
  $("promptRequirements").textContent = JSON.stringify(unit.prompt_requirements || [], null, 2);
  hydrateDecisionForm(unit.latest_decision);
  renderCandidates(unit);
  renderSuggestions(unit.agent_suggestions || []);
  renderUnitList();
}

function hydrateDecisionForm(decision) {
  $("decision").value = decision?.decision || "preference";
  $("confidence").value = decision?.confidence || "medium";
  $("metricFlags").value = (decision?.metric_flags || []).join(", ");
  $("rationale").value = decision?.rationale || "";
}

function renderCandidates(unit) {
  $("candidateList").innerHTML = unit.candidates.map((candidate) => {
    const metrics = candidate.metrics || {};
    return `
      <article class="candidate-card">
        <div class="candidate-head">
          <div>
            <h4>${escapeHtml(candidate.candidate_id)} - ${escapeHtml(candidate.source_run_id)}</h4>
            <span class="status ${candidate.hard_invalid ? "skip" : "preference"}">
              score ${formatMetric(candidate.preference_score)} ${candidate.hard_invalid ? "hard invalid" : ""}
            </span>
          </div>
          <div class="candidate-actions">
            <label><input type="radio" name="chosen" value="${escapeHtml(candidate.candidate_key)}"> chosen</label>
            <label><input type="radio" name="rejected" value="${escapeHtml(candidate.candidate_key)}"> rejected</label>
          </div>
        </div>
        <div class="metric-grid">
          ${metricCell("YAML", metrics.yaml_parse_ok)}
          ${metricCell("Blocks", metrics.block_parse_ok)}
          ${metricCell("Prompt F1", metrics.prompt_requirement_f1)}
          ${metricCell("KDV", metrics.kubernetes_domain_validity_score)}
          ${metricCell("Gate pass", metrics.kubernetes_domain_gate_pass)}
          ${metricCell("Req fields", metrics.required_field_complete_resource_rate)}
          ${metricCell("Level", metrics.level_exact_match_rate)}
          ${metricCell("Line F1", metrics.line_text_f1)}
          ${metricCell("Lines", `${formatMetric(metrics.line_count_prediction)} / ${formatMetric(metrics.line_count_reference)}`)}
        </div>
        <details>
          <summary>Model output</summary>
          <pre>${escapeHtml(candidate.model_output_text || "")}</pre>
        </details>
        <details open>
          <summary>Reconstructed YAML</summary>
          <pre>${escapeHtml(candidate.reconstructed_yaml || "")}</pre>
        </details>
        <details>
          <summary>Full evaluation</summary>
          <pre>${escapeHtml(JSON.stringify(candidate.evaluation || {}, null, 2))}</pre>
        </details>
      </article>
    `;
  }).join("");
}

function metricCell(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(formatMetric(value))}</strong></div>`;
}

async function saveDecision(event) {
  event.preventDefault();
  if (!state.currentUnit) return;
  const payload = decisionPayload();
  payload.unit_id = state.currentUnit.unit_id;
  $("saveStatus").textContent = "Saving...";
  try {
    await api("/api/preferences", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    $("saveStatus").textContent = "Saved";
    await loadUnits();
    await selectUnit(state.currentUnit.unit_id);
  } catch (error) {
    $("saveStatus").textContent = error.message;
  }
}

function decisionPayload() {
  const chosen = document.querySelector('input[name="chosen"]:checked')?.value || null;
  const rejected = document.querySelector('input[name="rejected"]:checked')?.value || null;
  return {
    decision: $("decision").value,
    chosen_candidate_key: chosen,
    rejected_candidate_key: rejected,
    confidence: $("confidence").value,
    rationale: $("rationale").value,
    metric_flags: $("metricFlags").value.split(",").map((item) => item.trim()).filter(Boolean),
  };
}

async function copyDecisionPacket() {
  if (!state.currentUnit) return;
  const packet = await api(`/api/units/${encodeURIComponent(state.currentUnit.unit_id)}/decision-packet`);
  await navigator.clipboard.writeText(JSON.stringify(packet, null, 2));
  $("saveStatus").textContent = "Decision packet copied";
}

async function importAgentSuggestion() {
  if (!state.currentUnit) return;
  $("agentStatus").textContent = "Importing...";
  try {
    const suggestion = JSON.parse($("agentSuggestion").value);
    suggestion.unit_id = state.currentUnit.unit_id;
    suggestion.review_status = "pending";
    await api("/api/agent-suggestions", {
      method: "POST",
      body: JSON.stringify(suggestion),
    });
    $("agentStatus").textContent = "Imported";
    await selectUnit(state.currentUnit.unit_id);
  } catch (error) {
    $("agentStatus").textContent = error.message;
  }
}

function renderSuggestions(suggestions) {
  state.currentSuggestions = suggestions;
  if (!suggestions.length) {
    $("suggestions").innerHTML = "";
    return;
  }
  $("suggestions").innerHTML = suggestions.map((suggestion, index) => `
    <details class="suggestion">
      <summary>${escapeHtml(suggestion.review_status)} - ${escapeHtml(suggestion.confidence)} - ${escapeHtml(suggestion.pair_type || "pair")} - ${escapeHtml(suggestion.agent_policy_version || suggestion.annotator || "agent")}</summary>
      <div class="suggestion-actions">
        <button type="button" data-suggestion-load="${index}">Load into form</button>
        <button type="button" data-suggestion-approve="${index}">Approve as human</button>
      </div>
      <pre>${escapeHtml(JSON.stringify(suggestion, null, 2))}</pre>
    </details>
  `).join("");
  document.querySelectorAll("[data-suggestion-load]").forEach((button) => {
    button.addEventListener("click", () => loadSuggestion(Number(button.dataset.suggestionLoad)));
  });
  document.querySelectorAll("[data-suggestion-approve]").forEach((button) => {
    button.addEventListener("click", () => approveSuggestion(Number(button.dataset.suggestionApprove)));
  });
}

function loadSuggestion(index) {
  const suggestion = state.currentSuggestions[index];
  if (!suggestion) return;
  $("decision").value = suggestion.decision || "preference";
  $("confidence").value = suggestion.confidence || "medium";
  $("metricFlags").value = (suggestion.metric_flags || []).join(", ");
  $("rationale").value = suggestion.rationale || "";
  setRadioValue("chosen", suggestion.chosen_candidate_key);
  setRadioValue("rejected", suggestion.rejected_candidate_key);
  $("saveStatus").textContent = "Suggestion loaded";
}

async function approveSuggestion(index) {
  const suggestion = state.currentSuggestions[index];
  if (!suggestion || !state.currentUnit) return;
  const payload = {
    unit_id: state.currentUnit.unit_id,
    decision: suggestion.decision,
    chosen_candidate_key: suggestion.chosen_candidate_key,
    rejected_candidate_key: suggestion.rejected_candidate_key,
    confidence: suggestion.confidence || "medium",
    rationale: suggestion.rationale || "",
    metric_flags: suggestion.metric_flags || [],
    pair_type: suggestion.pair_type,
    score_margin: suggestion.score_margin,
    agent_policy_version: suggestion.agent_policy_version,
    source_annotation_id: suggestion.annotation_id,
  };
  $("agentStatus").textContent = "Approving...";
  try {
    await api("/api/preferences", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    $("agentStatus").textContent = "Approved as human";
    await loadUnits();
    await selectUnit(state.currentUnit.unit_id);
  } catch (error) {
    $("agentStatus").textContent = error.message;
  }
}

function setRadioValue(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
    input.checked = input.value === value;
  });
}

async function exportFinal() {
  const payload = await api("/api/export-final", { method: "POST", body: "{}" });
  $("summary").textContent = `Exported ${payload.count} final pairs to ${payload.path}`;
}

async function loadAuditCases() {
  const payload = await api("/api/audit-cases");
  state.auditCases = payload.cases;
  renderAuditList();
}

function renderAuditList() {
  $("auditList").innerHTML = state.auditCases.slice(0, 80).map((row) => `
    <button class="audit-item ${state.currentAuditCase?.audit_case_id === row.audit_case_id ? "active" : ""}" data-case="${escapeHtml(row.audit_case_id)}">
      <strong>${escapeHtml(row.sample_id)} / ${escapeHtml(row.prompt_variant)}</strong>
      <span>${escapeHtml(row.audit_bucket)} - ${escapeHtml(row.status)}</span>
    </button>
  `).join("");
  document.querySelectorAll(".audit-item").forEach((button) => {
    button.addEventListener("click", () => selectAuditCase(button.dataset.case));
  });
}

function selectAuditCase(caseId) {
  const row = state.auditCases.find((item) => item.audit_case_id === caseId);
  if (!row) return;
  state.currentAuditCase = row;
  $("auditEditor").classList.remove("hidden");
  $("auditTitle").textContent = `${row.sample_id} / ${row.prompt_variant}`;
  $("auditPrompt").textContent = row.prompt_text || "";
  const gold = row.gold_requirements?.length ? row.gold_requirements : row.extracted_prompt_requirements || [];
  $("goldRequirements").value = JSON.stringify(gold, null, 2);
  $("goldNotes").value = row.notes || "";
  renderAuditList();
}

async function saveGoldCase() {
  if (!state.currentAuditCase) return;
  const payload = {
    ...state.currentAuditCase,
    status: "reviewed",
    gold_requirements: JSON.parse($("goldRequirements").value || "[]"),
    notes: $("goldNotes").value,
  };
  const response = await api("/api/audit-gold", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  $("auditReport").textContent = JSON.stringify(response.report.overall, null, 2);
  await loadAuditCases();
}

async function showAuditReport() {
  const report = await api("/api/audit-report");
  $("auditEditor").classList.remove("hidden");
  $("auditReport").textContent = JSON.stringify(report, null, 2);
}

$("refreshBtn").addEventListener("click", loadAll);
$("exportBtn").addEventListener("click", exportFinal);
$("statusFilter").addEventListener("change", loadUnits);
$("decisionForm").addEventListener("submit", saveDecision);
$("packetBtn").addEventListener("click", copyDecisionPacket);
$("importAgentBtn").addEventListener("click", importAgentSuggestion);
$("saveGoldBtn").addEventListener("click", saveGoldCase);
$("auditReportBtn").addEventListener("click", showAuditReport);

loadAll().catch((error) => {
  $("summary").textContent = error.message;
});
