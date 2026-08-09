import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, "rendered");

const W = 1280;
const H = 720;
const C = {
  ink: "#0B1020",
  navy: "#16324F",
  blue: "#2F6FED",
  teal: "#20BFA9",
  tealDark: "#087F72",
  amber: "#F0B35A",
  red: "#E05A63",
  paper: "#F6F8FB",
  white: "#FFFFFF",
  slate: "#5C677D",
  muted: "#8B95A8",
  line: "#D9E0EA",
  softBlue: "#EAF0FF",
  softTeal: "#E5F7F3",
  softAmber: "#FFF3DF",
  softRed: "#FDECEE",
};

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opts.size ?? 24,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    typeface: opts.typeface ?? "Aptos",
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    insets: opts.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function addBox(slide, x, y, w, h, fill = C.white, line = C.line, radius = "rounded-xl") {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function addLine(slide, x1, y1, x2, y2, color = C.line, width = 2, dash = "solid") {
  return slide.shapes.add({
    geometry: "line",
    position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 },
    fill: "none",
    line: { style: dash, fill: color, width },
  });
}

function addPill(slide, text, x, y, w, fill, color) {
  addBox(slide, x, y, w, 30, fill, fill, "rounded-full");
  addText(slide, text, x + 8, y + 4, w - 16, 22, {
    size: 13, bold: true, color, align: "center", valign: "middle",
  });
}

function addHeader(slide, number, title, kicker) {
  addText(slide, `0${number}`, 62, 38, 42, 26, { size: 14, bold: true, color: C.tealDark });
  addText(slide, kicker.toUpperCase(), 112, 38, 330, 25, { size: 13, bold: true, color: C.muted });
  addText(slide, title, 62, 78, 1156, 62, { size: 36, bold: true, color: C.ink });
  addLine(slide, 62, 151, 1218, 151, C.line, 1);
}

function addFooter(slide, text = "TEAM PRISM  |  RC-OMD WORKSHOP") {
  addText(slide, text, 62, 684, 650, 18, { size: 10, bold: true, color: C.muted });
}

function setNotes(slide, script, sources) {
  const block = `${script}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]`;
  slide.speakerNotes.textFrame.setText(block);
  slide.speakerNotes.setVisible(true);
}

function addMember(slide, x, y, w, name, role, accent) {
  addBox(slide, x, y, w, 84, C.white, C.line);
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: 6, height: 84 },
    fill: accent,
    line: { style: "solid", fill: accent, width: 0 },
  });
  addText(slide, name, x + 18, y + 14, w - 30, 24, { size: 15, bold: true, color: C.navy });
  addText(slide, role, x + 18, y + 44, w - 30, 24, { size: 13, color: C.slate });
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT, { recursive: true });
  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  // Slide 1 — minimal title + member chips, adapted from Codex Grid slide-01.
  {
    const s = deck.slides.add();
    s.background.fill = C.paper;
    addPill(s, "REINFORCEMENT LEARNING & ROBOTIC AUTOMATION", 62, 50, 380, C.softBlue, C.blue);
    addText(s, "Reliability-Calibrated\nOnline Mirror Descent", 62, 116, 850, 150, {
      size: 54, bold: true, color: C.ink,
    });
    addText(s, "What survives sparse rewards — and what breaks under shared parameters", 66, 284, 1050, 48, {
      size: 24, color: C.slate,
    });
    addText(s, "TEAM PRISM  ·  RL-GROUP 1", 66, 352, 500, 28, { size: 17, bold: true, color: C.tealDark });

    const names = [
      ["HUANG YUXUAN", "Project lead & final report", C.blue],
      ["LIU YIPU", "Theory", C.teal],
      ["XIE HAOXIANG", "Environments & estimators", C.amber],
      ["ZHONG YIFAN", "Algorithms & infrastructure", C.red],
      ["LI YUFEI", "Experiments & analysis", C.navy],
    ];
    const startX = 62;
    const gap = 12;
    const cardW = (1156 - gap * 4) / 5;
    names.forEach(([name, role, accent], i) => addMember(s, startX + i * (cardW + gap), 466, cardW, name, role, accent));
    addText(s, "A five-person optimization study with cross-review of evidence and claims.", 62, 582, 900, 30, {
      size: 16, color: C.slate,
    });
    addFooter(s);
    setNotes(s,
      "Good morning. We are Team PRISM from RL Group 1. Our project asks a narrow optimization question: when a sparse terminal reward gives us a noisy step-level credit estimate, can its reliability control how far Online Mirror Descent moves? Huang coordinates the project, Liu leads theory, Xie builds environments and estimators, Zhong develops algorithms and infrastructure, and Li leads experiments and analysis. Today I will show one preregistered positive result, followed by a deliberately retained negative transfer result.",
      ["E:/RC-OMD/outputs/Team_PRISM_Project_Proposal.tex", "E:/RC-OMD/paper/main.tex"]
    );
  }

  // Slide 2 — three-column problem framing, adapted from Codex Grid slide-07.
  {
    const s = deck.slides.add();
    s.background.fill = C.paper;
    addHeader(s, 2, "One terminal bit must supervise a long trajectory", "Problem");
    const cards = [
      {
        x: 62, label: "SPARSE SIGNAL", big: "R(τ) ∈ {0,1}",
        body: "The verifier scores the completed trajectory, not each intermediate decision.",
        fill: C.softBlue, color: C.blue,
      },
      {
        x: 448, label: "GROUP BASELINE", big: "Aᵢ = rᵢ − r̄", 
        body: "Group-relative updates compare sampled outcomes without training a separate critic.",
        fill: C.softTeal, color: C.tealDark,
      },
      {
        x: 834, label: "BROADCAST", big: "Aᵢ → every step",
        body: "The same sequence-level signal can move pivotal and routine decisions together.",
        fill: C.softAmber, color: "#9B640C",
      },
    ];
    cards.forEach((c) => {
      addBox(s, c.x, 190, 324, 326, C.white, C.line);
      addPill(s, c.label, c.x + 24, 216, 150, c.fill, c.color);
      addText(s, c.big, c.x + 24, 276, 276, 54, { size: 30, bold: true, color: C.ink, align: "center" });
      addLine(s, c.x + 24, 350, c.x + 300, 350, C.line, 1);
      addText(s, c.body, c.x + 24, 378, 276, 102, { size: 18, color: C.slate, align: "center", valign: "middle" });
    });
    addText(s, "Our boundary: this is an associative credit-allocation study — not a causal credit proof.", 62, 554, 1156, 50, {
      size: 20, bold: true, color: C.navy, align: "center", valign: "middle",
    });
    addFooter(s);
    setNotes(s,
      "The setting is sparse verifiable reward. A complete trajectory receives one binary score. Group-relative methods form an advantage by subtracting the group mean, which avoids a learned critic. But that one trajectory-level advantage is then broadcast through the whole sequence. This creates an allocation problem: routine and pivotal decisions can move together, and all-correct or all-wrong groups can contain no relative signal. Our project does not claim to discover causal responsibility. It studies whether an online reliability signal can control the geometry of the update under this limited feedback.",
      ["https://arxiv.org/abs/2402.03300", "https://arxiv.org/abs/2604.11056", "https://arxiv.org/abs/2604.09459", "E:/RC-OMD/paper/main.tex"]
    );
  }

  // Slide 3 — large framework statement + 3 callouts, adapted from Codex Grid slide-09.
  {
    const s = deck.slides.add();
    s.background.fill = C.paper;
    addHeader(s, 3, "RC-OMD separates direction from permitted movement", "Method");
    addBox(s, 62, 186, 1156, 184, C.navy, C.navy);
    addText(s, "credit estimate ĉₜ,ₖ", 92, 220, 270, 40, { size: 25, bold: true, color: C.white, align: "center" });
    addText(s, "chooses WHERE", 92, 272, 270, 30, { size: 16, bold: true, color: "#BFD1E3", align: "center" });
    addText(s, "+", 378, 246, 46, 52, { size: 40, bold: true, color: C.teal, align: "center" });
    addText(s, "reliability qₜ,ₖ", 438, 220, 250, 40, { size: 25, bold: true, color: C.white, align: "center" });
    addText(s, "chooses HOW FAR", 438, 272, 250, 30, { size: 16, bold: true, color: "#BFD1E3", align: "center" });
    addText(s, "⇒", 714, 242, 56, 52, { size: 40, bold: true, color: C.amber, align: "center" });
    addText(s, "ηₜ,ₖ = η₀ f(qₜ,ₖ)", 796, 220, 348, 40, { size: 29, bold: true, color: C.white, align: "center" });
    addText(s, "local KL / step-size budget", 796, 272, 348, 30, { size: 16, bold: true, color: "#BFD1E3", align: "center" });

    const callouts = [
      [62, "1  ONLINE RELIABILITY", "A persistent standardized score updates after each grouped rollout."],
      [458, "2  LOCAL GEOMETRY", "Reliable evidence permits larger movement; weak evidence stays conservative."],
      [854, "3  EVALUATION", "Known distractor positions are used only to measure wasted KL in controlled tasks."],
    ];
    callouts.forEach(([x, heading, body], i) => {
      addBox(s, x, 410, 364, 172, C.white, C.line);
      addText(s, heading, x + 22, 432, 320, 26, { size: 15, bold: true, color: i === 1 ? C.tealDark : C.blue });
      addText(s, body, x + 22, 478, 320, 74, { size: 17, color: C.slate, valign: "middle" });
    });
    addText(s, "Tabular, factorized policy only in the positive study.", 62, 616, 700, 30, { size: 16, bold: true, color: C.red });
    addFooter(s);
    setNotes(s,
      "RC-OMD separates two questions. The estimated credit chooses where the policy should move. A reliability score chooses how far that local decision is allowed to move through a step-specific learning rate, or equivalently a local KL budget. In our online variant, reliability is a persistent standardized score updated from grouped rollouts. The controlled environments know which positions are pivotal or distracting, but those labels are used only for evaluation. The algorithm receives the terminal reward and its online reliability statistics. Therefore our positive claim is about movement allocation in a tabular factorized policy, not causal credit recovery.",
      ["https://arxiv.org/abs/2005.09814", "E:/RC-OMD/algorithms", "E:/RC-OMD/credit_estimators", "E:/RC-OMD/paper/main.tex"]
    );
  }

  // Slide 4 — preregistered evidence table.
  {
    const s = deck.slides.add();
    s.background.fill = C.paper;
    addHeader(s, 4, "Pre-registered OOD result: 4 of 4 scenarios pass", "Positive evidence");
    addPill(s, "GO", 1086, 85, 100, C.softTeal, C.tealDark);
    const table = s.tables.add({
      rows: 5,
      columns: 5,
      left: 62,
      top: 188,
      width: 1156,
      height: 286,
      columnWidths: [420, 182, 182, 182, 190],
      values: [
        ["OOD scenario (n = 10 seeds)", "Uniform AUC", "Online AUC", "AUC Δ", "Distractor KL ratio"],
        ["Dense 2-of-6 · tiny group", "0.992182", "0.990865", "−0.001317", "0.248"],
        ["Needle 5-of-5 · long horizon", "0.980026", "0.980782", "+0.000755", "0.239"],
        ["Threshold 3-of-5 · small group", "0.992195", "0.990567", "−0.001628", "0.189"],
        ["Threshold 4-of-6 · 3 actions", "0.989446", "0.987688", "−0.001758", "0.199"],
      ],
    });
    table.styleOptions = { headerRow: true, bandedRows: true };
    table.borders.assign({ style: "solid", fill: C.line, width: 1 });
    table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 5 }).assign({
      fill: C.navy,
      textStyle: { bold: true, color: C.white, fontSize: 14, alignment: "center" },
      margins: { top: 8, right: 8, bottom: 8, left: 8 },
    });
    table.cells.block({ row: 1, column: 0, rowCount: 4, columnCount: 5 }).assign({
      textStyle: { color: C.ink, fontSize: 14, alignment: "center" },
      margins: { top: 8, right: 8, bottom: 8, left: 8 },
    });
    table.cells.block({ row: 1, column: 0, rowCount: 4, columnCount: 1 }).assign({
      textStyle: { color: C.navy, fontSize: 14, bold: true, alignment: "left" },
    });
    table.cells.block({ row: 1, column: 4, rowCount: 4, columnCount: 1 }).assign({
      fill: C.softTeal,
      textStyle: { color: C.tealDark, fontSize: 15, bold: true, alignment: "center" },
    });
    addBox(s, 62, 508, 355, 112, C.softBlue, C.softBlue);
    addText(s, "≤ 0.0018", 84, 526, 311, 38, { size: 29, bold: true, color: C.blue, align: "center" });
    addText(s, "absolute AUC difference", 84, 571, 311, 24, { size: 14, bold: true, color: C.slate, align: "center" });
    addBox(s, 440, 508, 355, 112, C.softTeal, C.softTeal);
    addText(s, "18.9%–24.8%", 462, 526, 311, 38, { size: 29, bold: true, color: C.tealDark, align: "center" });
    addText(s, "of Uniform distractor KL", 462, 571, 311, 24, { size: 14, bold: true, color: C.slate, align: "center" });
    addBox(s, 818, 508, 400, 112, C.softAmber, C.softAmber);
    addText(s, "6.9%–10.7%", 840, 526, 356, 38, { size: 29, bold: true, color: "#9B640C", align: "center" });
    addText(s, "runtime overhead · all < 1.5×", 840, 571, 356, 24, { size: 14, bold: true, color: C.slate, align: "center" });
    addFooter(s, "PRE-REGISTERED PROTOCOL  ood-v1-2026-08-08  |  EXECUTION d37a5ff");
    setNotes(s,
      "Before running the OOD study, we froze the tasks, hyperparameters and pass rule. Online RC-OMD had to lose no more than 0.01 normalized success AUC, use at most half the distractor KL of Uniform, and pass at least three of four scenarios. It passed all four. The absolute AUC differences were at most 0.0018, while distractor KL fell to 18.9 to 24.8 percent of Uniform. Runtime overhead was 6.9 to 10.7 percent. This is a Pareto result with different preregistered base steps, not a claim that reliability scaling always improves reward at the same nominal step size.",
      ["E:/RC-OMD/docs/MILESTONE_5_OOD_RESULTS.md", "E:/RC-OMD/results/ood_preregistered/protocol_evaluation.json", "E:/RC-OMD/configs/ood_preregistered.json"]
    );
  }

  // Slide 5 — NO-GO with horizontal ratio bars + professor questions, adapted from Codex Grid slide-20.
  {
    const s = deck.slides.add();
    s.background.fill = C.paper;
    addHeader(s, 5, "Shared parameters break the local-control mechanism", "NO-GO & discussion");
    addPill(s, "1 / 3 PASS", 1036, 85, 150, C.softRed, C.red);

    addText(s, "Distractor-KL ratio  (Online / Uniform)", 62, 184, 560, 28, { size: 17, bold: true, color: C.navy });
    const barX = 174;
    const barW = 430;
    const max = 1.25;
    const rows = [
      ["Separable", 0.193, C.teal],
      ["Partial alias", 0.967, C.amber],
      ["Complete alias", 1.176, C.red],
    ];
    const y0 = 250;
    rows.forEach(([label, value, color], i) => {
      const y = y0 + i * 98;
      addText(s, label, 62, y + 11, 102, 28, { size: 15, bold: true, color: C.slate, align: "right" });
      addBox(s, barX, y, barW, 48, "#E8EDF4", "#E8EDF4", "rounded-full");
      addBox(s, barX, y, Math.max(12, barW * value / max), 48, color, color, "rounded-full");
      addText(s, value.toFixed(3), barX + 12, y + 11, barW - 24, 28, { size: 17, bold: true, color: value > 0.75 ? C.ink : C.white, align: "right" });
    });
    const thresholdX = barX + barW * 0.75 / max;
    addLine(s, thresholdX, 225, thresholdX, 502, C.navy, 2, "dashed");
    addText(s, "frozen pass limit 0.75", thresholdX - 84, 208, 168, 18, { size: 11, bold: true, color: C.navy, align: "center" });
    addText(s, "AUC conditions passed in all 3; failure came entirely from coupled distractor KL.", 62, 530, 560, 54, {
      size: 18, bold: true, color: C.red, align: "center", valign: "middle",
    });

    addText(s, "Questions for the professor", 682, 184, 536, 28, { size: 18, bold: true, color: C.navy });
    const qs = [
      ["01", "Geometry", "Should reliability enter a Fisher / natural-gradient metric?"],
      ["02", "Projection", "Can we fit pivotal OMD targets while constraining low-reliability KL?"],
      ["03", "Next test", "Which nonlinear benchmark should precede any neural or LLM claim?"],
    ];
    qs.forEach(([n, head, body], i) => {
      const y = 230 + i * 122;
      addBox(s, 682, y, 536, 98, C.white, C.line);
      addText(s, n, 704, y + 22, 42, 34, { size: 20, bold: true, color: i === 2 ? C.red : C.tealDark, align: "center" });
      addText(s, head.toUpperCase(), 764, y + 14, 420, 22, { size: 13, bold: true, color: C.muted });
      addText(s, body, 764, y + 42, 420, 42, { size: 17, bold: true, color: C.ink });
    });
    addBox(s, 682, 606, 536, 46, C.softRed, C.softRed);
    addText(s, "NO neural-network or LLM effectiveness claim.", 700, 617, 500, 24, { size: 16, bold: true, color: C.red, align: "center" });
    addFooter(s, "PRE-REGISTERED PROTOCOL  function-approx-v1-2026-08-09  |  EXECUTION 2c91c69");
    setNotes(s,
      "Our first shared-parameter transfer is a NO-GO. With separable linear features, the distractor-KL ratio was 0.193 and the scenario passed. Under partial aliasing it rose to 0.967, and under complete aliasing to 1.176. All three AUC conditions still passed; the failure came entirely from the frozen KL criterion. Diagnostics show that reliability still separates pivotal from distractor positions, but projection couples their parameter updates. Therefore local step-size control is insufficient when the policy cannot realize independent local trust regions. We do not claim neural or LLM effectiveness. We would value guidance on parameter-space geometry, constrained projection, and the next nonlinear benchmark.",
      ["E:/RC-OMD/docs/MILESTONE_6_FUNCTION_APPROX_RESULTS.md", "E:/RC-OMD/results/function_approx_preregistered/protocol_evaluation.json", "E:/RC-OMD/configs/function_approx_preregistered.json"]
    );
  }

  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(OUT, "RC_OMD_Workshop_montage.webp"), montage);
  for (const [i, slide] of deck.slides.items.entries()) {
    await writeBlob(path.join(OUT, `slide-${String(i + 1).padStart(2, "0")}.png`), await deck.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(OUT, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text(), "utf8");
  }
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(path.join(HERE, "RC_OMD_Workshop_5min.pptx"));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
