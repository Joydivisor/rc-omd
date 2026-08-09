import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const pptxPath = path.join(here, "RC_OMD_Workshop_5min.pptx");
const rendered = path.join(here, "rendered");

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(pptxPath));
if (presentation.slides.items.length !== 5) {
  throw new Error(`Expected 5 slides, found ${presentation.slides.items.length}`);
}

await fs.mkdir(rendered, { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  await writeBlob(
    path.join(rendered, `roundtrip-${String(index + 1).padStart(2, "0")}.png`),
    await presentation.export({ slide, format: "png", scale: 1 }),
  );
}

const expectedClaims = [
  "0.193", "0.967", "1.176", "1 / 3 PASS",
  "NO neural-network or LLM effectiveness claim",
];
for (const claim of expectedClaims) {
  const result = await presentation.inspect({
    kind: "slide,textbox,shape,table",
    search: claim,
    maxChars: 5000,
  });
  if (!result.ndjson.includes(claim)) {
    throw new Error(`Missing expected visible claim: ${claim}`);
  }
}

const tableResult = await presentation.inspect({
  kind: "table",
  search: "OOD scenario",
  maxChars: 5000,
});
const tableRecord = tableResult.ndjson
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line))
  .find((record) => record.kind === "table");
if (!tableRecord) throw new Error("Could not resolve the OOD result table");
const table = presentation.resolve(tableRecord.id);
const tableValues = [];
for (let row = 0; row < 5; row += 1) {
  for (let column = 0; column < 5; column += 1) {
    tableValues.push(String(table.getCell(row, column).value));
  }
}
for (const value of [
  "0.992182", "0.990865", "0.980026", "0.980782",
  "0.992195", "0.990567", "0.989446", "0.987688",
  "0.248", "0.239", "0.189", "0.199",
]) {
  if (!tableValues.includes(value)) throw new Error(`Missing OOD table value: ${value}`);
}

const notes = await presentation.inspect({
  kind: "notes",
  maxChars: 20000,
});
const noteMatches = (notes.ndjson.match(/\[Sources\]/g) || []).length;
if (noteMatches < 5) {
  throw new Error(`Expected source blocks on 5 slides, found ${noteMatches}`);
}

for (let index = 1; index <= 5; index += 1) {
  const file = path.join(rendered, `slide-${String(index).padStart(2, "0")}.layout.json`);
  const layout = JSON.parse(await fs.readFile(file, "utf8"));
  for (const element of layout.elements || []) {
    if (!Array.isArray(element.bbox)) continue;
    const [left, top, width, height] = element.bbox;
    const epsilon = 0.5;
    if (left < -epsilon || top < -epsilon || left + width > 1280 + epsilon || top + height > 720 + epsilon) {
      throw new Error(`Out-of-bounds element on slide ${index}: ${JSON.stringify(element.bbox)}`);
    }
  }
}

console.log("Workshop QA passed: 5 slides, expected claims present, source notes present, no layout bbox overflow.");
