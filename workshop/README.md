# RC-OMD Workshop Package

Generate the editable deck from the repository root:

```powershell
C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe workshop\RC_OMD_Workshop_5min.mjs
```

The generated PowerPoint is `RC_OMD_Workshop_5min.pptx`. Speaker notes contain the five-minute script and per-slide source blocks. The PDF is exported from the generated PowerPoint during the release workflow.

Literature preparation is split into two layers:

- `RC_OMD_Literature_Matrix.md`: the ten-item one-page positioning matrix.
- `RC_OMD_Extended_Literature_Inventory.md`: the 22-paper inventory, claim-to-citation map, reading priorities, and Harvard bibliography.

Run the round-trip content and layout check with:

```powershell
C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe workshop\verify_workshop.mjs
```
