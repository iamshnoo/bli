# Evaluation data lineage

`evaluation_data_lineage.csv` is the normalized inventory of terms used for
alignment and evaluation. It contains four record types:

- `concept`: each of the 1,000 English cultural concepts and its eight available
  translations;
- `axis_endpoint`: both endpoints of all 50 semantic axes, with translation
  availability recorded for all nine languages;
- `alignment_anchor`: the 3,000 English words used to fit representation maps;
- `negative_control`: the 100 concrete English control words.

Axis endpoint 1 is coded `-1` and endpoint 2 is coded `+1`. These values only
fix the direction of a signed comparison; they do not label an endpoint as
negative or positive in a normative sense.

The model comparisons use English concept and axis terms in every model space.
The translations support translation-quality stratification; they are not
substituted into the model prompts. Thirty-five unique axis endpoint terms fall
outside the 1,000-concept translation inventory, so their non-English rows are
marked `not_translated` rather than filled with unverified translations.

For translated terms, the file preserves the NLLB language code, back
translation, string similarity, COMETKiwi score and tier, manual-review flag,
and duplicate-translation flag. Axis rows also include the citation keys,
titles, years, and URLs that motivated each axis. Concept rows give the sources
for their broader category framework.

Rebuild and validate the file from the canonical probe JSON and translation
CSVs with:

```bash
python src/probes/build_eval_data_lineage.py
```
