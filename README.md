# generic-comment-analyzer

A regulation-agnostic pipeline for analyzing public comments on U.S. federal
regulations (regulations.gov). It classifies each comment's position and concerns,
identifies who's commenting, detects form-letter campaigns, and produces an
interactive HTML report — plus an optional "Read the Rule" page that lets you read
the proposed rule section-by-section and see the comments citing each section.

Everything regulation-specific lives in one config file per regulation
(`analyzer_config.yaml`) — the code is generic.

## How it works

1. **Load** a regulations.gov bulk-export CSV (+ extract attachment text: PDF/DOCX/TXT, optional OpenAI image OCR).
2. **Deduplicate** identical comment text.
3. **Analyze** each unique comment with an LLM (OpenAI via LiteLLM) into a validated,
   config-defined Pydantic schema (positions/concerns, entity type, key quote, etc.).
4. **Verify** ambiguous stance/entity classifications with a second LLM pass.
5. **Detect campaigns** (MinHash near-duplicate clustering) and derive each campaign's stance.
6. **Report** to a self-contained `index.html` dashboard, and a `read-the-rule.html` page.

Runs are **resumable** (text-keyed checkpoint + parquet snapshots) so a restart never
loses work.

## The report

- Grouped stat cards (Overview / Position / Topics) with consistent, config-driven stance
  colors (Oppose / Support / Unclear).
- Concern, CFR-section, and campaign breakdowns as stacked Oppose/Support bars.
- An interactive comments table with a "+ Add filter" chip system and **shareable filter URLs**,
  CSV export, and detail modals.
- Campaigns labeled by their actual text; a per-section citation view.
- **Read the Rule**: the proposed rule's text by section, each colored by the stance of the
  comments citing it, with a link into the filtered analysis.

## Config drives everything

`analyzer_config.yaml` is the single source of truth. Key blocks:

- **`fields:`** — the analysis schema; each field (name + type + options) declared once drives
  the Pydantic model, the LLM prompt, and the frontend (columns/filters/cards). `source: regex`
  fields (e.g. CFR-section extraction) are computed at report time, no LLM.
- **`regex_flags:`** — boolean topic flags → stat cards + filters.
- **`second_pass:`** — verification model, triggers, and prompts.
- **`campaigns:`** — form-letter detection settings, including the two opt-in latent-semantic
  refinements described under [Campaign similarity](#campaign-similarity-optional-lsa-refinements).
- **`report:`** — `colors:` (the full palette — edit any color in one place) and display toggles.
- **`rule_text:`** — Federal Register document + CFR part for the Read-the-Rule page.

## Layout

```
regulations/<slug>/
  analyzer_config.yaml       # config: fields, flags, prompts, colors, rule_text   (committed)
  regulation_metadata.json   # name, docket id, agency                             (committed)
  source.csv                 # regulations.gov bulk export                         (local only)
  full_run.parquet           # analysis output                                     (local only)
  rule_sections.json         # parsed proposed-rule text                           (local only)
  index.html / read-the-rule.html   # the report pages                            (local only)
```

Only the two config files per regulation are version-controlled; large data stays local.

## Setup

```bash
python -m venv myenv && source myenv/bin/activate
uv pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

## Run

```bash
# analyze
python pipeline.py --regulation <slug> --workers 24

# (optional) fetch the proposed-rule text for the Read-the-Rule page
python fetch_rule_text.py --regulation <slug>

# (optional) test your model on just a few comments and compile a full report
python pipeline.py --regulation <slug> --sample 10
python generate_report.py --parquet regulations/<slug>/sample_10.parquet --output regulations/<slug>/index-10.html
```

The report is a large single-file HTML (~120 MB at ~47k comments), so deploy it locally or to a static
host like Netlify rather than GitHub Pages.

## Campaign similarity (optional LSA refinements)

Campaigns are found with **MinHash** on word 5-grams: fast, deterministic, and precise about
"is this the same form letter, word for word". That stays the default and decides who is *in* a
campaign. Two things it handles less well, which is what the optional latent-semantic (LSA:
TF-IDF followed by a truncated SVD) tools in `lsa_similarity.py` are for:

- A comment can land in more than one raw cluster. By default it goes to whichever cluster is
  processed last, and a campaign's stored `campaign_size` counts members that ended up elsewhere.
- One letter often shows up as several campaigns, because variants differ word for word (a
  different opening, a personalised paragraph).

Both options below are **off by default**. With them off, a comment gets no new keys and
`campaign_id`, `campaign_size` and the family fields come out exactly as before. They are set in the
`campaigns:` block of `analyzer_config.yaml`:

```yaml
campaigns:
  resolve_overlaps: true   # settle multi-cluster comments; store the true campaign_size
  letters: true            # add letter_id / letter_label: campaigns that are the same letter
  letters_threshold: 0.9   # cosine of campaign centres above which campaigns merge (default 0.9)
  lsa_k: 200               # SVD dimensions (default 200)
```

- **`resolve_overlaps`** gives a multi-cluster comment to the cluster whose canonical text it
  matches best word for word. LSA similarity only breaks near-ties; on its own it was no more
  reliable than the default rule at picking the better match. `campaign_size` becomes the number
  of comments the campaign actually holds. `campaign_id` is unchanged.
- **`letters`** groups campaigns whose LSA centres are close (average linkage, so a chain of
  similar campaigns does not merge its unrelated ends) and tags every member with `letter_id` (the
  id of the group's largest campaign) and `letter_label`. It never changes campaign or family
  membership. The report does not use these fields yet; they are columns in `full_run.parquet`.

If a corpus is too small or too uniform to fit a subspace, the pipeline logs a warning and falls
back to the default behaviour instead of failing.

### Campaign map

`campaign_map.py` draws one map of every campaign in a regulation, reading the same
`full_run.parquet` as the report (it only reads `campaign_id` / `family_id`):

```bash
python campaign_map.py --regulation <slug>
python campaign_map.py --regulation <slug> --k 200 --layout pca --formats png,svg
```

- Each campaign is a shaded disc (orange = opposes, blue = supports, by majority stance), with
  radius growing with the cube root of its members so small campaigns stay visible next to huge ones.
- Each *distinct* text in a campaign is a dot with area proportional to its exact copies; the
  farther a dot is from its disc's centre, the more that text deviates from the campaign's template.
- Distinct texts that are not in a campaign are small grey dots. Nearby means similar wording.
  Letter families are labelled.

It writes `campaign_map.png` (and `.svg`/`.pdf` if asked) and `campaign_map_campaigns.csv` (one row
per campaign: size, stance, how tight it is, and the terms that most define it) into the regulation
folder, or `--out-dir`. Like the other outputs, they reflect the comment data, so keep them local.

The 2-D layout is UMAP, an optional dependency; `--layout pca` needs nothing extra:

```bash
pip install -r requirements-optional.txt
```

Only campaign centres and non-campaign texts go through the layout. Members of a form letter are
nearly identical, and UMAP inflates tiny differences into fake clusters, so members fan out around
their campaign instead.

### Reading LSA similarity safely

- A short or idiosyncratic comment can keep very little of its text in the subspace. Its direction
  is then mostly noise, and scaling it to unit length makes a confident-looking cosine out of
  nothing. `lsa_similarity.retained_norm` reports how much was kept; gate on it.
- Too small a `k` produces near-1 cosines between campaigns that share no wording at all. Raise `k`
  until they disappear (the default of 200 is a starting point, not a rule).
- The subspace is fit on distinct texts weighted by the square root of their copies, so one huge
  campaign neither owns every axis nor drops out of the subspace.
