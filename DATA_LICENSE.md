# Data Licensing and Attribution

This repository contains source code (licensed under the MIT License (see
[LICENSE](LICENSE)) and data artifacts derived from third-party wiki content.
The data artifacts are governed by the licenses of their upstream sources,
which differ between the two domains covered by this project.

## Data artifacts in this repository

The following artifacts are derivative works of third-party wiki content:

- Chunked JSONL files for Dota 2 and League of Legends content
- The ChromaDB vector index (if redistributed)
- BM25 indices (if redistributed)
- The 323-question evaluation benchmark, where questions reference in-game
  entities, mechanics, items, abilities, or other content sourced from the
  wikis below

These artifacts are redistributed under the licenses of their respective
upstream sources, as detailed below.

## Source attributions

### Dota 2 content

Dota 2 content is derived from the Dota 2 Wiki on Fandom
(<https://dota2.fandom.com>), which is licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported
(CC BY-NC-SA 3.0)](https://creativecommons.org/licenses/by-nc-sa/3.0/).

All Dota 2-derived artifacts in this repository are redistributed under
**CC BY-NC-SA 3.0**. This means:

- **Attribution**: You must give appropriate credit to the Dota 2 Wiki
  contributors and indicate if changes were made.
- **NonCommercial**: You may not use these artifacts for commercial
  purposes.
- **ShareAlike**: If you remix, transform, or build upon these artifacts,
  you must distribute your contributions under the same license.

### League of Legends content

League of Legends content is derived from the League of Legends Wiki on
Fandom (<https://lol.fandom.com>), which is licensed under
[Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).

All League of Legends-derived artifacts in this repository are redistributed
under **CC BY-SA 3.0**. This means:

- **Attribution**: You must give appropriate credit to the League of
  Legends Wiki contributors and indicate if changes were made.
- **ShareAlike**: If you remix, transform, or build upon these artifacts,
  you must distribute your contributions under the same license.

CC BY-SA 3.0 does not impose the non-commercial restriction; only the
Dota 2-derived artifacts carry that restriction.

## Practical consequences

Because the two source wikis use different licenses, downstream users should
treat the combined data artifacts as carrying the **stricter of the two
licenses (CC BY-NC-SA 3.0)** whenever Dota 2-derived content is used or
redistributed alongside League of Legends-derived content. Users who wish to
use only the League of Legends artifacts are free to do so under CC BY-SA 3.0.

## Trademarks and game ownership

Dota 2 is a trademark of Valve Corporation. League of Legends is a trademark
of Riot Games, Inc. This project is not affiliated with, endorsed by, or
sponsored by Valve Corporation or Riot Games, Inc. All game names, character
names, item names, ability names, and related terminology are the property
of their respective owners and are used here for the purposes of academic
research and non-commercial reproducibility.

## Model weights

This repository does **not** redistribute model weights. The following models
must be obtained from their respective upstream sources under their own
licenses:

- **Llama 3.1 8B (Q4_K_M GGUF)** — Obtainable via Meta or community GGUF
  conversions. Subject to the
  [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/),
  which includes an acceptable use policy and naming requirements for
  derived models.
- **Qwen3-Embedding-4B (Q8 GGUF)** - Apache 2.0.
- **bge-reranker-large** (BAAI) - MIT.
- **deberta-v3-large-zeroshot-v2.0** (MoritzLaurer) - MIT.

Users are responsible for complying with each model's license when
downloading and using these weights.

## Summary table

| Component                              | License             | Source                                |
|----------------------------------------|---------------------|---------------------------------------|
| Source code (this repository)          | MIT                 | See [LICENSE](LICENSE)                |
| Dota 2-derived data artifacts          | CC BY-NC-SA 3.0     | <https://dota2.fandom.com>            |
| League of Legends-derived data artifacts | CC BY-SA 3.0      | <https://lol.fandom.com>              |
| Model weights                          | Various (see above) | Not redistributed                     |