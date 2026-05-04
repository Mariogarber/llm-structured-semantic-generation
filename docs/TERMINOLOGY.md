# Project Terminology

This glossary defines the terms used in the Kubernetes v1 documentation,
processed dataset fields, structural-target files, and dataset-analysis plots.
Definitions are based on the current implementation, especially
`utils/kubernetes_dataset_preprocessor.py`,
`src/llm_structured_semantic_generation/structure.py`, and
`scripts/analyze_kubernetes_dataset.py`.

## Core Dataset Terms

| Term | Meaning |
| --- | --- |
| `sample` | One original Kubernetes task identified by a `sample_id` such as `q2`. A sample has one target YAML file and two prompt variants. |
| `prompt_variant` | The prompt source used for a row. Current values are `question` and `question_simplified`. Both variants for the same sample must stay in the same split. |
| `target_yaml_raw` | The original YAML text read from the source `labeled_code.yaml` file. |
| `target_yaml_normalized` | The canonical YAML produced after parsing and re-rendering the raw target with stable indentation and sorted keys. This is the main target used downstream. |
| `YAML document` | One document returned by `yaml.safe_load_all`. Multi-document YAML separated by `---` is treated as one sample with multiple documents. |
| `resource document` | A YAML document interpreted as one Kubernetes resource for analysis purposes. Most samples have one resource document, but multi-document samples may have more. |

## Structural Target Terms

| Term | Meaning |
| --- | --- |
| `block` | One pre-parser representation unit corresponding to one YAML physical line after removing leading indentation. It stores the line content plus explicit structural metadata. |
| `line_text` | The text of a block without leading indentation. Example: `name: game-demo`, not `  name: game-demo`. |
| `level` | The indentation level assigned to a block. In v1, one level equals two leading spaces in normalized YAML. A top-level line has `level = 0`; a line indented by two spaces has `level = 1`. |
| `line_index` | Zero-based position of the block inside its YAML document. |
| `document_index` | Zero-based index of the YAML document that contains the block. It is used to reconstruct `---` document separators. |
| `block_count` | Number of blocks generated for a sample or prompt row. It is the length of the `blocks` list. |
| `max_block_level` | Largest `level` value observed among the blocks of a sample. This measures line indentation depth, not parsed YAML tree depth. |

Important distinction: `level` and `yaml_max_depth` are related but not the
same metric. `level` comes from indentation in rendered YAML lines. `yaml_max_depth`
comes from the parsed YAML object tree.

## YAML Complexity Metrics

These fields are computed from the parsed YAML object, not from raw text length.
They describe dataset complexity and are used in descriptive plots.

| Term | Meaning |
| --- | --- |
| `yaml_max_depth` | Maximum recursive depth of the parsed YAML documents in a sample. Scalars have depth `1`; empty mappings/lists have depth `1`; non-empty mappings/lists have depth `1 + max(child depths)`. For multi-document YAML, the sample value is the maximum over documents. |
| `yaml_mapping_nodes` | Number of mapping objects, equivalent to parsed YAML dictionaries. Mapping keys themselves are not counted as separate scalar nodes. |
| `yaml_list_nodes` | Number of list objects in the parsed YAML. |
| `yaml_scalar_nodes` | Number of scalar values in the parsed YAML, including strings, numbers, booleans, and null-like values. |
| `yaml_total_nodes` | Sum of `yaml_mapping_nodes`, `yaml_list_nodes`, and `yaml_scalar_nodes`. For multi-document YAML, counts are summed across documents. |
| `nodes` | In analysis plots, this means parsed YAML tree nodes. It does not mean Kubernetes `Node` resources or cluster machines unless explicitly written as Kubernetes nodes. |
| `target_yaml_char_count` | Number of characters in the raw target YAML text. This is a text-size metric, not a structural metric. |

## Kubernetes Analysis Terms

| Term | Meaning |
| --- | --- |
| `kind` | The Kubernetes `kind` field of a parsed resource document, such as `ConfigMap`, `Deployment`, or `Service`. |
| `primary_kind` | The first available `kind` among mapping documents in a sample. It is used for sample-level grouping in plots. Multi-document samples can contain additional kinds, which are represented in resource-level tables. |
| `apiVersion` | The Kubernetes `apiVersion` field of a resource document. |
| `resource_count` | Number of parsed YAML documents in a sample. It is not a count of Kubernetes objects found by schema validation. |
| `semantic fields` | A fixed list of recursively searched key names, such as `metadata`, `spec`, `containers`, `image`, and `ports`. These are approximate coverage signals, not full semantic validation. |

## Split And Leakage Terms

| Term | Meaning |
| --- | --- |
| `split` | Dataset partition assigned at leakage-group level. Current values are `train`, `validation`, and `test`. |
| `leakage_group` | Group identifier used to keep exact or near duplicate samples in the same split. |
| `leakage_reasons` | Explanation for why a sample belongs to its leakage group, for example `sample_id_only` or `exact_yaml_duplicate`. |
| `duplicate_yaml_group_size` | Number of samples sharing the same normalized YAML target. |
| `duplicate_prompt_group_size` | Number of samples sharing the same normalized prompt text, when such duplicates exist. |
| `near_duplicate_group_size` | Number of samples grouped by the near-prompt-duplicate heuristic, when such duplicates exist. |
| `prompt_pair_similarity` | Sequence similarity between the original and simplified prompt text for the same sample after prompt normalization. |

## Evaluation Terms

| Term | Meaning |
| --- | --- |
| `yaml_parse_ok` | Whether the YAML text can be parsed by the YAML parser. |
| `block_parse_ok` | Whether a predicted block sequence satisfies the parser-side block contract before YAML reconstruction. |
| `parsed_equal_to_reference` | Whether parsed predicted YAML equals parsed reference YAML. This compares parsed structures, not raw text. |
| `content_exact_match_rate` | Fraction of reference block `line_text` values that exactly match the prediction at the same position. |
| `level_exact_match_rate` | Fraction of reference block `level` values that exactly match the prediction at the same position. |

