# KG Build Report — `kg_v1`

- **Provider**: deepinfra
- **Pipeline stage**: `done`
- **Runs in this folder**: 8 (last `run_008`)
- **Generated**: 2026-08-06T17:29:16.442384+00:00

## 1. Execution overview

- Agent calls: **29047** (29028 ok, 3 parse-failed, 16 errored, 284 re-asks, 76 truncated)
- Tokens: **74,597,251** (60,135,612 in / 14,461,639 out)
- Estimated cost: **$8.2382** — 16 paid calls reported no cost figure
- Request time: 156h 25m 9s · wall clock 3h 54m 59s
- Graph: 54730 nodes, 123645 edges

### Runs

| Run | Status | Started (UTC) | Duration | Stages | Provider | Calls | Tokens | Cost | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run_001 | completed | 2026-08-06T08:33:26 | 5m 1s | oe | deepinfra | 19 | 101125 | $0.0173 | 72d60ad2 |
| run_002 | completed | 2026-08-06T08:49:36 | 2h 38m 2s | ke | deepinfra | 15738 | 61732668 | $6.8336 | 72d60ad2 |
| run_003 | failed | 2026-08-06T13:38:17 | 4m 48s | kf | deepinfra | 2119 | 1790518 | $0.1928 | 72d60ad2 |
| run_004 | completed | 2026-08-06T14:07:13 | 14m 32s | kf | deepinfra | 5444 | 4794528 | $0.5161 | 72d60ad2 |
| run_005 | completed | 2026-08-06T15:24:12 | 15m 56s | ke | deepinfra | 90 | 434602 | $0.0479 | 72d60ad2 |
| run_006 | completed | 2026-08-06T15:50:10 | 13m 14s | ke | deepinfra | 53 | 502534 | $0.0631 | 72d60ad2 |
| run_007 | completed | 2026-08-06T16:54:01 | 10m 37s | ke | deepinfra | 43 | 360846 | $0.0421 | 72d60ad2 |
| run_008 | completed | 2026-08-06T17:15:18 | 12m 46s | kf | deepinfra | 5541 | 4880430 | $0.5252 | 72d60ad2 |

### Stages

| Stage | Calls | OK | Parse fail | Errors | Re-asks | Truncated | In tokens | Out tokens | Cost | Request time | Wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OE | 19 | 19 | 0 | 0 | 0 | 0 | 86250 | 14875 | $0.0173 | 25m 58s | 5m 1s |
| KE | 15924 | 15905 | 3 | 16 | 271 | 76 | 50280986 | 12749664 | $6.9868 | 134h 30m 12s | 3h 17m 38s |
| KF | 13104 | 13104 | 0 | 0 | 13 | 0 | 9768376 | 1697100 | $1.2342 | 21h 28m 58s | 32m 1s |

### Agents

| Agent | Model | Prompt | Calls | Failed | Re-asks | Avg in | Avg out | Avg time | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ke_builder | Qwen/Qwen3.5-9B | v6 | 8037 | 17 | 43 | 3190 | 1072 | 29.3s | $3.8561 |
| ke_critic | google/gemma-4-26B-A4B-it | v4 | 7887 | 2 | 228 | 3125 | 524 | 31.6s | $3.1306 |
| kf_builder | Qwen/Qwen3.5-9B | v2 | 11718 | 0 | 13 | 758 | 132 | 5.7s | $1.1193 |
| kf_critic | google/gemma-4-26B-A4B-it | v2 | 1386 | 0 | 0 | 641 | 112 | 7.6s | $0.1148 |
| oe_builder | google/gemma-4-26B-A4B-it | v2 | 9 | 0 | 0 | 3900 | 804 | 49.4s | $0.0049 |
| oe_critic | Qwen/Qwen3.6-35B-A3B | v2 | 9 | 0 | 0 | 4736 | 606 | 114.3s | $0.0094 |
| oe_merge_critic | Qwen/Qwen3.6-35B-A3B | v1 | 1 | 0 | 0 | 8525 | 2185 | 84.4s | $0.0029 |

## 2. Ontology Engineering (OE)

- Shards reviewed: **9**, approved **9** (100%)
- Force-accepted after the iteration limit: **0**
- Builder iterations: 9 · critic reviews: 9
- Issues raised: **20** (0 blocking)
- Accepted ontology: 7 entity types, 12 relation types (overlay: 5 / 7)

### Per-shard loop

| Shard | Iterations | Approved | Issues | Blocking | Entities | Relations | Reused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lei_5869_1973 | 1 | yes | 2 | 0 | 3 | 4 | no |
| lei_13105_2015 | 1 | yes | 3 | 0 | 4 | 4 | no |
| lei_10406_2002 | 1 | yes | 2 | 0 | 4 | 5 | no |
| tail_1 | 1 | yes | 2 | 0 | 4 | 5 | no |
| cf_1988 | 1 | yes | 2 | 0 | 5 | 5 | no |
| dl_3689_1941 | 1 | yes | 2 | 0 | 4 | 6 | no |
| tail_2 | 1 | yes | 2 | 0 | 5 | 5 | no |
| tail_3 | 1 | yes | 3 | 0 | 5 | 7 | no |
| dl_5452_1943 | 1 | yes | 2 | 0 | 6 | 6 | no |

### Critic issues

| Category | Severity | Count |
| --- | --- | --- |
| type_specificity | advisory | 4 |
| redundancy | advisory | 4 |
| naming_clarity | advisory | 4 |
| generalidade | advisory | 2 |
| granularidade | advisory | 2 |
| relation_coverage | advisory | 1 |
| nomeclatura | advisory | 1 |
| relation_redundancy | advisory | 1 |
| type_generalization | advisory | 1 |

Most frequent issue texts:

- (1×) A distinção entre 'LegalConcept' e 'ProceduralAct' pode ser difícil de manter consistentemente na extração automática. Termos como 'Sentença' (Art. 483) ou 'Nulidade' (Art. 243) são atos processuais, mas também conceitos jurídicos. A relação 'regulates' exige que o artigo discipline a execução/forma, enquanto 'defines' estabelece a natureza. Na prática normativa, a definição de um instituto processual (ex: o que é nulidade) e a regulação de seu ato (ex: quando ocorre) estão frequentemente no mesmo artigo ou próximos, gerando ambiguidade na classificação do tipo alvo.
- (1×) A relação 'attributes_duty_or_power_to' cobre bem competências (Art. 139, 125), mas a amostra contém artigos que estabelecem condições de capacidade ou legitimidade (Art. 3, 7) que não são exatamente 'deveres/poderes' atribuídos a um sujeito existente, mas sim requisitos para a existência de um sujeito processual válido. A relação 'establishes_requirement_for' cobre parcialmente isso, mas pode ser mais eficaz se aplicada a 'LegalSubject' também (ex: requisito para ser parte).
- (1×) O tipo 'LegalSubject' é extremamente genérico. Na amostra do CPC, os sujeitos são tipicamente 'autor', 'réu', 'juiz', 'exequente', 'executado'. A distinção entre 'sujeito de direito' e 'parte no processo' pode ser útil, mas a amostra foca quase exclusivamente em partes processuais. Considerar renomear para 'ProceduralParty' ou 'ProcessualSubject' para alinhar melhor com a natureza processual do corpus, evitando confusão com sujeitos de direito material (ex: pessoas jurídicas não partes).
- (1×) O tipo 'LegalConcept' pode absorver muitos termos que hoje estão em 'LegalRequirement' ou 'ProceduralAct'. Por exemplo, 'Coisa Julgada' (conceito) está ligada diretamente a efeitos processuais. A distinção entre 'instituto' e 'ato' é clara, mas 'condições de direito' pode ser muito amplo. Avaliar se 'Tutela Específica' (Art. 536) é um conceito ou um tipo de provimento/ato. A amostra sugere que 'LegalConcept' pode ficar muito vago se não houver limites claros de extração.
- (1×) O nome 'LegalRequirement' é funcional, mas 'ProceduralCondition' ou 'NormativeConstraint' poderia ser mais preciso para prazos e requisitos formais. A amostra mostra claramente prazos ('15 dias') e requisitos de petição. O termo 'Requirement' é correto, mas 'Condition' pode capturar melhor a natureza condicional ('se não contestar, será revel').
- (1×) O tipo 'LegalSubject' é extremamente genérico. Na amostra, os sujeitos são tipicamente 'devedor', 'credor', 'terceiro', 'juiz' ou 'pessoa natural/jurídica'. Considerando que a ontologia será consolidada com outros conjuntos, recomenda-se verificar se 'LegalSubject' deve ser subdividido (ex: LegalPerson, LegalRole) ou mantido como um 'bucket' único para evitar fragmentação excessiva na fase de consolidação.
- (1×) A distinção entre 'LegalEffect' (consequência jurídica) e 'LegalObligation' (modalidade/estado) pode ser sutil em alguns artigos. Por exemplo, o Art. 334 diz que o depósito 'extingue a obrigação'. Isso é um efeito (extinção) ou uma mudança de estado da obrigação? A proposta trata 'extinção' como efeito, o que é correto, mas a relação 'establishes_effect' pode conflitar semanticamente com 'regulates_obligation' se não houver clareza sobre quando uma consequência é um 'efeito' distinto da própria regulação da obrigação.
- (1×) Os tipos 'LegalSubject' e 'PublicPower' apresentam sobreposição conceitual significativa na amostra. A Constituição trata da União, Estados e Municípios (LegalSubjects) e dos Poderes (PublicPower) de forma integrada na estrutura federativa e de separação de poderes. Para a recuperação de artigos da OAB, distinguir rigidamente entre 'ente federativo' e 'poder/órgão' pode fragmentar a busca desnecessariamente, já que muitos artigos definem competências que abrangem ambos ou se aplicam à administração pública em geral. Uma fusão para 'StateEntity' ou 'PublicActor' simplifica a ontologia sem perder a utilidade prática para a maioria das questões.
- (1×) O tipo 'SocialDuty' é definido como 'Deveres do Estado ou da sociedade'. No entanto, a relação 'attributes_duty_to' sugere que o dever é atribuído *a* um sujeito. Na amostra, o Art. 205 ('educação... dever do Estado e da família') e o Art. 144 ('segurança pública, dever do Estado') indicam que o 'dever' é uma obrigação normativa imposta a um ator. O termo 'SocialDuty' pode ser confundido com 'Direito Social' (como saúde e educação listados no Art. 6). Seria mais claro nomear o tipo como 'NormativeObligation' ou 'PublicDuty' para distinguir claramente de direitos subjetivos.
- (1×) O tipo 'PublicAuthority' é um subconjunto específico de 'LegalSubject'. Na amostra, a ANPD é qualificada como 'pessoa jurídica de direito público' (Art. 23 LGPD) e os Juizados como 'órgãos da Justiça Ordinária' (Art. 1 LJE). A distinção atual exige que o recuperador saiba se um termo é 'autoridade' ou 'sujeito' para escolher a relação correta, o que pode ser ambíguo em textos normativos genéricos. Considerando que a consolidação unificará camadas, sugere-se avaliar se 'PublicAuthority' deve ser fundido a 'LegalSubject' com um atributo de 'nature' ou se mantido apenas para relações de competência específicas.

### Types dropped by the support filter

| Kind | Name | Support | Threshold |
| --- | --- | --- | --- |
| entity | LegalEffect | 1 | 2 |
| entity | LegalInstrument | 1 | 2 |
| entity | LegalRequirement | 1 | 2 |
| relation | establishes_effect | 1 | 2 |
| relation | requires_instrument | 1 | 2 |

### Merge bookkeeping repaired

| Contradiction | Count | Names |
| --- | --- | --- |
| Survivor also listed as dropped | 1 | LegalSubject |
| Name absorbed into itself | 6 | LegalEffect, LegalInstrument, LegalObligation, LegalRequirement, LegalSanction, establishes_effect |
| Survivor listed as absorbed | 4 | LegalAction, LegalEffect, attributes_competence_to, requires_instrument |

## 3. Knowledge Extraction (KE)

- Batches: **4842** (26 failed, 60 empty, 3 truncated)
- Articles covered: 14078
- Triples kept: **53754** (11.10 per batch)
- Retried batches: 3095, improved 55% of the time
- Violations: **35612**

### Violations by code

| Code | Count |
| --- | --- |
| non_atomic_span | 9063 |
| low_quality | 6753 |
| unsupported | 6326 |
| non_verbatim_span | 5652 |
| wrong_modality | 4336 |
| inverted | 2029 |
| unknown_relation | 630 |
| duplicate | 604 |
| repeated_triple | 212 |
| unknown_article | 5 |
| unknown_modality | 2 |

## 4. Knowledge Fusion (KF)

- Exact-duplicate merges: 8316
- Citation merges: 336
- Similarity merges: 8500
- Agent merges accepted / rejected: 467 / 695
- Clusters that failed to parse: 0
- Fusion rounds: 5 (blocking: embedding)

### Graph

- Nodes: **54730** · Edges: **123645** · Ontology `kg_v1_oe`

| Node type | Count |
| --- | --- |
| LegalConcept | 16514 |
| LegalAction | 15204 |
| Article | 14078 |
| LegalObligation | 3618 |
| LegalSubject | 3399 |
| LegalSanction | 1775 |
| NormativeAct | 142 |

| Relation | Edges |
| --- | --- |
| mentions | 52223 |
| regulates | 24304 |
| contains | 14078 |
| requires | 11625 |
| refers_to | 4708 |
| prohibits | 4325 |
| grants | 4034 |
| attributes_competence_to | 3116 |
| establishes_sanction | 2880 |
| defines | 2220 |
| amends | 92 |
| revokes | 40 |

## 5. Errors and health

- Calls that never parsed: **19**
- Calls stopped by the token limit: **76**
- Decision events: `batch_done`×4894, `cluster_skipped`×3797, `retry`×3124, `cluster_rejected`×836, `cluster_accepted`×550, `codes_normalized`×34, `batch_failed`×26, `shard_proposal`×9, `shard_review`×9, `type_dropped`×5, `blocking`×5, `auto_merge`×3, `citations_routed`×3, `update_reopened`×3, `batches_redone`×3, `graph_saved`×2, `shards_built`×1, `merge`×1, `merge_sanitized`×1, `merge_unaccounted`×1, `overlay_accepted`×1, `dangling_types_pruned`×1

### Failed calls

| Call | Agent | Context | Attempts | Finish | Error |
| --- | --- | --- | --- | --- | --- |
| ke-03051 | ke_builder | {"batch_id": "lei_8212_1991:art_68@1991", "act_id": "lei_821 | 2 | stop | Agent 'ke_builder' failed after 2 attempts: Invalid JSON: Expecting ',' delimiter: line 63 column 211 (char 2532) |
| ke-04277 | ke_builder | {"batch_id": "lei_10257_2001:art_39@2001", "act_id": "lei_10 | 2 | stop | Agent 'ke_builder' failed after 2 attempts: Invalid JSON: Expecting ',' delimiter: line 217 column 2261 (char 8422) |
| ke-07229 | ke_builder | {"batch_id": "lei_10406_2002:art_1380@2002", "act_id": "lei_ | 2 | stop | Agent 'ke_builder' failed after 2 attempts: Invalid JSON: Extra data: line 112 column 3 (char 5113) |
| ke-07807 | ke_critic | {"batch_id": "lei_13019_2014:art_38@2015", "act_id": "lei_13 | 0 | None | APIStatusError: Error code: 499 - {'error': {'message': 'client disconnected', 'type': 'invalid_request_error', 'param': |
| ke-08127 | ke_critic | {"batch_id": "lei_6019_1974:art_19-C@2017", "act_id": "lei_6 | 0 | None | APIStatusError: Error code: 499 - {'error': {'message': 'client disconnected', 'type': 'invalid_request_error', 'param': |
| ke-15814 | ke_builder | {"batch_id": "lei_12965_2014:art_10@2014", "act_id": "lei_12 | 0 | None | APITimeoutError: Request timed out. |
| ke-15815 | ke_builder | {"batch_id": "lei_8429_1992:art_12@1992", "act_id": "lei_842 | 0 | None | APITimeoutError: Request timed out. |
| ke-15816 | ke_builder | {"batch_id": "cf_1988:art_125@1988", "act_id": "cf_1988", "a | 0 | None | APITimeoutError: Request timed out. |
| ke-15817 | ke_builder | {"batch_id": "cf_1988:art_61@1988", "act_id": "cf_1988", "ar | 0 | None | APITimeoutError: Request timed out. |
| ke-15818 | ke_builder | {"batch_id": "dl_5452_1943:art_799@1946", "act_id": "dl_5452 | 0 | None | APITimeoutError: Request timed out. |
| ke-15819 | ke_builder | {"batch_id": "lei_5869_1973:art_20@1976", "act_id": "lei_586 | 0 | None | APITimeoutError: Request timed out. |
| ke-15820 | ke_builder | {"batch_id": "lei_8212_1991:art_32-A@2009", "act_id": "lei_8 | 0 | None | APITimeoutError: Request timed out. |
| ke-15821 | ke_builder | {"batch_id": "decreto_592_1992:art_2@1992", "act_id": "decre | 0 | None | APITimeoutError: Request timed out. |
| ke-15822 | ke_builder | {"batch_id": "lei_5869_1973:art_527@2001", "act_id": "lei_58 | 0 | None | APITimeoutError: Request timed out. |
| ke-15823 | ke_builder | {"batch_id": "dl_2848_1940:art_343@2001", "act_id": "dl_2848 | 0 | None | APITimeoutError: Request timed out. |
| ke-15824 | ke_builder | {"batch_id": "dl_2848_1940:art_149-A@2016", "act_id": "dl_28 | 0 | None | APITimeoutError: Request timed out. |
| ke-15825 | ke_builder | {"batch_id": "lei_13303_2016:art_75@2016", "act_id": "lei_13 | 0 | None | APITimeoutError: Request timed out. |
| ke-15826 | ke_builder | {"batch_id": "lei_9279_1996:art_230@1996", "act_id": "lei_92 | 0 | None | APITimeoutError: Request timed out. |
| ke-15827 | ke_builder | {"batch_id": "lei_5869_1973:art_694@2006", "act_id": "lei_58 | 0 | None | APITimeoutError: Request timed out. |

### Truncated calls

| Call | Agent | Out tokens | Max tokens | Context |
| --- | --- | --- | --- | --- |
| ke-00313 | ke_builder | 4096 | 4096 | {"batch_id": "cf_1988:art_5@1988", "act_id": "cf_1988", "art |
| ke-00467 | ke_builder | 4096 | 4096 | {"batch_id": "lc_64_1990:art_1@1990", "act_id": "lc_64_1990" |
| ke-00640 | ke_builder | 4096 | 4096 | {"batch_id": "lei_5869_1973:art_20@1976", "act_id": "lei_586 |
| ke-00671 | ke_builder | 4096 | 4096 | {"batch_id": "cf_1988:art_17@1988", "act_id": "cf_1988", "ar |
| ke-00740 | ke_builder | 4096 | 4096 | {"batch_id": "cf_1988:art_22@1988", "act_id": "cf_1988", "ar |
| ke-01407 | ke_builder | 4096 | 4096 | {"batch_id": "lei_9394_1996:art_36-C@2008", "act_id": "lei_9 |
| ke-01566 | ke_builder | 4096 | 4096 | {"batch_id": "cf_1988:art_61@1988", "act_id": "cf_1988", "ar |
| ke-01634 | ke_builder | 4096 | 4096 | {"batch_id": "lei_9394_1996:art_36-C@2008", "act_id": "lei_9 |
| ke-02511 | ke_builder | 4096 | 4096 | {"batch_id": "cf_1988:art_125@1988", "act_id": "cf_1988", "a |
| ke-02615 | ke_builder | 4096 | 4096 | {"batch_id": "lei_9279_1996:art_230@1996", "act_id": "lei_92 |
| ke-02797 | ke_builder | 4096 | 4096 | {"batch_id": "lei_8212_1991:art_32-A@2009", "act_id": "lei_8 |
| ke-02848 | ke_builder | 4096 | 4096 | {"batch_id": "lei_8934_1994:art_15@1994", "act_id": "lei_893 |
| ke-04636 | ke_builder | 4096 | 4096 | {"batch_id": "lei_5869_1973:art_295@1973", "act_id": "lei_58 |
| ke-05325 | ke_builder | 4096 | 4096 | {"batch_id": "dl_2848_1940:art_121@1940", "act_id": "dl_2848 |
| ke-05405 | ke_builder | 4096 | 4096 | {"batch_id": "dl_2848_1940:art_149-A@2016", "act_id": "dl_28 |
| ke-05566 | ke_builder | 4096 | 4096 | {"batch_id": "lei_5869_1973:art_527@2001", "act_id": "lei_58 |
| ke-06266 | ke_builder | 4096 | 4096 | {"batch_id": "lc_123_2006:art_3@2006", "act_id": "lc_123_200 |
| ke-06374 | ke_builder | 4096 | 4096 | {"batch_id": "lc_123_2006:art_18@2006", "act_id": "lc_123_20 |
| ke-06621 | ke_builder | 4096 | 4096 | {"batch_id": "lei_5869_1973:art_694@2006", "act_id": "lei_58 |
| ke-06684 | ke_builder | 4096 | 4096 | {"batch_id": "decreto_592_1992:art_2@1992", "act_id": "decre |
| ke-06798 | ke_builder | 4096 | 4096 | {"batch_id": "lei_13303_2016:art_75@2016", "act_id": "lei_13 |
| ke-06933 | ke_builder | 4096 | 4096 | {"batch_id": "lei_13303_2016:art_75@2016", "act_id": "lei_13 |
| ke-07872 | ke_builder | 4096 | 4096 | {"batch_id": "dl_2848_1940:art_218-C@1940", "act_id": "dl_28 |
| ke-08851 | ke_builder | 4096 | 4096 | {"batch_id": "dl_2848_1940:art_343@2001", "act_id": "dl_2848 |
| ke-08997 | ke_builder | 4096 | 4096 | {"batch_id": "lei_12462_2011:art_9@2011", "act_id": "lei_124 |
| ke-09109 | ke_builder | 4096 | 4096 | {"batch_id": "lei_6404_1976:art_55@1976", "act_id": "lei_640 |
| ke-10965 | ke_builder | 4096 | 4096 | {"batch_id": "lei_9503_1997:art_19@1997", "act_id": "lei_950 |
| ke-11410 | ke_builder | 4096 | 4096 | {"batch_id": "dl_5452_1943:art_799@1946", "act_id": "dl_5452 |
| ke-12511 | ke_builder | 4096 | 4096 | {"batch_id": "dl_5452_1943:art_899@1968", "act_id": "dl_5452 |
| ke-12516 | ke_builder | 4096 | 4096 | {"batch_id": "lei_4737_1965:art_165@1965", "act_id": "lei_47 |
| ke-12671 | ke_builder | 4096 | 4096 | {"batch_id": "lei_5172_1966:art_205@1966", "act_id": "lei_51 |
| ke-12742 | ke_builder | 4096 | 4096 | {"batch_id": "lei_6830_1980:art_7@2014", "act_id": "lei_6830 |
| ke-12877 | ke_builder | 4096 | 4096 | {"batch_id": "lei_6830_1980:art_7@2014", "act_id": "lei_6830 |
| ke-14220 | ke_builder | 4096 | 4096 | {"batch_id": "cf_1988:adct:art_41@1988", "act_id": "cf_1988" |
| ke-14687 | ke_builder | 4096 | 4096 | {"batch_id": "lei_12965_2014:art_10@2014", "act_id": "lei_12 |
| ke-15074 | ke_builder | 4096 | 4096 | {"batch_id": "dl_3689_1941:art_167@1941", "act_id": "dl_3689 |
| ke-15719 | ke_builder | 4096 | 4096 | {"batch_id": "lei_8429_1992:art_12@1992", "act_id": "lei_842 |
| ke-15842 | ke_builder | 8192 | 8192 | {"batch_id": "lei_5869_1973:art_527@2001", "act_id": "lei_58 |
| ke-15843 | ke_builder | 8192 | 8192 | {"batch_id": "lei_12965_2014:art_10@2014", "act_id": "lei_12 |
| ke-15844 | ke_builder | 8192 | 8192 | {"batch_id": "lei_8212_1991:art_32-A@2009", "act_id": "lei_8 |

## 6. Figures

### Tokens by agent

![Tokens by agent](figures/kg_tokens_by_agent.png)

### Call latency by agent

![Call latency by agent](figures/kg_latency_by_agent.png)

### Cumulative tokens over the build

![Cumulative tokens over the build](figures/kg_tokens_timeline.png)

### OE critic issues

![OE critic issues](figures/kg_oe_issues.png)

### KE violations

![KE violations](figures/kg_ke_violations.png)

### Triples per batch

![Triples per batch](figures/kg_triples_per_batch.png)

### Graph composition

![Graph composition](figures/kg_graph_composition.png)

## 7. Provenance

Last run `run_008` · commit `72d60ad2a6c6` on branch `main` (working tree dirty)

### Inputs

| Role | Path | Bytes | sha256 |
| --- | --- | --- | --- |
| laws_csv | `data/laws/laws_v3_2018.csv` | 18813191 | 4bddeec1f630 |
| ontology_core | `data/kg/ontology/core_v1.json` | 1939 | 18ba143488a9 |
| ontology_seed | `data/kg/ontology/overlay_seed_v1.json` | 1974 | 2ec450d8be57 |
| articles_csv | `data/laws/articles_v4.csv` | 25607238 | 0cf0c76953c8 |

### Agent configuration

| Agent | Model | Prompt | Prompt sha256 | Temp | Max tokens | num_ctx |
| --- | --- | --- | --- | --- | --- | --- |
| oe_builder | google/gemma-4-26B-A4B-it | v2 | 4313b36a462c | 0.0 | 4096 | None |
| oe_critic | Qwen/Qwen3.6-35B-A3B | v2 | dd83468b4072 | 0.0 | 4096 | None |
| oe_merge_critic | Qwen/Qwen3.6-35B-A3B | v1 | 5f86a8c51f50 | 0.0 | 4096 | None |
| ke_builder | Qwen/Qwen3.5-9B | v6 | ffab2457f94f | 0.0 | 8192 | None |
| ke_critic | google/gemma-4-26B-A4B-it | v4 | 884e7c1b866f | 0.0 | 4096 | None |
| kf_builder | Qwen/Qwen3.5-9B | v2 | edb632236c96 | 0.0 | 4096 | None |
| kf_critic | google/gemma-4-26B-A4B-it | v2 | 00e17560372a | 0.0 | 4096 | None |

