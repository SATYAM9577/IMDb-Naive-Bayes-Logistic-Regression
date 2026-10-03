# Three TF-IDF model disagreement case studies

Review row numbers refer to zero-based row positions in the official IMDb test parquet split. Review text is not reproduced; the word evidence below is based on each review's active TF-IDF features.

## Case 1 — test row 4

- Actual: **negative**
- MultinomialNB TF-IDF: **negative**, probability of its predicted class **0.8107**.
- Logistic Regression TF-IDF: **positive**, probability of its predicted class **0.8825**.
- NB strongest active-feature log-odds contributions: `damme` (-0.6589), `segal` (-0.3188), `dolph` (-0.2413), `nowhere` (-0.1333), `bloodsport` (-0.1230).
- Logistic strongest active-feature contributions: `best` (+0.5815), `damme` (-0.4427), `enjoyed` (+0.3630), `highly` (+0.3503), `fun` (+0.3301).

**Case-specific analysis:** This review is a mixed, audience-qualified appraisal of an action film: the writer describes enjoyment and recommends it to genre/actor fans, while qualifying its plot and broader appeal. NB's largest negative terms are actor names (`damme`, `segal`, `dolph`) whose corpus-level associations outweigh the scattered positive terms under its independent-feature likelihood sum. Logistic Regression instead weights evaluative evidence such as `enjoyed`, `best`, `highly`, and `fun`, so it predicts positive. The apparent mismatch with the negative test label is consistent with a mixed review and a star-derived binary label that a word model cannot infer from qualification alone.

## Case 2 — test row 22194

- Actual: **positive**
- MultinomialNB TF-IDF: **negative**, probability of its predicted class **0.7644**.
- Logistic Regression TF-IDF: **positive**, probability of its predicted class **0.8945**.
- NB strongest active-feature log-odds contributions: `seagal` (-0.9216), `unrealistic` (-0.1297), `sorry` (-0.1027), `unfortunately` (-0.0809), `fun` (+0.0752).
- Logistic strongest active-feature contributions: `fun` (+0.5224), `seagal` (-0.4967), `unfortunately` (-0.3721), `well` (+0.3568), `best` (+0.3209).

**Case-specific analysis:** This positive review discusses a Steven Seagal film while saying it differs from the actor's familiar formula. Several Seagal-related tokens have learned negative class-conditional associations, with `seagal` dominating NB's negative evidence. But the review's overall assessment is favorable: Logistic Regression's positive contributions from `fun`, `best`, and `well` exceed its negative evidence and it is substantially more confident. This case shows NB can overgeneralize topic/person-name correlations, while LR can use discriminative sentiment cues; neither model fully resolves the scope of the few negative clauses.

## Case 3 — test row 23947

- Actual: **positive**
- MultinomialNB TF-IDF: **negative**, probability of its predicted class **0.6963**.
- Logistic Regression TF-IDF: **positive**, probability of its predicted class **0.9048**.
- NB strongest active-feature log-odds contributions: `avoid` (-0.1884), `crap` (-0.1868), `stigmata` (-0.1730), `excellent` (+0.1253), `solid` (+0.1071).
- Logistic strongest active-feature contributions: `excellent` (+0.5320), `great` (+0.4185), `still` (+0.4052), `avoid` (-0.3571), `well` (+0.3498).

**Case-specific analysis:** This positive review praises a horror film, but its strongest NB negative evidence includes `crap` and `avoid`. In context, those words criticize other contemporary horror movies and advise against a later sequel, rather than criticize the film being reviewed. Unigram TF-IDF does not represent which movie a sentiment word refers to, so NB adds those terms as negative evidence. Logistic Regression is pushed positive by `great`, `excellent`, and `solid`, which more directly describe the reviewed film. This is a clear target/scope error caused by losing phrase context and reference in the bag-of-words features.
