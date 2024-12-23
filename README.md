# neoseg
A tool for Lexematic Segmentation by Paul Lerner.

## Introduction

Unlike modern approaches (e.g. https://aclanthology.org/2022.sigmorphon-1.11) that words into a sequence of morphemes 
(e.g. "invaluable" = "in|value|able") we aim to segment a given lexeme into an affix and a base 
(e.g. "in- valuable" where "valuable" can be further decomposed into "value -able").
This lexematic approach has not been adressed recently, and is closer to morphosemantic analysis of [Dérif](https://www.cnrtl.fr/outils/DeriF/) 
(Namer, 2003 http://w3.erss.univ-tlse2.fr/publications/CDG/28/CG28-3-Namer.pdf), which is only avaible for French.

## Methods

Our first experiments are based upon [MorphyNet](
https://github.com/kbatsuren/MorphyNet/tree/378144f64df58c78db5245af19d16a511ccecf3a), which supports 15 languages

We compared three systems:
- an encoder-decoder biLSTM with attention trained from scratch (baseline system of Peters and Martins, 2022 https://aclanthology.org/2022.sigmorphon-1.14/)
- fine-tuning mT5 https://aclanthology.org/2021.naacl-main.41
- fine-tuning byT5 https://doi.org/10.1162/tacl_a_00461

Note the Peters and Martins proposed to replace softmax with a sparse version. 
However, we had trouble using their code which is apparently not maintained anymore https://github.com/deep-spin/entmax/issues/21 

## Results
### Quantitative
| Test metric          | FR LSTM | FR byT5 | EN LSTM |
| -------------------- | ------- | ------- | ------- |
| eval/BinaryAccuracy  | 91.9%   | 93.7%   | 94.9%   |
| eval/BinaryF1Score   | 85.2%   | 88.6%   | 94.6%   |
| eval/BinaryPrecision | 85.7%   | 88.7%   | 94.2%   |
| eval/BinaryRecall    | 84.8%   | 88.5%   | 95.0%   |
| eval/em              | 68.4%   | 74.2%   | 87.3%   |

Our preliminary results show that:
- byT5 outperforms LSTM (see above table)
- EN is easier than FR (more data, perhaps less allomorphies and orthographic rules:
  79% affixations are purely orthographic and concatenative in the EN corpus vs. 39% for FR)
- ByT5 outperforms mT5 or, at least, is much faster to converge.

### Error analysis
The EM above seems fairly low. 
However, there are ambiguous case that probably stem from annotation inconsistency and should not be considered errors.
For example:
- `réutilisabilité | ré<pre>utilisabilité | réutilisable<suff>ité` (suffix or prefix?)
-  `potinage | potin<suff>age | potiner<suff>age` (verbal or nominal base?)
- `hautement | haut<suff>ment | hauter<suff>ment` 
  (also a problem with the base but this time relates more to semantic, both derivatives would be interpreted differently)
- `palification | palifier<suff>ification | palifier<suff>ation` (normalization of allomorphies)

## Future Work

- more languages, more extensive experiments/hyperparameter tuning
- add monomorphemic lexemes
- experiment with multilingual training

Feel free to contribute!

# Acknowledgements

This preliminary work was partly advised by François Yvon 
and funded by the French Agence Nationale de la Recherche (ANR) under the project MaTOS - “ANR-22-CE23-0033-03”.
