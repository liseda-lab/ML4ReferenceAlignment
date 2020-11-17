# ML-For-Ontology-Matching

We present a preliminary investigation that focuses on the generalization of machine learning models trained on the output alignments of multiple systems for a task where a reference alignment is available to other alignment tasks.
This work was presented at the [Ontology Matching Workshop](http://om2020.ontologymatching.org/) co-located with [ISWC2020](http://iswc2020.semanticweb.org/).
Link to [paper](http://www.di.fc.ul.pt/~catiapesquita/papers/lima-ref-alignments-om2020.pdf).


## Data
The alignments produced by the ontology matching tools that participated in the Anatomy, Large BioMed and Conference tracks of OAEI 2019 were used as datasources.
These data are available for download here:
 - Anatomy: http://oaei.ontologymatching.org/2019/results/anatomy/index.html
 - Large BioMed: https://github.com/ernestojimenezruiz/oaei-evaluation/tree/master/mappings/largebio/2019
 - Conference: http://oaei.ontologymatching.org/2019/results/conference/#data
 
## Results
![boxplots](https://user-images.githubusercontent.com/43668147/89839601-08435900-db66-11ea-98a1-6ecd04adb34b.png)

 - The full table of f1-score results can be found in the results-table.xlsx file.
 - Information on hyperparameters, accuracy and precision can be found in the hyperparameter.csv file.
 
 ## How to cite
 
Beatriz Lima, Ruben Branco, João Castanheira, Gustavo Fonseca, & Catia Pesquita (2020). Learning reference alignments for ontology matching within and across domains. In Ontology Matching Workshop co-located with International Semantic Web Conference.
 
@inproceedings {lima-learning2020,
    author    = "Beatriz Lima and Ruben Branco and João Castanheira and Gustavo Fonseca and Catia Pesquita",
    title     = "Learning reference alignments for ontology matching within and across domains",
    booktitle = "Ontology Matching Workshop co-located with International Semantic Web Conference",
    year      = "2020"
}


## Authors
- Beatriz Lima
- Ruben Branco
- João Castanheira
- Gustavo Fonseca
- Cátia Pesquita

## License
See the LICENSE file for details.

## Acknowledgements
CP and BL are funded by the FCT through LASIGE Research Unit, ref. UIDB/00408/2020 and ref. UIDP/00408/2020, and by projects SMILAX (ref. PTDC/EEI-ESS/4633/2014). CP is also funded by GADgET (ref. DSAIPA/DS/0022/2018).
