# Benchmarking Cell Phenotyping Methods in Spatial Porteomics.(spCell-Eval)

We present, "spCellEval Benchmark", a quantitative comparison of automated/semi-automated cell phenotyping methods for Spatial Proteomics datasets on a diverse set of 9 curated public datasets. The methods are compared with a list of label transfer metrics divided into 4 categories; classification performance, distribution recovery, stability and scalability. We hope this benchmark acts as a foundation to evaluate and improve automated cell phenotyping. 

![Alt text](img/benchmark_overview.png?raw=true "Title")

## Current Results Overview: 
![Alt text](img/results_overview.png?raw=true "Title")

## Getting Started

Scripts to run each method are proveided within `src/<method>`.
Parameters to recreate the each method can be found here.

### Evaluation Script
The notebook, eval.ipynb, can be used to get the complete metrics on all methods for each dataset. 

### Adding your own method
To officially add your own method, Please provide us with the following to reproduce
1. GitHub repo for the method
2. List of Parameters use (if any)
3. Your predictions (optional: would make things faster for us)

Folder Strucutre to add you predictions in 
```
results/
├── Method1/
│   └── predictions_*.csv
├── Method2/
└── Method3/
```

## TODO:
- [x] Add overview (and results) figure
- [ ] remove .DS files
- [x] expand text

