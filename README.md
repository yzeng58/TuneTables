<br/>
<p align="center"><img src="img/tunetables_logo.png" width=700 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)

We introduce TuneTables, a tabular classification algorithm that overcomes the limitations of prior-data fitted networks to achieve strong performance on large datasets.

While TabPFN achieves very strong performance on small tabular datasets, its current limitations include fewer than 1000 datapoints, fewer than 100 features, and fewer than 10 class labels. In this work, we overcome these limitations and substantially improve the performance of PFNs by developing context optimization techniques; specifically, we propose TuneTables, a novel prompt-tuning strategy. TuneTables scales TabPFN to be competitive with state-of-the-art tabular classification methods on larger datasets, while having additional benefits as well: (1) substantially lower inference time than TabPFN, (2) can be used as an interpretability tool, and (3) can mitigate biases by optimizing a fairness objective.

<p align="center"><img src="img/tunetables_overview.png" width=700 /></p>

This codebase extends the excellent public repository [TabPFN]([xxx](https://github.com/automl/tabpfn)), by Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter.

## Installation

1. Clone TabPFN-pt (TuneTables) repository to your local instance.
2. From the TabPFN-pt directory, run --

```bash
pip install . 
```

## Getting started

## TabPFN-PT notes

* Everything expects to run from TabPFN-pt/tabpfn

python3 batch/run_tabpfn_job.py --resume models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt --base_path data --datasets metadata/test_datasets.txt --tasks metadata/test_tasks.txt --bptt 128