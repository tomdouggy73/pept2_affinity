**Overview**
This repository contains ligand-only Random Forest Regression models for affinity predictions to Solute Carrier family protein SLC15a2 (PEPT2). This work is a continuation of the research of Dr Simon Lichtinger, aided by Dr Matthew Warren, and scripts they have authored are detailed in the repo structure. 

This research, for my part II project in Biochemistry, aims to assess the validity of different evaluation techniques on small training sets and understand the impacts of model bias. We hope to optimise Dr Lichtinger's Random Forest model to reduce overfitting and implement a post-hoc filter to evaluate the nature of predicted PEPT2 binders - substrate or inhibitor. The structure of the repo is detailed below:

**Repo Structure:**
  - adhoc: code written by myself to filter the zinc canocialized database to isolate compounds with potential amine mimics, with hope to elucidate new chemistry with high pept2 binding affinity.
  - matt_code: contains the source code in ml_networks for the random forest models, authored by Dr Warren with some edits made for data complatibility
  - penG_predictions: contains code written by Dr Lichtinger to predict the penG ligands using tuned models.
  - peptides: contains a script written by Dr Lichtinger to generate peptide smiles.
  - posthoc: contains posthoc_filter.py, written by myself to classify candidate compounds as susbtrates/inhibitors based on the presence of a primary amine and carboxyl groups, building from the structural work in Parker et al 2024. (see ref below)
  - regression_models: implementation of the source code in ml_networks, with 3 different descriptor inputs, written by Simon Lichtinger. Nested CV, held out test set and XGboost are improved algorithms written by myself.
  - tanimoto_based_clusters: variety of scripts written by myself which cluster the data based on diversity, using a precalculated tanimoto matrix to calculate distances.
  - noise_estimator: data preprocessing scripts written by myself to make our affinity data comptabile with the 'noise estimator', Crusius et al 2024. (see ref below)
  - vendi: scripts written by myself, making use of the vendi_score library to assess the diversity of different clustering arrangements generated in tanimoto_based_clusters
  - zinc: scripts written by myself to divide the zinc database into clusters for parallisation, then use final regression models to predict affinities of the zinc database


**Key references:**
  - Parker et al 2024 - Structural basis for antibiotic transport and inhibition in PepT2    Nature Communications | (2024) 15:8755
  - Crusius et al 2024 - Are we fitting data or noise? Analysing the predictive power of commonly used datasets in drug-, materials-, and molecular-discovery.    DOI: 10.1039/D4FD00091A (Paper) Faraday Discuss., 2024, Advance Article
