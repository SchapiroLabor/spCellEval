from models_classes.gridsearch import GridSearcher
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

rs = 20240925
param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 5, 10],
    'max_iter': [100, 500, 1000, 2000, 5000],
    'class_weight': ['balanced', None]
}

cHL2 = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
cHL2.train_tune_evaluate(path='/datasets/Maps_data/cHL_2_MIBI/quantification/processed/kfolds', param_grid=param_grid)
cHL2.save_results(save_path='/datasets/Maps_data/cHL_2_MIBI/results_logreg_gscv', label_path='/datasets/Maps_data/cHL_2_MIBI/quantification/processed/labels.csv')

cHL1 = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
cHL1.train_tune_evaluate(path='/datasets/Maps_data/cHL_1_MIBI/quantification/processed/kfolds', param_grid=param_grid)
cHL1.save_results(save_path='/datasets/Maps_data/cHL_1_MIBI/results_logreg_gscv', label_path='/datasets/Maps_data/cHL_1_MIBI/quantification/processed/labels.csv')

cHL_CODEX = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
cHL_CODEX.train_tune_evaluate(path='/datasets/Maps_data/cHL_CODEX/quantification/processed/kfolds', param_grid=param_grid)
cHL_CODEX.save_results(save_path='/datasets/Maps_data/cHL_CODEX/results_logreg_gscv', label_path='/datasets/Maps_data/cHL_CODEX/quantification/processed/labels.csv')

del cHL2, cHL1, cHL_CODEX

MRL_CODEX = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
MRL_CODEX.train_tune_evaluate(path='/datasets/MRL_CODEX/quantification/processed/kfolds', param_grid=param_grid)
MRL_CODEX.save_results(save_path='/datasets/MRL_CODEX/results_logreg_gscv', label_path='/datasets/MRL_CODEX/quantification/processed/labels.csv')

TB = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
TB.train_tune_evaluate(path='/datasets/TB_MIBI_atlas/quantification/processed/TB/kfolds', param_grid=param_grid)
TB.save_results(save_path='/datasets/TB_MIBI_atlas/quantification/processed/TB/results_logreg_gscv', label_path='/datasets/TB_MIBI_atlas/quantification/processed/TB/labels.csv')

sarc = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
sarc.train_tune_evaluate(path='/datasets/TB_MIBI_atlas/quantification/processed/sarc/kfolds', param_grid=param_grid)
sarc.save_results(save_path='/datasets/TB_MIBI_atlas/quantification/processed/sarc/results_logreg_gscv', label_path='/datasets/TB_MIBI_atlas/quantification/processed/sarc/labels.csv')

del MRL_CODEX, TB, sarc

tonsil_codex = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
tonsil_codex.train_tune_evaluate(path='/datasets/tonsil_CODEX/quantification/processed/kfolds', param_grid=param_grid)
tonsil_codex.save_results(save_path='/datasets/tonsil_CODEX/results_logreg_gscv', label_path='/datasets/tonsil_CODEX/quantification/processed/labels.csv')

tonsil_CODEX2 = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
tonsil_CODEX2.train_tune_evaluate(path='/datasets/tonsil_CODEX2/quantification/processed/kfolds', param_grid=param_grid)
tonsil_CODEX2.save_results(save_path='/datasets/tonsil_CODEX2/results_logreg_gscv', label_path='/datasets/tonsil_CODEX2/quantification/processed/labels.csv')

del tonsil_codex, tonsil_CODEX2

lymphoma_CODEX = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
lymphoma_CODEX.train_tune_evaluate(path='/datasets/lymphoma_CODEX/quantification/processed/kfolds', param_grid=param_grid)
lymphoma_CODEX.save_results(save_path='/datasets/lymphoma_CODEX/results_logreg_gscv', label_path='/datasets/lymphoma_CODEX/quantification/processed/labels.csv')

intestine_CODEX = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
intestine_CODEX.train_tune_evaluate(path='/datasets/intestine_CODEX/quantification/processed/kfolds', param_grid=param_grid)
intestine_CODEX.save_results(save_path='/datasets/intestine_CODEX/results_logreg_gscv', label_path='/datasets/intestine_CODEX/quantification/processed/labels.csv')

del lymphoma_CODEX, intestine_CODEX

feto_maternal = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
feto_maternal.train_tune_evaluate(path='/datasets/feto_maternal/quantification/processed/kfolds', param_grid=param_grid)
feto_maternal.save_results(save_path='/datasets/feto_maternal/results_logreg_gscv', label_path='/datasets/feto_maternal/quantification/processed/labels.csv')

CRC_FFPE = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
CRC_FFPE.train_tune_evaluate(path='/datasets/CRC_FFPE/quantification/processed/kfolds', param_grid=param_grid)
CRC_FFPE.save_results(save_path='/datasets/CRC_FFPE/results_logreg_gscv', label_path='/datasets/CRC_FFPE/quantification/processed/labels.csv')

BE_tonsil = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
BE_tonsil.train_tune_evaluate(path='/datasets/BA_CODEX/quantification/BE_tonsil/processed/kfolds', param_grid=param_grid)
BE_tonsil.save_results(save_path='/datasets/BA_CODEX/quantification/BE_tonsil/results_logreg_gscv', label_path='/datasets/BA_CODEX/quantification/BE_tonsil/processed/labels.csv')

tonsil_training = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
tonsil_training.train_tune_evaluate(path='/datasets/BA_CODEX/quantification/tonsil_training/processed/kfolds', param_grid=param_grid)
tonsil_training.save_results(save_path='/datasets/BA_CODEX/quantification/tonsil_training/results_logreg_gscv', label_path='/datasets/BA_CODEX/quantification/tonsil_training/processed/labels.csv')

del feto_maternal, CRC_FFPE, BE_tonsil, tonsil_training

AML = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
AML.train_tune_evaluate(path='/datasets/AML_bone_marrow/quantification/AML/processed/kfolds', param_grid=param_grid)
AML.save_results(save_path='/datasets/AML_bone_marrow/quantification/AML/results_logreg_gscv', label_path='/datasets/AML_bone_marrow/quantification/AML/processed/labels.csv')

healthy_BM = GridSearcher(random_state=rs, n_jobs=-1, model='logistic_regression')
healthy_BM.train_tune_evaluate(path='/datasets/AML_bone_marrow/quantification/healthy_BM/processed/kfolds', param_grid=param_grid)
healthy_BM.save_results(save_path='/datasets/AML_bone_marrow/quantification/healthy_BM/results_logreg_gscv', label_path='/datasets/AML_bone_marrow/quantification/healthy_BM/processed/labels.csv')