from models_classes.gridsearch_multi import Tune_Eval
import warnings
warnings.filterwarnings("ignore")

base_path = '/gpfs/bwfor/work/ws/hd_bm327-phenotyping_benchmark/phenotyping_benchmark/datasets
rs = 20240925
n = 32
param_grid = {
    'C': [0.1, 0.5, 1, 5],
    'max_iter': [100, 250, 500,],
    'class_weight': ['balanced', None]
}

cHL2 = Tune_Eval(random_state = rs, model = 'logistic_regression')
cHL2.train_tune_evaluate(path=base_path + '/Maps_data/cHL_2_MIBI/quantification/processed/kfolds', param_grid=param_grid, n_processes=n)
cHL2.save_results(save_path=base_path + '/Maps_data/cHL_2_MIBI/results_logreg', label_path=base_path + '/Maps_data/cHL_2_MIBI/quantification/processed/labels.csv')

cHL1 = Tune_Eval(random_state = rs, model = 'logistic_regression')
cHL1.train_tune_evaluate(path=base_path + '/Maps_data/cHL_1_MIBI/quantification/processed/kfolds', n_processes=n, param_grid=param_grid)
cHL1.save_results(save_path=base_path + '/Maps_data/cHL_1_MIBI/results_logreg', label_path=base_path + '/Maps_data/cHL_1_MIBI/quantification/processed/labels.csv')

cHL_CODEX = Tune_Eval(random_state = rs, model = 'logistic_regression')
cHL_CODEX.train_tune_evaluate(path=base_path + '/Maps_data/cHL_CODEX/quantification/processed/kfolds', n_processes=n, param_grid = param_grid)
cHL_CODEX.save_results(save_path=base_path + '/Maps_data/cHL_CODEX/results_logreg', label_path=base_path + '/Maps_data/cHL_CODEX/quantification/processed/labels.csv')

del cHL2, cHL1, cHL_CODEX

MRL_CODEX = Tune_Eval(random_state = rs, model = 'logistic_regression')
MRL_CODEX.train_tune_evaluate(path=base_path + '/MRL_CODEX/quantification/processed/kfolds', n_processes=n, param_grid = param_grid)
MRL_CODEX.save_results(save_path=base_path + '/MRL_CODEX/results_logreg', label_path=base_path + '/MRL_CODEX/quantification/processed/labels.csv')

TB = Tune_Eval(random_state = rs, model = 'logistic_regression')
TB.train_tune_evaluate(path=base_path + '/TB_MIBI_atlas/quantification/processed/TB/kfolds', n_processes=n, param_grid = param_grid)
TB.save_results(save_path=base_path + '/TB_MIBI_atlas/quantification/processed/TB/results_logreg', label_path=base_path + '/TB_MIBI_atlas/quantification/processed/TB/labels.csv')

sarc = Tune_Eval(random_state = rs, model = 'logistic_regression')
sarc.train_tune_evaluate(path=base_path + '/TB_MIBI_atlas/quantification/processed/sarc/kfolds', n_processes=n, param_grid = param_grid)
sarc.save_results(save_path=base_path + '/TB_MIBI_atlas/quantification/processed/sarc/results_logreg', label_path=base_path + '/TB_MIBI_atlas/quantification/processed/sarc/labels.csv')

del MRL_CODEX, TB, sarc

tonsil_codex = Tune_Eval(random_state = rs, model = 'logistic_regression')
tonsil_codex.train_tune_evaluate(path=base_path + '/tonsil_CODEX/quantification/processed/kfolds', n_processes=n, param_grid = param_grid)
tonsil_codex.save_results(save_path=base_path + '/tonsil_CODEX/results_logreg', label_path=base_path + '/tonsil_CODEX/quantification/processed/labels.csv')

tonsil_CODEX2 = Tune_Eval(random_state = rs, model = 'logistic_regression')
tonsil_CODEX2.train_tune_evaluate(path=base_path + '/tonsil_CODEX2/quantification/processed/kfolds', n_processes=n, param_grid = param_grid)
tonsil_CODEX2.save_results(save_path=base_path + '/tonsil_CODEX2/results_logreg', label_path=base_path + '/tonsil_CODEX2/quantification/processed/labels.csv')


del tonsil_codex, tonsil_CODEX2

lymphoma_CODEX = Tune_Eval(random_state = rs, model = 'logistic_regression')
lymphoma_CODEX.train_tune_evaluate(path=base_path + '/lymphoma_CODEX/quantification/processed/kfolds', n_processes=n, param_grid = param_grid)
lymphoma_CODEX.save_results(save_path=base_path + '/lymphoma_CODEX/results_logreg', label_path=base_path + '/lymphoma_CODEX/quantification/processed/labels.csv')

intestine_CODEX = Tune_Eval(random_state = rs, model = 'logistic_regression')
intestine_CODEX.train_tune_evaluate(path=base_path + '/datasets/intestine_CODEX/quantification/processed/kfolds', n_processes=n, param_grid = param_grid)
intestine_CODEX.save_results(save_path=base_path + '/intestine_CODEX/results_logreg', label_path=base_path + '/intestine_CODEX/quantification/processed/labels.csv')

del lymphoma_CODEX, intestine_CODEX

feto_maternal = Tune_Eval(random_state = rs, model = 'logistic_regression')
feto_maternal.train_tune_evaluate(path=base_path + '/feto_maternal/quantification/processed/kfolds', n_processes=n, param_grid = param_grid)
feto_maternal.save_results(save_path=base_path + '/feto_maternal/results_logreg', label_path=base_path + '/feto_maternal/quantification/processed/labels.csv')

CRC_FFPE = Tune_Eval(random_state = rs, model = 'logistic_regression')
CRC_FFPE.train_tune_evaluate(path=base_path + '/CRC_FFPE/quantification/processed/kfolds', n_processes=n, param_grid = param_grid)
CRC_FFPE.save_results(save_path=base_path + '/CRC_FFPE/results_logreg', label_path=base_path + '/CRC_FFPE/quantification/processed/labels.csv')

BE_tonsil = Tune_Eval(random_state = rs, model = 'logistic_regression')
BE_tonsil.train_tune_evaluate(path=base_path + '/BA_CODEX/quantification/BE_tonsil/processed/kfolds', n_processes=n, param_grid = param_grid)
BE_tonsil.save_results(save_path=base_path + '/BA_CODEX/quantification/BE_tonsil/results_logreg', label_path=base_path + '/BA_CODEX/quantification/BE_tonsil/processed/labels.csv')

tonsil_training = Tune_Eval(random_state = rs, model = 'logistic_regression')
tonsil_training.train_tune_evaluate(path=base_path + '/BA_CODEX/quantification/tonsil_training/processed/kfolds', n_processes=n, param_grid = param_grid)
tonsil_training.save_results(save_path=base_path + '/BA_CODEX/quantification/tonsil_training/results_logreg', label_path=base_path + '/BA_CODEX/quantification/tonsil_training/processed/labels.csv')

del feto_maternal, CRC_FFPE, BE_tonsil, tonsil_training

AML = Tune_Eval(random_state = rs, model = 'logistic_regression')
AML.train_tune_evaluate(path=base_path + '/AML_bone_marrow/quantification/AML/processed/kfolds', n_processes=n, param_grid = param_grid)
AML.save_results(save_path=base_path + '/AML_bone_marrow/quantification/AML/results_logreg', label_path=base_path + '/AML_bone_marrow/quantification/AML/processed/labels.csv')

healthy_BM = Tune_Eval(random_state = rs, model = 'logistic_regression')
healthy_BM.train_tune_evaluate(path=base_path + '/AML_bone_marrow/quantification/healthy_BM/processed/kfolds', n_processes=n, param_grid = param_grid)
healthy_BM.save_results(save_path=base_path + '/AML_bone_marrow/quantification/healthy_BM/results_logreg', label_path=base_path + '/AML_bone_marrow/quantification/healthy_BM/processed/labels.csv')
