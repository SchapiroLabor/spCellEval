# %%

from models_classes.gridsearch import Tune_Eval
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

base_path = '/Volumes/Lukas_SSD/phenotyping_benchmark/datasets'#'/gpfs/bwfor/work/ws/hd_bm327-phenotyping_benchmark/phenotyping_benchmark/datasets'
rs = 20240925
n = 8
param_grid = {
    'n_estimators': [100, 250],
    'max_depth': [20, 25, 30, 40],
    'min_samples_split' : [2, 3],
    'min_samples_leaf' : [1, 2]
}

# df1 = pd.read_csv(base_path + '/Maps_data/cHL_2_MIBI/quantification/processed/kfolds/fold_1_test.csv')
# df2 = pd.read_csv(base_path + '/Maps_data/cHL_2_MIBI/quantification/processed/kfolds/fold_1_train.csv')
# df = pd.concat([df1, df2])
# y = df['encoded_phenotype']
# class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
# class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
# cHL2 = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
# cHL2.train_tune_evaluate(path=base_path + '/Maps_data/cHL_2_MIBI/quantification/processed/kfolds', param_grid=param_grid, n_processes=n)
# cHL2.save_results(save_path=base_path + '/Maps_data/cHL_2_MIBI/results_rfc', label_path=base_path + '/Maps_data/cHL_2_MIBI/quantification/processed/labels.csv')

# del cHL2

df1 = pd.read_csv(base_path + '/Maps_data/cHL_1_MIBI/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/Maps_data/cHL_1_MIBI/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
cHL1 = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
cHL1.train_tune_evaluate(path=base_path + '/Maps_data/cHL_1_MIBI/quantification/processed/kfolds', param_grid=param_grid)
cHL1.save_results(save_path=base_path + '/Maps_data/cHL_1_MIBI/results_rfc', label_path=base_path + '/Maps_data/cHL_1_MIBI/quantification/processed/labels.csv')

del cHL1

df1 = pd.read_csv(base_path + '/Maps_data/cHL_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/Maps_data/cHL_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
cHL_CODEX = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
cHL_CODEX.train_tune_evaluate(path=base_path + '/Maps_data/cHL_CODEX/quantification/processed/kfolds', param_grid=param_grid)
cHL_CODEX.save_results(save_path=base_path + '/Maps_data/cHL_CODEX/results_rfc', label_path=base_path + '/Maps_data/cHL_CODEX/quantification/processed/labels.csv')

del cHL_CODEX

df1 = pd.read_csv(base_path + '/MRL_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/MRL_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
MRL_CODEX = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
MRL_CODEX.train_tune_evaluate(path=base_path + '/MRL_CODEX/quantification/processed/kfolds', param_grid=param_grid)
MRL_CODEX.save_results(save_path=base_path + '/MRL_CODEX/results_rfc', label_path=base_path + '/MRL_CODEX/quantification/processed/labels.csv')

del MRL_CODEX

df1 = pd.read_csv(base_path + '/TB_MIBI_atlas/quantification/processed/sarc/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/TB_MIBI_atlas/quantification/processed/sarc/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
sarc = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
sarc.train_tune_evaluate(path=base_path + 'TB_MIBI_atlas/quantification/processed/sarc/kfolds', param_grid=param_grid)
sarc.save_results(save_path=base_path + '/TB_MIBI_atlas/results_rfc', label_path=base_path + '/TB_MIBI_atlas/quantification/processed/sarc/labels.csv')

del sarc

df1 = pd.read_csv(base_path + '/TB_MIBI_atlas/quantification/processed/TB/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/TB_MIBI_atlas/quantification/processed/TB/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
tb = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
tb.train_tune_evaluate(path=base_path + '/TB_MIBI_atlas/quantification/processed/TB/kfolds', param_grid=param_grid)
tb.save_results(save_path=base_path + '/TB_MIBI_atlas/results_rfc', label_path=base_path + '/TB_MIBI_atlas/quantification/processed/TB/labels.csv')

del tb

df1 = pd.read_csv(base_path + '/tonsil_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/tonsil_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
tonsil_CODEX = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
tonsil_CODEX.train_tune_evaluate(path=base_path + '/tonsil_CODEX/quantification/processed/kfolds', param_grid=param_grid)
tonsil_CODEX.save_results(save_path=base_path + '/tonsil_CODEX/results_rfc', label_path=base_path + '/tonsil_CODEX/quantification/processed/labels.csv')

del tonsil_CODEX

df1 = pd.read_csv(base_path + '/tonsil_CODEX2/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/tonsil_CODEX2/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
tonsil_CODEX2 = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
tonsil_CODEX2.train_tune_evaluate(path=base_path + '/tonsil_CODEX2/quantification/processed/kfolds', param_grid=param_grid)
tonsil_CODEX2.save_results(save_path=base_path + '/tonsil_CODEX2/results_rfc', label_path=base_path + '/tonsil_CODEX2/quantification/processed/labels.csv')

del tonsil_CODEX2

df1 = pd.read_csv(base_path + '/lymphoma_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/lymphoma_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
lymphoma_CODEX = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
lymphoma_CODEX.train_tune_evaluate(path=base_path + '/lymphoma_CODEX/quantification/processed/kfolds', param_grid=param_grid)
lymphoma_CODEX.save_results(save_path=base_path + '/lymphoma_CODEX/results_rfc', label_path=base_path + '/lymphoma_CODEX/quantification/processed/labels.csv')


del lymphoma_CODEX


df1 = pd.read_csv(base_path + '/intestine_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/intestine_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
intestine_CODEX = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
intestine_CODEX.train_tune_evaluate(path=base_path + '/intestine_CODEX/quantification/processed/kfolds', param_grid=param_grid)
intestine_CODEX.save_results(save_path=base_path + '/intestine_CODEX/results_rfc', label_path=base_path + '/intestine_CODEX/quantification/processed/labels.csv')

del intestine_CODEX

df1 = pd.read_csv(base_path + '/feto_maternal/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/feto_maternal/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
feto_maternal = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
feto_maternal.train_tune_evaluate(path=base_path + '/feto_maternal/quantification/processed/kfolds', param_grid=param_grid)
feto_maternal.save_results(save_path=base_path + '/feto_maternal/results_rfc', label_path=base_path + '/feto_maternal/quantification/processed/labels.csv')


del feto_maternal


df1 = pd.read_csv(base_path + '/CRC_FFPE/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/CRC_FFPE/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
CRC_FFPE = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
CRC_FFPE.train_tune_evaluate(path=base_path + '/CRC_FFPE/quantification/processed/kfolds', param_grid=param_grid)
CRC_FFPE.save_results(save_path=base_path + '/CRC_FFPE/results_rfc', label_path=base_path + '/CRC_FFPE/quantification/processed/labels.csv')

del CRC_FFPE

df1 = pd.read_csv(base_path + '/BA_CODEX/quantification/BE_tonsil/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/BA_CODEX/quantification/BE_tonsil/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
BE_tonsil = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
BE_tonsil.train_tune_evaluate(path=base_path + '/BA_CODEX/quantification/BE_tonsil/processed/kfolds', param_grid=param_grid)
BE_tonsil.save_results(save_path=base_path + '/BA_CODEX/results_rfc', label_path=base_path + '/BA_CODEX/quantification/BE_tonsil/processed/labels.csv')

del BE_tonsil

df1 = pd.read_csv(base_path + '/BA_CODEX/quantification/tonsil_training/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/BA_CODEX/quantification/tonsil_training/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
tonsil_training = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
tonsil_training.train_tune_evaluate(path=base_path + '/BA_CODEX/quantification/tonsil_training/processed/kfolds', param_grid=param_grid)
tonsil_training.save_results(save_path=base_path + '/BA_CODEX/results_rfc', label_path=base_path + '/BA_CODEX/quantification/tonsil_training/processed/labels.csv')

del tonsil_training

df1 = pd.read_csv(base_path + '/AML_bone_marrow/quantification/AML/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/AML_bone_marrow/quantification/AML/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
AML = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
AML.train_tune_evaluate(path=base_path + '/AML_bone_marrow/quantification/AML/processed/kfolds', param_grid=param_grid)
AML.save_results(save_path=base_path + '/AML_bone_marrow/results_rfc', label_path=base_path + '/AML_bone_marrow/quantification/AML/processed/labels.csv')

del AML

df1 = pd.read_csv(base_path + '/AML_bone_marrow/quantification/healthy_BM/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv(base_path + '/AML_bone_marrow/quantification/healthy_BM/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
healthy_BM = Tune_Eval(class_weight=class_weight_dict, random_state=rs, model='random_forest')
healthy_BM.train_tune_evaluate(path=base_path + '/AML_bone_marrow/quantification/healthy_BM/processed/kfolds', param_grid=param_grid)
healthy_BM.save_results(save_path=base_path + '/AML_bone_marrow/results_rfc', label_path=base_path + '/AML_bone_marrow/quantification/healthy_BM/processed/labels.csv')

del healthy_BM