from models_classes.gridsearch import GridSearcher
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

rs = 20240925

param_grid = {
    'n_estimators': [100, 250, 500, 1000],
    'max_depth': [10, 20, 25, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

df1 = pd.read_csv('/datasets/Maps_data/cHL_2_MIBI/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/cHL_2_MIBI/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

cHL2 = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
cHL2.train_tune_evaluate(path='/datasets/Maps_data/cHL_2_MIBI/quantification/processed/kfolds', param_grid=param_grid)
cHL2.save_results(save_path='/datasets/Maps_data/cHL_2_MIBI/results_rfc_gscv', label_path='/datasets/Maps_data/cHL_2_MIBI/quantification/processed/labels.csv')


df1 = pd.read_csv('/datasets/Maps_data/cHL_1_MIBI/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/cHL_1_MIBI/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

cHL1 = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
cHL1.train_tune_evaluate(path='/datasets/Maps_data/cHL_1_MIBI/quantification/processed/kfolds', param_grid=param_grid)
cHL1.save_results(save_path='/datasets/Maps_data/cHL_1_MIBI/results_rfc_gscv', label_path='/datasets/Maps_data/cHL_1_MIBI/quantification/processed/labels.csv')


df1 = pd.read_csv('/datasets/Maps_data/cHL_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/cHL_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

cHL_CODEX = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
cHL_CODEX.train_tune_evaluate(path='/datasets/Maps_data/cHL_CODEX/quantification/processed/kfolds', param_grid=param_grid)
cHL_CODEX.save_results(save_path='/datasets/Maps_data/cHL_CODEX/results_rfc_gscv', label_path='/datasets/Maps_data/cHL_CODEX/quantification/processed/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/MRL_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/MRL_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}


MRL_CODEX = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
MRL_CODEX.train_tune_evaluate(path='/datasets/Maps_data/MRL_CODEX/quantification/processed/kfolds', param_grid=param_grid)
MRL_CODEX.save_results(save_path='/datasets/Maps_data/MRL_CODEX/results_rfc_gscv', label_path='/datasets/Maps_data/MRL_CODEX/quantification/processed/labels.csv')


df1 = pd.read_csv('/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/sarc/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/sarc/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

sarc = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
sarc.train_tune_evaluate(path='/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/sarc/kfolds', param_grid=param_grid)
sarc.save_results(save_path='/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/sarc/results_rfc_gscv', label_path='/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/sarc/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/TB/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/TB/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

tb = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
tb.train_tune_evaluate(path='/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/TB/kfolds', param_grid=param_grid)
tb.save_results(save_path='/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/TB/results_rfc_gscv', label_path='/datasets/Maps_data/TB_MIBI_atlas/quantification/processed/TB/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/tonsil_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/tonsil_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

tonsil_CODEX = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
tonsil_CODEX.train_tune_evaluate(path='/datasets/Maps_data/tonsil_CODEX/quantification/processed/kfolds', param_grid=param_grid)
tonsil_CODEX.save_results(save_path='/datasets/Maps_data/tonsil_CODEX/results_rfc_gscv', label_path='/datasets/Maps_data/tonsil_CODEX/quantification/processed/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/tonsil_CODEX2/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/tonsil_CODEX2/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

tonsil_CODEX2 = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
tonsil_CODEX2.train_tune_evaluate(path='/datasets/Maps_data/tonsil_CODEX2/quantification/processed/kfolds', param_grid=param_grid)
tonsil_CODEX2.save_results(save_path='/datasets/Maps_data/tonsil_CODEX2/results_rfc_gscv', label_path='/datasets/Maps_data/tonsil_CODEX2/quantification/processed/labels.csv')

del cHL1, cHL2, cHL_CODEX, MRL_CODEX, sarc, tb, tonsil_CODEX, tonsil_CODEX2

df1 = pd.read_csv('/datasets/Maps_data/lymphoma_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/lymphoma_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

lymphoma_CODEX = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
lymphoma_CODEX.train_tune_evaluate(path='/datasets/Maps_data/lymphoma_CODEX/quantification/processed/kfolds', param_grid=param_grid)
lymphoma_CODEX.save_results(save_path='/datasets/Maps_data/lymphoma_CODEX/results_rfc_gscv', label_path='/datasets/Maps_data/lymphoma_CODEX/quantification/processed/labels.csv')

del lymphoma_CODEX

df1 = pd.read_csv('/datasets/Maps_data/intestine_CODEX/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/intestine_CODEX/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

intestine_CODEX = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
intestine_CODEX.train_tune_evaluate(path='/datasets/Maps_data/intestine_CODEX/quantification/processed/kfolds', param_grid=param_grid)
intestine_CODEX.save_results(save_path='/datasets/Maps_data/intestine_CODEX/results_rfc_gscv', label_path='/datasets/Maps_data/intestine_CODEX/quantification/processed/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/feto_maternal/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/feto_maternal/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

feto_maternal = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
feto_maternal.train_tune_evaluate(path='/datasets/Maps_data/feto_maternal/quantification/processed/kfolds', param_grid=param_grid)
feto_maternal.save_results(save_path='/datasets/Maps_data/feto_maternal/results_rfc_gscv', label_path='/datasets/Maps_data/feto_maternal/quantification/processed/labels.csv')

del intestine_CODEX, feto_maternal

df1 = pd.read_csv('/datasets/Maps_data/CRC_FFPE/quantification/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/CRC_FFPE/quantification/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

CRC_FFPE = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
CRC_FFPE.train_tune_evaluate(path='/datasets/Maps_data/CRC_FFPE/quantification/processed/kfolds', param_grid=param_grid)
CRC_FFPE.save_results(save_path='/datasets/Maps_data/CRC_FFPE/results_rfc_gscv', label_path='/datasets/Maps_data/CRC_FFPE/quantification/processed/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/BA_CODEX/quantification/BE_tonsil/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/BA_CODEX/quantification/BE_tonsil/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

BE_tonsil = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
BE_tonsil.train_tune_evaluate(path='/datasets/Maps_data/BA_CODEX/quantification/BE_tonsil/processed/kfolds', param_grid=param_grid)
BE_tonsil.save_results(save_path='/datasets/Maps_data/BA_CODEX/quantification/BE_tonsil/results_rfc_gscv', label_path='/datasets/Maps_data/BA_CODEX/quantification/BE_tonsil/processed/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/BA_CODEX/quantification/tonsil_training/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/BA_CODEX/quantification/tonsil_training/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

tonsil_training = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
tonsil_training.train_tune_evaluate(path='/datasets/Maps_data/BA_CODEX/quantification/tonsil_training/processed/kfolds', param_grid=param_grid)
tonsil_training.save_results(save_path='/datasets/Maps_data/BA_CODEX/quantification/tonsil_training/results_rfc_gscv', label_path='/datasets/Maps_data/BA_CODEX/quantification/tonsil_training/processed/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/AML_bone_marrow/quantification/AML/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/AML_bone_marrow/quantification/AML/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

AML = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
AML.train_tune_evaluate(path='/datasets/Maps_data/AML_bone_marrow/quantification/AML/processed/kfolds', param_grid=param_grid)
AML.save_results(save_path='/datasets/Maps_data/AML_bone_marrow/quantification/AML/results_rfc_gscv', label_path='/datasets/Maps_data/AML_bone_marrow/quantification/AML/processed/labels.csv')

df1 = pd.read_csv('/datasets/Maps_data/AML_bone_marrow/quantification/healthy_BM/processed/kfolds/fold_1_test.csv')
df2 = pd.read_csv('/datasets/Maps_data/AML_bone_marrow/quantification/healthy_BM/processed/kfolds/fold_1_train.csv')
df = pd.concat([df1, df2])
y = df['encoded_phenotype']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

healthy_BM = GridSearcher(class_weight=class_weight_dict, random_state=rs, n_jobs=-1, model='random_forest')
healthy_BM.train_tune_evaluate(path='/datasets/Maps_data/AML_bone_marrow/quantification/healthy_BM/processed/kfolds', param_grid=param_grid)
healthy_BM.save_results(save_path='/datasets/Maps_data/AML_bone_marrow/quantification/healthy_BM/results_rfc_gscv', label_path='/datasets/Maps_data/AML_bone_marrow/quantification/healthy_BM/processed/labels.csv')
