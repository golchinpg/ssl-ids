import numpy as np
import pandas as pd
import tensorflow as tf
import glob, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input, Lambda
from sklearn.preprocessing import MinMaxScaler
#from imblearn.over_sampling import SMOTE
#from keras.optimizers.legacy import Adam
from keras.optimizers import Adam
from keras.losses import cosine_similarity
from sklearn.metrics import roc_auc_score

#dataset_path = "/Users/pegah/Desktop/KOM/Datasets/preprocessed_csv/"
dataset_path = "/home/pegah/Codes/ssl-ids/Dataset/"


class generalization_analysis:
    def __init__(self, dataset):
        self.dataset = dataset

    def Preprocessing(self):
        print("initial amount of columns:", self.dataset.shape)
        for col in self.dataset.columns:
            if "Unnamed" in col:
                self.dataset = self.dataset.drop(col, axis=1)
            if "ms" in col and not "duration" in col:
                self.dataset = self.dataset.drop(col, axis=1)
        # print('columns after preprocessing:',self.dataset.columns)
        # print('shape of new dataset:', self.dataset.shape)
        # print('test:############',self.dataset['Label'].shape)
        return self.dataset
    
    def make_separation(self, df):
        df_attack = df[df['Label']==1]
        print(df_attack.shape)
        return(df_attack)
    
    def Split_dataset(self, X, y, test_size):
        #X = df.iloc[:, :-1]
        #y = df["Label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print("shape of X_train and X_test:", X_train.shape, y_train.shape)
        return (X_train, y_train, X_test, y_test)

    def feature_importance(self, num_features, X_train, y_train):
        # Random Forest feature importance
        def normalize_dictionary(selected_dict):
            # Find the minimum and maximum values
            min_value = min(selected_dict.values())
            max_value = max(selected_dict.values())
            # Normalize the values to the range [0, 1]
            normalized_dict = {
                key: (value - min_value) / (max_value - min_value)
                for key, value in selected_dict.items()
            }
            return normalized_dict

        def RF_feature_importance(X_train, y_train):
            print("*********** Random Forest feature extraction ******")
            new_dict = {}
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_model.fit(X_train, y_train)
            importances = rf_model.feature_importances_
            indices = importances.argsort()[::-1]
            features = X_train.columns
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)
            for j, feature_imp in enumerate(features):
                if feature_imp in new_dict.keys():
                    new_dict[feature_imp] += importances[j]
                else:
                    new_dict[feature_imp] = importances[j]
            rf_features = {
                k: v
                for k, v in sorted(
                    new_dict.items(), key=lambda item: item[1], reverse=True
                )
            }
            rf_features = normalize_dictionary(rf_features)
            print("final rf features:", rf_features)
            return (rf_features, rf_model)

        # Logistic Regression feature importance
        def lr_feature_importance(X_train, y_train):  # , X_val, y_val
            print("********* Logistic regression feature extraction ******")
            lasso_selection = {}
            lr_model = SelectFromModel(
                LogisticRegression(C=10, penalty="l1", solver="liblinear")
            )
            lr_model.fit(X_train, y_train)  # , classes=numpy.unique(y_train))
            selected_feat = X_train.columns[(lr_model.get_support())]
            importances = abs(lr_model.estimator_.coef_[0])
            features = X_train.columns
            for j, feature_imp in enumerate(features):
                if feature_imp in lasso_selection.keys():
                    lasso_selection[feature_imp] += importances[j]
                else:
                    lasso_selection[feature_imp] = importances[j]
            final_lasso = {
                k: v
                for k, v in sorted(
                    lasso_selection.items(), key=lambda item: item[1], reverse=True
                )
            }
            final_lasso = normalize_dictionary(final_lasso)
            print("final dictionary", final_lasso)
            return (final_lasso, lr_model)
            #

        def svc_feature_importance(X_train, y_train):  # , X_val, y_val
            print("********* SVC feature extraction ******")
            svc_selection = {}
            svc_model = SelectFromModel(
                LinearSVC(C=10, penalty="l1", dual=False, max_iter=10000)
            )
            svc_model.fit(X_train, y_train)  # , classes=numpy.unique(y_train))
            selected_feat = X_train.columns[(svc_model.get_support())]
            importances = abs(svc_model.estimator_.coef_[0])
            features = X_train.columns
            for j, feature_imp in enumerate(features):
                if feature_imp in svc_selection.keys():
                    svc_selection[feature_imp] += importances[j]
                else:
                    svc_selection[feature_imp] = importances[j]
            final_svc = {
                k: v
                for k, v in sorted(
                    svc_selection.items(), key=lambda item: item[1], reverse=True
                )
            }
            final_svc = normalize_dictionary(final_svc)
            print("final svc dictionary:", final_svc)
            return (final_svc, svc_model)

        def select_final_features(feature_dic1, feature_dic2, feature_dic3):
            final_selection = {}
            for key_feature, val_feature in feature_dic1.items():
                if key_feature in feature_dic2.keys():
                    temp_val1 = feature_dic2[key_feature]
                if key_feature in feature_dic3.keys():
                    temp_val2 = feature_dic3[key_feature]
                final_selection[key_feature] = val_feature + temp_val1 + temp_val2
            for key_feature in feature_dic2.keys():
                if not key_feature in list(final_selection.keys()):
                    if key_feature in feature_dic3.keys():
                        final_selection[key_feature] = (
                            feature_dic2[key_feature] + feature_dic3[key_feature]
                        )
                    else:
                        final_selection[key_feature] = feature_dic2[key_feature]
            for key_feature in feature_dic3.keys():
                if not key_feature in list(final_selection.keys()):
                    final_selection[key_feature] = feature_dic3[key_feature]
            sorted_final_selection = dict(
                sorted(final_selection.items(), key=lambda item: item[1], reverse=True)
            )

            print("####### FINAL SELECTION:\n", sorted_final_selection)
            print(len(list(sorted_final_selection.keys())))
            return sorted_final_selection

        def select_ntop(n, features_dict):
            # select ntop of a dictionary of dictionary
            # final_features = []
            feature_list = list(features_dict.keys())
            return feature_list[:n]

        rf_FS, rf_model = RF_feature_importance(X_train, y_train)
        lr_FS, lr_model = lr_feature_importance(X_train, y_train)
        svc_FS, svc_model = svc_feature_importance(X_train, y_train)
        print("############### EXTRACT THE FINAL FEATURE SPACE ##############\n")
        final_fs = select_final_features(rf_FS, lr_FS, svc_FS)
        selection_ntop = select_ntop(num_features, final_fs)
        print(
            "######################\n       Final N top       \n###################### "
        )
        print(selection_ntop)

        return (selection_ntop, rf_model, lr_model, svc_model)
        """
        for name in model_name:
            print(name)
            if name == 'rf':
                rf_FS = RF_feature_importance(X_train, y_train)
                rf_ntop = select_ntop(num_features, rf_FS)
                return(rf_ntop)
            if name == 'lr':
                lr_FS = lr_feature_importance(X_train, y_train)
                lr_ntop = select_ntop(num_features, lr_FS)
                return(lr_ntop)
            if name == 'svc':
                svc_FS = svc_feature_importance(X_train, y_train)
                svc_ntop = select_ntop(num_features, svc_FS)
                return(svc_ntop)
        """

    def training(self, X_train, X_test, y_train, y_test):
        # X_train = X_train[final_features]
        # X_test = X_test[final_features]
        # RANDOM FOREST TRAINING/EVALUATING
        rf = RandomForestClassifier(n_estimators=20)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_pred_prob_rf = rf.predict_proba(X_test)[:,1]
        report_rf = classification_report(y_test, y_pred_rf)
        print("##### RF test Evaluation ########\n", report_rf)
        auroc_rf = roc_auc_score(y_test, y_pred_prob_rf)
        print("AUROC: {:.4f}".format(auroc_rf))
        """
        # LOGISTIC REGRESSION TRAINING/EVALUATION
        log_reg_classifier = LogisticRegression(random_state=42)
        log_reg_classifier.fit(X_train, y_train)
        y_pred_lr = log_reg_classifier.predict(X_test)
        y_pred_prob_lr = log_reg_classifier.predict_proba(X_test)[:, 1]
        report_lr = classification_report(y_test, y_pred_lr)
        print("##### LR test Evaluation ########\n", report_lr)
        auroc_lr = roc_auc_score(y_test, y_pred_prob_lr)
        print("AUROC: {:.4f}".format(auroc_lr))
        
        model = Sequential()
        model.add(Dense(20, input_dim=X_train.shape[1], activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        model.fit(
            X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test)
        )
        y_pred_mlp = (model.predict(X_test) > 0.5).astype("int32")
        y_pred_prob_mlp = model.predict(X_test)#[:, 1]
        report_mlp = classification_report(y_test, y_pred_mlp)
        print("##### MLP test Evaluation ########\n", report_mlp)
        auroc_mlp = roc_auc_score(y_test, y_pred_prob_mlp)
        print("AUROC: {:.4f}".format(auroc_mlp))
        """
        return (rf)#, model)#, log_reg_classifier, model)


class evaluating_feature_selection:
    def __init__(self, dataset, final_feature_set, test_dataset, day_name):
        self.dataset = dataset
        self.fainl_feature_set = final_feature_set
        self.test_dataset = test_dataset
        self.day_name = day_name

    def evaluating_with_different_day_attack(
        self, X, y, rf_training_model, lr_training_model, mlp_training_model
    ):
        #X = self.test_dataset.iloc[:, :-1]
        #y = self.test_dataset["Label"]
        #X_selected = X[self.fainl_feature_set]
        #y_pred_rf = rf_training_model.predict(X_selected)
        y_pred_rf = rf_training_model.predict(X)
        report_rf = classification_report(y, y_pred_rf)
        print(
            "##### RF test Evaluation on Dataset " + self.day_name + " ########\n",
            report_rf,
        )
        #y_pred_lr = lr_training_model.predict(X_selected)
        y_pred_lr = lr_training_model.predict(X)
        report_lr = classification_report(y, y_pred_lr)
        print(
            "##### LR test Evaluation on Dataset " + self.day_name + " ########\n",
            report_lr,
        )
        #y_pred_mlp = (mlp_training_model.predict(X_selected) > 0.5).astype("int32")
        y_pred_mlp = (mlp_training_model.predict(X) > 0.5).astype("int32")
        # y_pred_mlp = mlp_training_model.predict(X_selected)
        report_mlp = classification_report(y, y_pred_mlp)
        print(
            "##### MLP test Evaluation on Dataset " + self.day_name + " ########\n",
            report_mlp,
        )

class autoencoder_model:
    def __init__(self) -> None:
        pass

    def autoencoder_training(self, X_train, X_test, y_train, y_test):
        
        input_layer = Input(shape=(45,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.2)(encoded)  # Add dropout

        encoded = Dense(64, activation='relu')(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.2)(encoded)  # Add dropout

        encoded = Dense(32, activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.2)(encoded)  # Add dropout

        encoded = Dense(32, activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)

        decoded = Dense(64, activation='relu')(encoded)
        #decoded = BatchNormalization()(decoded)
        #decoded = Dropout(0.2)(decoded)  # Add dropout

        decoded = Dense(128, activation='relu')(decoded)
        #decoded = BatchNormalization()(decoded)
        #decoded = Dropout(0.2)(decoded)  # Add dropout
        decoded = Dense(64, activation='relu')(decoded)

        decoded = Dense(45, activation='sigmoid')(decoded)


        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')#, loss='mean_squared_error')
        autoencoder.fit(X_train, X_train, epochs=40, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

        # Create a separate model to extract the learned representations
        encoder = Model(inputs=input_layer, outputs=encoded)

        # Extract representations for training and testing data
        X_train_encoded = encoder.predict(X_train)
        X_test_encoded = encoder.predict(X_test)
        return(encoder, X_train_encoded, X_test_encoded)
        """
        # Define and train an MLP classifier using the learned representations
        classifier = Sequential()
        classifier.add(Dense(64, activation='relu', input_dim=32))
        classifier.add(Dense(1, activation='sigmoid'))  # Binary classification, adjust for your problem

        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the classifier on the learned representations
        classifier.fit(X_train_encoded, y_train, epochs=50, batch_size=32, validation_data=(X_test_encoded, y_test))
        """

### Running few shot:
name = 'merged_1-6.csv'
#name = 'allattack_mondaybenign.csv'
file_path = dataset_path+ name
training_df = pd.read_csv(file_path, header=0, sep=",")
cols = training_df.columns
generalization = generalization_analysis(training_df)
preprocessed_dataset = generalization.Preprocessing()
X = preprocessed_dataset.iloc[:, :-1]
y = preprocessed_dataset["Label"]
X_use, y_use, X_test, y_test = generalization.Split_dataset(
            X, y, 0.4
        )
training_portion = [0.00005,0.0005, 0.001, 0.01,0.5,0.9]# 0.00005, 0.0001, 0.0003,
for portion in training_portion:
    print('############### train with '+str(portion))
    X_ret, y_ret, X_train, y_train = generalization.Split_dataset(
                X_use, y_use , portion
            )
    print('portion is ', float(len(X_use))*portion)
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    #print('number of attacks:', y_train[y_train['Label']==1].shape)
    #print('number of benigns:', y_train[y_train['Label']==0].shape)
    rf_model = generalization.training(
        X_train, X_test, y_train, y_test
    ) 



"""ok            


### running 
merged_file = dataset_path+'allattack_mondaybenign.csv'
#merged_file = dataset_path+'merged_1-6.csv'
#merged_file = dataset_path+'combined_normal.csv'
dataset = pd.read_csv(merged_file, header=0, sep=",")
cols = dataset.columns
generalization = generalization_analysis(dataset)
preprocessed_dataset = generalization.Preprocessing()
print('number of attacks:', preprocessed_dataset[preprocessed_dataset['Label']==1].shape)
print('number of benigns:', preprocessed_dataset[preprocessed_dataset['Label']==0].shape)

X_train, y_train, X_test, y_test = generalization.Split_dataset(
            preprocessed_dataset
        )

final_FS = ['src2dst_fin_packets', 'dst2src_rst_packets', 'src2dst_syn_packets', 'bidirectional_urg_packets', 
                     'src2dst_urg_packets', 'src2dst_max_ps', 'src2dst_bytes', 'src2dst_mean_ps', 'dst2src_mean_ps',
                       'src2dst_cwr_packets', 'dst2src_syn_packets', 'dst2src_ece_packets', 'dst2src_fin_packets',
                         'bidirectional_max_ps', 'dst2src_max_ps', 'bidirectional_bytes', 'bidirectional_syn_packets', 
                         'bidirectional_mean_ps', 'bidirectional_fin_packets', 'bidirectional_rst_packets']

"""
"""
scaler = MinMaxScaler()
preprocessed_dataset_scaled = scaler.fit_transform(preprocessed_dataset)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print('test', preprocessed_dataset_scaled.shape)
#Let's try self-supervised learning
#selfsupervised_method = contrastive_learning()
print(preprocessed_dataset_scaled.shape[1])
#print(X_train.columns)
df = pd.DataFrame(preprocessed_dataset_scaled, columns=preprocessed_dataset.columns, index=preprocessed_dataset.index)
print('test2',df.shape)
selfsupervised_method.contrastive_model(df,X_train_scaled, y_train, df.shape[1]-1)
"""
"""ok
#reduce feature size using autoencoder
representation = autoencoder_model()
encoder, X_train_encoded, X_test_encoded = representation.autoencoder_training(X_train_scaled, X_test_scaled, y_train, y_test)
X_train_encoded_df = pd.DataFrame(X_train_encoded)
X_test_encoded_df = pd.DataFrame(X_test_encoded)

print(X_train_encoded.shape, X_test_encoded.shape)
#print(X_train_encoded.columns)
rf_model , lr_model, mlp_model = generalization.training([], X_train_encoded_df, X_test_encoded_df, y_train, y_test)
"""
"""okok
#checking on a new dataset
test_dataset_names = ['ISCX-SlowDoS_1.csv','allattack_mondaybenign.csv', 'botnet43_truncated.csv',
                       'botnet_iscx_training.csv', 'merged_1-6.csv']
#'ISCX-SlowDoS_1.csv',
#df_unknown0 = pd.read_csv(dataset_path+'merged_1-6.csv', sep=',', header=0)
    
    #extracting feature selection using hybrid feature importance methods
    #final_FS, _, _, _ = generalization.feature_importance(20, X_train, y_train)
X_train_new = X_train#[final_FS]
X_test_new = X_test#[final_FS]
rf_model, mlp_model = generalization.training(
    final_FS, X_train_new, X_test_new, y_train, y_test
) #, lr_model, mlp_model
for name in test_dataset_names:
    print(name)
    df_unknown = pd.read_csv(dataset_path+name, sep=',', header=0)

    #df_unknown_fs = df_unknown#[final_FS]
    generalization_unknown = generalization_analysis(df_unknown)
    df_unknown_fs = generalization_unknown.Preprocessing()
    X_test_unknown, y_test_unknown = df_unknown_fs.iloc[:, :-1], df_unknown_fs['Label']
    print('*************** test data: '+name+' *********************')
    y_pred_rf = rf_model.predict(X_test_unknown)
    y_pred_prob_rf = rf_model.predict_proba(X_test_unknown)[:, 1]
    print('**** RF classification report on new dataset with the selected features:\n')
    print(classification_report(y_test_unknown, y_pred_rf))
    auroc_rf_unk = roc_auc_score(y_test_unknown, y_pred_prob_rf)
    print("AUROC of Unknown dataset: {:.4f}".format(auroc_rf_unk))
    

    y_pred_mlp = (mlp_model.predict(X_test_unknown) > 0.5).astype("int32")
    y_pred_prob_mlp = mlp_model.predict(X_test_unknown)#[:, 1]
    print('**** MLP classification report on new dataset with the selected features:\n')
    print(classification_report(y_test_unknown, y_pred_mlp))
    auroc_mlp_unk = roc_auc_score(y_test_unknown, y_pred_prob_mlp)
    print("AUROC of Unknown dataset: {:.4f}".format(auroc_mlp_unk))

"""
"""
    y_pred_lr = lr_model.predict(X_test_unknown)
    y_pred_prob_lr = lr_model.predict_proba(X_test_unknown)[:, 1]
    print('**** LR classification report on new dataset with the selected features:\n')
    print(classification_report(y_test_unknown, y_pred_lr))
    auroc_lr_unk = roc_auc_score(y_test_unknown, y_pred_prob_lr)
    print("AUROC of Unknown dataset: {:.4f}".format(auroc_lr_unk))

"""

    



"""
#ok
test_days_name = ["Tuesday_preprocessed.csv", "Wednesday_preprocessed.csv", "Thursday_preprocessed.csv", "Friday_preprocessed.csv", "Monday_preprocessed.csv"]
for day_name in test_days_name:
    day = day_name.split('_')[0]
    print("**********  Evaluating generalization for " + day_name)
    test_dataset = pd.read_csv(
        dataset_path + day_name , header=0, sep=","
    )
    testing = generalization_analysis(test_dataset)
    preprocessed_test_dataset = testing.Preprocessing()
    X_preprocessed = preprocessed_test_dataset.iloc[:, :-1]
    y_preprocessed = preprocessed_test_dataset["Label"]
    X_preprocessed_scaled = scaler.fit_transform(X_preprocessed)
    X_preprocessed_encoded = encoder.predict(X_preprocessed_scaled)

    evaluations = evaluating_feature_selection(
        [], preprocessed_test_dataset, day_name
    )
    evaluations.evaluating_with_different_day_attack(
        X_preprocessed_encoded, y_preprocessed, rf_model, lr_model, mlp_model
    )
"""
"""
for i, ds in enumerate(glob.glob(dataset_path + "*.csv")):
    print("dataset:", ds)
    if ("Monday" in ds): #or ("Monday" in ds) or ("Wednesday" in ds):
        dataset = pd.read_csv(ds, header=0, sep=",")
        generalization = generalization_analysis(dataset)
        preprocessed_dataset = generalization.Preprocessing()
        print(preprocessed_dataset.shape)
        #preprocessed_dataset.to_csv(dataset_path+'Monday_preprocessed_merged.csv')
        #df_attack = generalization.make_separation(preprocessed_dataset)

    
        print(preprocessed_dataset.shape)
        X_train, y_train, X_test, y_test = generalization.Split_dataset(
            preprocessed_dataset
        )
        # print(X_train.columns)
        final_FS, _, _, _ = generalization.feature_importance(20, X_train, y_train)
        # final_FS = ['src2dst_fin_packets','src2dst_max_ps','src2dst_mean_ps','src2dst_min_ps','dst2src_rst_packets', 'bidirectional_min_ps','src2dst_bytes', 'bidirectional_rst_packets', 'src2dst_rst_packets', 'bidirectional_max_ps','dst2src_fin_packets', 'bidirectional_bytes', 'dst2src_syn_packets', 'src2dst_syn_packets','bidirectional_syn_packets', 'dst2src_psh_packets', 'dst2src_ack_packets', 'bidirectional_mean_ps','dst2src_mean_ps', 'bidirectional_cwr_packets']
        X_train_new = X_train[final_FS]
        X_test_new = X_test[final_FS]
        rf_model, lr_model, mlp_model = generalization.training(
            final_FS, X_train_new, X_test_new, y_train, y_test
        )
        #
        test_days_name = ["Tuesday", "Wednesday", "Thursday", "Friday", "Monday"]
        for day_name in test_days_name:
            if not day_name in ds.split("/")[-1].split(".")[0]:
                print("**********  Evaluating generalization for " + day_name)
                test_dataset = pd.read_csv(
                    dataset_path + day_name + "_preprocessed.csv", header=0, sep=","
                )
                testing = generalization_analysis(test_dataset)
                preprocessed_test_dataset = testing.Preprocessing()
                evaluations = evaluating_feature_selection(
                    final_FS, preprocessed_test_dataset, day_name
                )
                evaluations.evaluating_with_different_day_attack(
                    rf_model, lr_model, mlp_model
                )
        
        #rf_features_ntop = generalization.feature_importance(20, ['rf'], X_train, y_train)
        #print('RF feature importance:', rf_features_ntop)
        #lr_features_ntop = generalization.feature_importance(20, ['lr'], X_train, y_train)
        #print('LR feature importance:', lr_features_ntop)
        #svc_features_ntop = generalization.feature_importance(20, ['svc'], X_train, y_train)
        #print('SVC feature importance:', svc_features_ntop)
        
    else:
        continue
"""