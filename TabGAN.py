from ctgan import CTGAN
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer



def read_data(dataset_path:str):
    merged_file = dataset_path+'Monday_preprocessed.csv' #only benign dataset

    df = pd.read_csv(merged_file, header=0, sep=",")
    for col in df.columns:
                if "Unnamed" in col:
                    df = df.drop(col, axis=1)
                if "ms" in col and not "duration" in col:
                    df = df.drop(col, axis=1)

    #scaler = MinMaxScaler()
    #preprocessed_dataset_scaled = scaler.fit_transform(df)

    #df_normalized = pd.DataFrame(preprocessed_dataset_scaled, columns=df.columns, index=df.index)
    #data, target = df_normalized.iloc[:, :-1], df_normalized['Label']
    data, target = df.iloc[:, :-1], df['Label']
    data = data[:3000]
    print(data.shape)
    return data, target

path = "/Users/pegah/Desktop/KOM/Datasets/preprocessed_csv/"
data, target = read_data(path)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
metadata.visualize()
#print(metadata)
non_discrete_features = ['bidirectional_duration_ms','src2dst_duration_ms', 'dst2src_duration_ms','bidirectional_mean_ps', 'bidirectional_stddev_ps', 'src2dst_mean_ps', 'src2dst_stddev_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps']
discreate_features = [col for col in data.columns if not col in non_discrete_features ]
print('Start training ...')
synthesizer = CTGANSynthesizer(metadata,epochs = 500, embedding_dim=32)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=200)
print(synthetic_data.head())
print('Start evaluation ....')
quality_report = evaluate_quality(
    data,
    synthetic_data, metadata
)