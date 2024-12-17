from nfstream import NFStreamer
import pandas as pd
import glob, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Lasso, LogisticRegression


def converting_pcap(src_path:str):
    print('Converting ...')
    my_streamer = NFStreamer(source=src_path,
                            # Disable L7 dissection for readability purpose.
                            n_dissections=0,
                            #idle_timeout= 600,
                            #active_timeout= 1800, 
                            accounting_mode= 3, 
                            statistical_analysis=True)
    df = my_streamer.to_pandas(columns_to_anonymize=[])
    print(df.shape)
    print(df.columns)
    print(df['src_ip'].value_counts())
    return(df)

def labeling (df, ips:list):
    print('Labeling ...')
    print(df.shape)
    if len(ips)==1 and '*' in ips:
        df['Label'] = 1
    else:
        df['Label']=np.where((df['src_ip'].isin(ips)), 1, 0)
    #df['Label']=np.where((df['src_ip'].isin(ips)) | (df['dst_ip'].isin(ips)), 1, 0)
    print(df[df['Label']==1].shape)
    #df_attack = df[df['Label']==1]
    #print(df_attack['src_ip'].value_counts())
    return df
def preprocessed (df, saved_path:str):
    print('preprocessed ...')
    print(df.columns)
    drop_cols = ['id', 'expiration_id', 'src_ip', 'src_mac', 'src_oui', 'src_port',
       'dst_ip', 'dst_mac', 'dst_oui', 'dst_port', 'protocol', 'ip_version',
       'vlan_id', 'tunnel_id']
    for col in df.columns:
        if col in drop_cols:
            #print('true')
            df = df.drop(col, axis=1)
    print('shape of created df:', df.shape)
    for col in df.columns:
        if "Unnamed" in col:
            df = df.drop(col, axis=1)
        if "ms" in col and not "duration" in col:
            df = df.drop(col, axis=1)
    print('df shape after droping some features:', df.shape)
    print(df.columns)
    print(df['Label'].value_counts())
    df.to_csv(saved_path)
    print('the csv file has been saved ...')
    return(df)

def merging_csv(csv_files_path):
    for i, files in enumerate(glob.glob(csv_files_path+'*.csv')):
        df = pd.read_csv(files, header=0, sep=',')
        if i != 0:
            merged_data = pd.concat([merged_data, df], ignore_index=True)
        else:
            merged_data = df
    print(merged_data.shape)
    merged_data.to_csv(csv_files_path+'merged_1-6.csv',columns=df.columns, index=False)

#Start Converting pcap file:
pcap_src_path = '/home/pegah/Codes/ssl-ids/pcap_files/'
pcap_file_name = 'gan_rce_redteam2_config_30_06-03-2024-07-43-33.pcap'
csv_destination = '/home/pegah/Codes/ssl-ids/Dataset/'
csv_name = pcap_file_name.split('.pcap')[0]+'.csv'
df = converting_pcap(pcap_src_path+pcap_file_name)
#attack_ip_list = ['147.32.84.165']
attack_ip_list = ['*']
df_new = labeling(df, attack_ip_list)
df_new_preprocessed = preprocessed(df_new, csv_destination+csv_name )
print(df_new_preprocessed)

