import pandas as pd
import ipaddress as ip

def remove_dash(x):
    if x == '-':
        return 0
    return x

def transform_labels(df):
    label = []
    detailed_label = []

    for string in df['label']:
        splits = string.split()
        if len(splits) > 1:
            label.append(splits[1])        
            if(splits[2] not in ['-','(EMPTY)']):
                detailed_label.append(splits[2])
            else:
                detailed_label.append("Benign")
        else:
            label.append(splits[0])
            if splits[0] in ['Benign','benign']:
                detailed_label.append("Benign")
            else:
                detailed_label.append("Malicious")

    df['label'] = label
    df['detailed-label'] = detailed_label
    return df

def history_len(df):
    history_len = []
    for h in df['history']:
        l = 0
        for char in set(h) :
            counts=h.count(char)
            l = l + 10**(counts - 1)
        history_len.append(l)
    df.loc[:,'history_len'] = history_len
    return df

def basic_preprocess(row):
    row = row.drop(['ts','uid','local_orig','local_resp','history','conn_state'], axis=1)

    d = {'-':0,'dns':1,'http':2,'dhcp':3,'irc':4,'ssl':5,'ssh':6}
    row.loc[:,'service'] = row['service'].apply(lambda x: d[x])   
    
    # Translate IPs to a number
    row.loc[:,'id.orig_h'] = row['id.orig_h'].apply(lambda x: int(ip.ip_address(x)))
    row.loc[:,'id.resp_h'] = row['id.resp_h'].apply(lambda x: int(ip.ip_address(x)))

    # Protocol
    d = {'udp':0,'tcp':1,'icmp':2}
    row.loc[:,'proto'] = row['proto'].apply(lambda x: d[x])

    row.loc[:,'orig_bytes'] = row['orig_bytes'].apply(lambda x: remove_dash(x))
    row.loc[:,'resp_bytes'] = row['resp_bytes'].apply(lambda x: remove_dash(x))
    row.loc[:,'duration'] = row['duration'].apply(lambda x: remove_dash(x))   
    return row

def detailed_preprocess(row):
    # Drop uid 
    row = row.drop(['ts','uid','local_orig','local_resp'], axis=1)

    # Translate IPs to a number
    row.loc[:,'id.orig_h'] = row['id.orig_h'].apply(lambda x: int(ip.ip_address(x)))
    row.loc[:,'id.resp_h'] = row['id.resp_h'].apply(lambda x: int(ip.ip_address(x)))

    # Protocol
    d = {'udp':0,'tcp':1,'icmp':2}
    row.loc[:,'proto'] = row['proto'].apply(lambda x: d[x])

    # Service
    d = {'-':0,'dns':1,'http':2,'dhcp':3,'irc':4,'ssl':5,'ssh':6}
    row.loc[:,'service'] = row['service'].apply(lambda x: d[x])   

    d = {'S0':0,'S1':1,'SF':2,'REJ':3,'S2':4,'S3':5,'RSTO':6,'RSTR':7,'RSTOS0':8,'RSTRH':9,'SH':10,'SHR':11,'OTH':12}
    row.loc[:,'conn_state'] = row['conn_state'].apply(lambda x: d[x])   
    
    row.loc[:,'orig_bytes'] = row['orig_bytes'].apply(lambda x: remove_dash(x))
    row.loc[:,'resp_bytes'] = row['resp_bytes'].apply(lambda x: remove_dash(x))
    row.loc[:,'duration'] = row['duration'].apply(lambda x: remove_dash(x))    

    new_hist = []
    for h in row['history']:
        hist = 0b0
        for char in h:
            if char == 's' or char == 'S':
                hist = hist | 0b10000000000000
            elif char == 'h' or char == 'H':
                hist = hist | 0b1000000000000            
            elif char == 'a' or char == 'A':
                hist = hist | 0b100000000000
            elif char == 'd' or char == 'D':
                hist = hist | 0b10000000000      
            elif char == 'f' or char == 'F':
                hist = hist | 0b1000000000             
            elif char == 'r' or char == 'R':
                hist = hist | 0b100000000 
            elif char == 'c' or char == 'C':
                hist = hist | 0b10000000   
            elif char == 'g' or char == 'G':
                hist = hist | 0b1000000
            elif char == 't' or char == 'T':
                hist = hist | 0b100000     
            elif char == 'w' or char == 'W':
                hist = hist | 0b10000                                                             
            elif char == 'i' or char == 'I':
                hist = hist | 0b1000             
            elif char == 'q' or char == 'Q':
                hist = hist | 0b100
            elif char == '^':
                hist = hist | 0b10
            else:
                hist = hist | 0b1           
        new_hist.append(hist)
    row.loc[:,'new_history'] = new_hist

    row.loc[:,'history_len'] = history_len(row)

    row = row.drop(['history'],axis=1)

    return row