import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from kmodes.kmodes import KModes
import numpy as np

# Read the input dataset
df = pd.read_csv('Input_data.csv')
print(f"Total members in dataset: {len(df)}")

def convert_multi_select(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Converts multi-select categorical fields into binary encoded format."""
    df_clean = df[df[column].notna()].copy()
    column_values = df_clean[column].str.split(';')
    mlb = MultiLabelBinarizer()
    binary_encoded = pd.DataFrame(
        mlb.fit_transform(column_values),
        columns=mlb.classes_,
        index=df_clean.index
    )
    return binary_encoded

# Process career aspirations
career_aspirations_binary = convert_multi_select(df, '\xa0Career Aspirations:')

# Merge processed data
df_final = pd.concat([df, career_aspirations_binary.add_prefix('CA_')], axis=1)
print(f"Total members after merging career aspirations: {len(df_final)}")

# Drop unnecessary columns
df_final = df_final.drop(columns=['Last modified time'], errors='ignore')

def map_custom_groups(time_zone: str, location: str) -> str:
    """Maps employees into predefined groups based on time zone and location."""
    if time_zone in ['Eastern Time (ET)', 'Central Time (CT'] and location in ['Remote by Design', 'Field']:
        return 'EST_CST_Remote_Field'
    elif time_zone in ['Mountain Time (MT)', 'Pacific Time (PT'] and location in ['Remote by Design', 'Field']:
        return 'MST_PST_Remote_Field'
    elif time_zone in ['Eastern Time (ET)'] and location == 'Onsite - Basking Ridge':
        return 'EST_Onsite'
    elif time_zone in ['Mountain Time (MT)'] and location == 'Onsite':
        return 'MST_Onsite'
    elif time_zone in ['Central Time (CT'] and location == 'Onsite':
        return 'CST_Onsite'
    elif time_zone in ['Pacific Time (PT'] and location == 'Onsite':
        return 'PST_Onsite'
    return f'{time_zone}_{location}'

# Apply grouping function
df_final['Custom Group'] = df_final.apply(lambda row: map_custom_groups(row['Time Zone:'], row['Location:']), axis=1)
print(f"Total unique custom groups formed: {df_final['Custom Group'].nunique()}")

def k_modes_clustering(df_group: pd.DataFrame, n_clusters: int) -> list:
    """Performs K-Modes clustering on career aspirations."""
    binary_features = df_group.filter(like='CA_')
    km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
    cluster_labels = km.fit_predict(binary_features)
    
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(df_group.index[idx])
    
    return list(clusters.values())

all_grouped_data = []

def process_clusters(df_group: pd.DataFrame, group_name: str):
    """Processes the groups by role and applies clustering where needed."""
    print(f"Processing group {group_name} with {len(df_group)} members")
    remaining_members = set(df_group.index.tolist())
    
    if len(df_group) < 5:
        for _, row in df_group.iterrows():
            all_grouped_data.append([group_name, row['Name'], row['Email'], row['\xa0Career Aspirations:']])
        remaining_members -= set(df_group.index.tolist())
        return
    
    role_groups = df_group.groupby('Current Role:')
    for role_name, role_group in role_groups:
        print(f"  Role: {role_name} has {len(role_group)} members")
        if len(role_group) < 5:
            for _, row in role_group.iterrows():
                all_grouped_data.append([f"{group_name} - {role_name}", row['Name'], row['Email'], row['\xa0Career Aspirations:']])
            remaining_members -= set(role_group.index.tolist())
        else:
            print(f"  Clustering {len(role_group)} members in {role_name} based on Career Aspirations")
            n_clusters = max(1, len(role_group) // 5)
            clusters = k_modes_clustering(role_group, n_clusters)
            
            for cluster_id, indices in enumerate(clusters, start=1):
                group = indices[:5]
                for idx in group:
                    row = role_group.loc[idx]
                    all_grouped_data.append([f"{group_name} - {role_name} - Cluster {cluster_id}", row['Name'], row['Email'], row['\xa0Career Aspirations:']])
                remaining_members -= set(group)
    
    if remaining_members:
        print(f"  Remaining ungrouped members in {group_name}: {len(remaining_members)}")
        for idx in remaining_members:
            row = df_group.loc[idx]
            all_grouped_data.append([f"{group_name} - Ungrouped", row['Name'], row['Email'], row['\xa0Career Aspirations:']])

for group_name, group in df_final.groupby('Custom Group'):
    process_clusters(group, group_name)

final_grouped = pd.DataFrame(all_grouped_data, columns=['Group', 'Name', 'Email', 'Career Aspirations'])
final_grouped.insert(0, 'Sl.no', range(1, len(final_grouped) + 1))
print(f"Total members in final output: {len(final_grouped)} (Expected: {len(df_final)})")

output_file = "Mentee_grouping.csv"
final_grouped.to_csv(output_file, index=False)