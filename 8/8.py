import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv(
    "flats_for_clustering.tsv",
    sep="\t",)

data.replace("parter", 0, inplace=True)
data.replace("niski parter", 0, inplace=True)
data['Piętro'] = data.apply(lambda row: row['Liczba pięter w budynku'] if row['Piętro'] == 'poddasze' else row['Piętro'], axis=1)
columns_to_check = ['cena', 'Powierzchnia w m2', 'Liczba pokoi', 'Liczba pięter w budynku', 'Piętro']
data = data.apply(pd.to_numeric, errors='coerce')
std_threshold = 3

for column in columns_to_check:
    mean_value = data[column].mean()
    std_value = data[column].std()
    lower_bound = mean_value - std_threshold * std_value
    upper_bound = mean_value + std_threshold * std_value
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

X = data

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)


data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data_imputed['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Klastry na płaszczyźnie PCA')
plt.xlabel('PCA Komponent 1')
plt.ylabel('PCA Komponent 2')
plt.show()





