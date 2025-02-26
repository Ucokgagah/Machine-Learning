# Install scikit-learn jika belum terinstall
# !pip install -U scikit-learn
from sklearn.datasets import load_iris
import pandas as pd
# Memuat dataset Iris dari scikit-learn
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target
print("Dataset Iris dari scikit-learn:")
print(df_iris.head())
# Memuat dataset dari file bungap3.csv
df_bunga = pd.read_csv('bungap3.csv')
print("\nDataset dari bungap3.csv:")
print(df_bunga.head())
# !pip install -U scikit-learn
from sklearn.datasets import load_iris
import pandas as pd
# Memuat dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
# Menampilkan 5 baris pertama dataset
df.head()# Memeriksa ukuran dataset
print(df.shape)
# Memeriksa tipe data setiap kolom
print(df.dtypes)
# Memeriksa nilai yang hilang
print(df.isnull().sum())
# Deskripsi statistik dasar
print(df.describe())

# Distribusi Variabel: Distribusi variabel individual dapat divisualisasikan menggunakan histogram atau boxplot.
import matplotlib.pyplot as plt
import seaborn as sns
# Histogram untuk setiap fitur
df.hist(bins=20, figsize=(10, 10))
plt.show()
# Boxplot untuk setiap fitur
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.iloc[:, :-1])
plt.title('Boxplot of Iris Features')
plt.show()

# Hubungan Antar Variabel: Hubungan antar variabel dapat divisualisasikan menggunakan pairplot atau scatter plot.
# Pairplot untuk melihat hubungan antar variabel
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()
# Scatter plot untuk hubungan spesifik
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species',
data=df, palette='viridis')
plt.title('Scatter plot of Sepal Length vs Sepal Width')
plt.show()

# Visualisasi Kelas: Distribusi kelas dalam dataset dapat divisualisasikan menggunakan count plot atau pie chart.
# Count plot untuk distribusi kelas
plt.figure(figsize=(10, 6))
sns.countplot(x='species', data=df)
plt.title('Distribution of Iris Species')
plt.show()
# Pie chart untuk distribusi kelas
df['species'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.title('Distribution of Iris Species')
plt.ylabel('')
plt.show()
