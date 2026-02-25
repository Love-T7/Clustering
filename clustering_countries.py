import pandas as pd
import numpy
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

#Import data at the global level
data=pd.read_excel('17 variables on energy for 45 European countries in 2020.xlsx')

#selecting the attributes....remove the attribute call country
select_attrib=data[['Mineral Depletion','Energy Depletion','Net ODA received','Fuel Exports','Fuel Imports','GDP','Population','Per capita electricity consumption','Human Development Index','Coal electricity generation','Oil electricity generation','Natural gas electricity generation','Nuclear electricity generation','Hydroelectric electricity generation','Solar electricity generation','Wind electricity generation','Other RS electricity generation']]


select_attrib = select_attrib.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
print(select_attrib.isna().sum())  # Check for missing values

# Fill missing values with column mean (or use another strategy)
select_attrib = select_attrib.fillna(select_attrib.mean())


#Scaling the dataset for analysis. Scaling reduces the bias in the data
scaler=MinMaxScaler(feature_range=(0,1)) #technque for scaling data. There are other techniques. Please explore them
df_scaled = scaler.fit_transform(select_attrib)

# this codes adds column heading to the dataset after scaling
col_header=['Mineral Depletion','Energy Depletion','Net ODA received','Fuel Exports','Fuel Imports','GDP','Population','Per capita electricity consumption','Human Development Index','Coal electricity generation','Oil electricity generation','Natural gas electricity generation','Nuclear electricity generation','Hydroelectric electricity generation','Solar electricity generation','Wind electricity generation','Other RS electricity generation']
scaled_data=pd.DataFrame(df_scaled,columns=col_header)

# Functions to create pages and place them in selectbox
def page1():
    st.header('Project description')
    '''Classification of European countries according to indicators related to electricity generation by Alvaro Gonzalez-Lorente, Montserrat Hernandez-Lopez, Imanol L. Nieto-Gonzalez  and Francisco Javier Martín-Alvarez. We will be critiquing this journal paper and performing same clustering using the ward's, single, complete, average and centroid linkage. We would also perform an alternative clustering method to compare and see which is best. By: Love Thompson (22255085), Ezekiel Tetteh Baako (22255416), Hotor Jasper Aseye (22254023), Asamoah Hayford (22253145)
'''
def page2():
    if st.checkbox("Original dataset"): #displaying data when a checkbox is selected
        st.write(data)

    if st.checkbox('Selected attributes'):  #displaying selected attribute data when a checkbox is selected
        st.write(select_attrib)

    if st.checkbox("Scaled data"): #displaying scaled data when a checkbox is selected
        st.write(scaled_data)

    #importing data from the web
    csv_file=st.file_uploader("Click to upload your Csv file", type=['csv'])

    # Check if file is uploaded
    if csv_file is not None:
        # Read CSV file
        df = pd.read_csv(csv_file)

        # Display dataset
        st.subheader("Uploaded Dataset:")
        st.write(df)

        # Display dataset statistics
        st.subheader("Dataset Statistics:")
        st.write(df.describe())

        # Display dataset information
        st.subheader("Dataset Information:")
        st.write(df.info())

def page3():
    st.header("Exploratory Data Analysis")

    st.write("click on the check box to display")
    if st.checkbox('Description of Selected Attributes'):
        st.write(select_attrib.describe())
    # Visualize missing values
    if st.checkbox("Checking for Missing Values"):
        missing_values = select_attrib.isna().sum()
        st.write("Number of Missing Values per Column:")
        st.write(missing_values)

    # Correlation Heatmap
    if st.checkbox('Correlation Heatmap'):
        corr = select_attrib.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Pairplot for Relationships
    if st.checkbox("Pairplots"):
    # List available variables
        available_vars = select_attrib.columns.tolist()
        st.write("Available variables for pairplot:", available_vars)

    # Multiselect for variables
        pairplot_vars = st.multiselect("Select at least two variables for pairplot", available_vars)

        if pairplot_vars:
            if len(pairplot_vars) >= 2:
                st.write(f"Variables selected: {pairplot_vars}")
                fig = sns.pairplot(select_attrib[pairplot_vars], diag_kind='kde')
                st.pyplot(fig)
            else:
                st.write("Please select at least two variables for the pairplot.")
        else:
            st.write("No variables selected. Please choose variables for pairplot.")

def page4():
    st.header("Hierarchical Clustering")

    # Agglomerative Clustering
    st.subheader("Agglomerative Clustering")

    # list of options for number of clusters
    cluster_options_Agg = [2, 3, 4, 5, 6]
    linkage_methods = ['ward', 'single', 'complete', 'average', 'centroid']

    # create a select box widget to hold the cluster options
    n_clusters_Agg = st.selectbox("Select your preferred number of cluster", cluster_options_Agg)
    selected_linkage = st.selectbox("Select the Linkage Method", linkage_methods)

    # compute for the distance
    distances = pdist(scaled_data, metric='euclidean')
    linkage_matrix = linkage(distances, method= selected_linkage)  # there is also complete, single

    # creating agglomerative instance with complete linkage
    Agg = AgglomerativeClustering(n_clusters=n_clusters_Agg, linkage= selected_linkage)  # there is also complete, single
    # fit the agglomerative on the dataset
    Agg.fit(scaled_data)

    data[f'{selected_linkage.capitalize()} Cluster'] = Agg.labels_
    st.write(f"Clusters created using {selected_linkage.capitalize()} Linkage:")
    st.write(data)

    # Creating a dendrogram
    # write the title for the dendrogram
    st.write(f"Dendrogram for {selected_linkage.capitalize()} Linkage")

    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linkage_matrix, ax=ax)
    ax.set_title(f"Dendrogram ({selected_linkage.capitalize()} Linkage)")
    ax.set_xlabel("Attributes")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

    # Evaluation
    sil = silhouette_score(scaled_data, Agg.labels_)  # the silhouette function takes in the dataset and the label
    st.write(f"Silhouette Score for {selected_linkage.capitalize()} Linkage: {sil * 100}")

    # Divisive Clustering
    st.subheader("Divisive Hierarchical Clustering")
    div = BisectingKMeans(n_clusters=n_clusters_Agg, random_state=42)
    div.fit(scaled_data)

    st.write("Check the last column of this table for the clusters: ")
    data['Div Cluster'] = div.labels_
    st.write(data)

    # Evaluation
    sil = silhouette_score(scaled_data, div.labels_)  # the silhouette function takes in the dataset and the label
    st.write(f"Silhouette Score for Divisive Clustering: {sil * 100}")

    # Creating a dendrogram
    # write the title for the dendrogram
    st.write("Dendrogram for Divisive Clustering")

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    dendrogram(linkage_matrix, ax=ax2)
    ax2.set_title("Dendrogram for Divisive Clustering")
    ax2.set_xlabel("Attributes")
    ax2.set_ylabel("Distance")
    st.pyplot(fig2)

def page5():
    st.header("K-Means Clustering")
    cluster_options = [2, 3, 4, 5, 6]

    # create a selectbox
    n_clusters = st.selectbox("Select your preferred number of clusters", cluster_options)

    # Kmeans clustering
    kmeansalg = KMeans(n_clusters=n_clusters, random_state=30)
    kmeansalg.fit(scaled_data)

    # diplaying the output. A new column has been added to the origainal data indicating the clusters.
    data['Kmeans Cluster'] = kmeansalg.labels_
    st.write("Check the last column of this table for the clusters: ")
    st.write(data)

    # Evaluation of the algorithm
    sil = silhouette_score(scaled_data, kmeansalg.labels_)  # the silhoutte function takes in the dataset and the lable
    st.write("The evaluation score is", sil * 100)

    # Visualisation the cluster points...Matplotib and seanborn are both 2D
    st.subheader("Fuel Exports VS Fuel Imports")
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    ax3.scatter(scaled_data['Fuel Exports'], scaled_data['Fuel Imports'], c=data['Kmeans Cluster'], cmap='viridis')
    ax3.set_title('Fuel Exports vs Fuel Imports datapoints')
    ax3.set_xlabel('Fuel Exports')
    ax3.set_ylabel('Fuel Imports')
    st.pyplot(fig3)

    st.subheader("GDP VS Population")
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    ax3.scatter(scaled_data['GDP'], scaled_data['Population'], c=data['Kmeans Cluster'], cmap='viridis')
    ax3.set_title('GDP vs Population Imports datapoints')
    ax3.set_xlabel('GDP')
    ax3.set_ylabel('Population')
    st.pyplot(fig3)

    st.subheader("Per capita electricity consumption VS Human Development Index")
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    ax4.scatter(scaled_data['Per capita electricity consumption'], scaled_data['Human Development Index'], c=data['Kmeans Cluster'], cmap='viridis')
    ax4.set_title('Per capita electricity consumption vs Population Imports datapoints')
    ax4.set_xlabel('Per capita electricity consumption')
    ax4.set_ylabel('Human Development Index')
    st.pyplot(fig4)

    st.subheader("Mineral Depletion VS Energy Depletion")
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    ax5.scatter(scaled_data['Mineral Depletion'], scaled_data['Energy Depletion'], c=data['Kmeans Cluster'], cmap='viridis')
    ax5.set_title('Mineral Depletion VS Energy Depletion datapoints')
    ax5.set_xlabel('Mineral Depletion')
    ax5.set_ylabel('Energy Depletion')
    st.pyplot(fig5)



#linking pages to the sidebar
pages={
    'Project Description':page1,
    'Loading Dataset':page2,
    'Exploratory Data Analysis':page3,
    'Hierarchical Clustering':page4,
    'K-means': page5,
}

# Sidebar items
select_page=st.sidebar.selectbox("select a page", list(pages.keys()))
st.sidebar.header (f'Unsupervised Learning Project')
st.sidebar.write('Unsupervised learning is a type of machine learning where algorithms are trained on unlabeled data to discover patterns, relationships, or groupings within the data')

#show the page
pages[select_page]()
