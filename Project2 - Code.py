import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_clusters(X_scaled, max_clusters=10):
    """Iterates through cluster sizes to find the optimal Silhouette score."""
    best_score = -1
    best_n = 2
    
    for n in range(2, max_clusters):
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        if score > best_score:
            best_score = score
            best_n = n
            
    print(f"Optimal clusters: {best_n} with Silhouette Score: {best_score:.2f}")
    return best_n

if __name__ == "__main__":
    # Update path accordingly
    data = pd.read_csv('ds_salaries.csv')
    
    X = data.drop(columns=['salary_in_usd'])
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    optimal_k = find_optimal_clusters(X_scaled)
    
    final_model = KMeans(n_clusters=optimal_k, random_state=0).fit(X_scaled)
    data['Cluster'] = final_model.labels_
    
    print(f"Largest cluster size: {pd.Series(final_model.labels_).value_counts().max()}")
