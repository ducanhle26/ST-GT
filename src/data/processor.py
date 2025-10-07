"""
Data Processing Module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class TopSectionsDataProcessor:
    """Process and prepare data from top sections"""
    
    def __init__(self, top_N: int = 15):
        self.top_N = top_N
        self.feature_combinations = {}

    def load_and_analyze_top_sections(self, filepath: str) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
        """
        Load data and identify top N sections by record count
        
        Args:
            filepath: Path to parquet file
            
        Returns:
            df: Full dataframe
            top_sections: List of top section IDs
            analysis_df: Analysis dataframe
        """
        print("🔍 Loading and analyzing top sections...")
        df = pd.read_parquet(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        section_analysis = []
        for section_id in df['SectionID'].unique():
            section_data = df[df['SectionID'] == section_id]
            cause_columns = [col for col in df.columns if col.startswith('Cause_')]
            total_causes = section_data[cause_columns].sum().sum()
            unique_timestamps = section_data['Timestamp'].nunique()
            date_span = (section_data['Timestamp'].max() - section_data['Timestamp'].min()).days
            
            section_analysis.append({
                'SectionID': section_id,
                'Records': len(section_data),
                'Unique_Timestamps': unique_timestamps,
                'Date_Span_Days': date_span,
                'Total_Failures': total_causes,
            })
            
        analysis_df = pd.DataFrame(section_analysis).sort_values('Records', ascending=False)
        top_sections = analysis_df.head(self.top_N)['SectionID'].tolist()
        print(f"✅ Top {self.top_N} sections selected")
        return df, top_sections, analysis_df

    def create_daily_series(self, section_data: pd.DataFrame, section_id: float, 
                          feature_combinations: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Create daily time series for a section
        
        Args:
            section_data: Data for specific section
            section_id: Section identifier
            feature_combinations: Dictionary of feature groups
            
        Returns:
            daily_df: Daily aggregated dataframe
        """
        section_data = section_data.copy()
        section_data.loc[:, 'Timestamp'] = pd.to_datetime(section_data['Timestamp'])
        
        if section_data['Timestamp'].dt.tz is not None:
            section_data['Timestamp'] = section_data['Timestamp'].dt.tz_localize(None)
        
        # Create daily index
        min_date = section_data['Timestamp'].min().floor('D')
        max_date = section_data['Timestamp'].max().floor('D')
        daily_index = pd.date_range(start=min_date, end=max_date, freq='D')
        
        daily_df = pd.DataFrame(index=daily_index)
        daily_df['date'] = daily_df.index
        daily_df['hour'] = 0
        daily_df['dayofweek'] = daily_df['date'].dt.dayofweek
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['days_since_start'] = (daily_df['date'] - min_date).dt.days
        daily_df['is_weekend'] = daily_df['dayofweek'].isin([5, 6]).astype(int)
        
        # Add static features
        static_features = [f for f in feature_combinations['all_static'] 
                          if f not in feature_combinations['temporal_sequence']]
        for f in static_features:
            if f in section_data.columns:
                daily_df[f] = section_data[f].iloc[0] if len(section_data) > 0 else np.nan
        
        # Aggregate failure data
        cause_columns = [col for col in section_data.columns if col.startswith('Cause_')]
        if cause_columns:
            section_data['date'] = pd.to_datetime(section_data['Timestamp'].dt.date)
            
            if daily_df['date'].dt.tz is not None:
                daily_df['date'] = daily_df['date'].dt.tz_localize(None)
            if section_data['date'].dt.tz is not None:
                section_data['date'] = section_data['date'].dt.tz_localize(None)
            
            daily_failures = section_data.groupby('date')[cause_columns].sum()
            daily_df = daily_df.join(daily_failures, on='date')
            
            daily_df['target_total_failures'] = daily_df[cause_columns].sum(axis=1)
            daily_df['target_has_failure'] = (daily_df['target_total_failures'] > 0).astype(int)
            
            daily_df['target_total_failures'] = daily_df['target_total_failures'].fillna(0)
            daily_df['target_has_failure'] = daily_df['target_has_failure'].fillna(0)
        else:
            daily_df['target_total_failures'] = 0
            daily_df['target_has_failure'] = 0
            
        return daily_df

    def define_enhanced_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Define and validate feature groups
        
        Args:
            df: Full dataframe
            
        Returns:
            Dictionary of feature groups
        """
        temporal_sequence_features = ['hour', 'dayofweek', 'month', 'days_since_start', 'is_weekend']
        spatial_features = ['Latitude', 'Longitude', 'min_distance_tail', 'min_distance_head', 
                           'same_community', 'same_cluster']
        topology_features = [
            'degree_centrality_avg', 'degree_centrality_diff', 'betweenness_centrality_avg', 
            'betweenness_centrality_diff', 'closeness_centrality_avg', 'closeness_centrality_diff', 
            'eigenvector_centrality_avg', 'eigenvector_centrality_diff', 'pagerank_avg', 
            'pagerank_diff', 'clustering_coeff_avg', 'clustering_coeff_diff', 'triangles_avg', 
            'triangles_diff'
        ]
        voltage_features = [col for col in df.columns if col.startswith('voltage_')]
        cluster_features = [col for col in df.columns if col.startswith('cluster_')]
        operations_features = [col for col in df.columns if col.startswith('ops_')]
        pmu_features = ['Count']

        def filter_valid_features(features):
            return [f for f in features if f in df.columns and df[f].nunique() > 1 and df[f].notna().sum() > 0]

        self.feature_combinations = {
            'temporal_sequence': filter_valid_features(temporal_sequence_features),
            'spatial': filter_valid_features(spatial_features),
            'topology': filter_valid_features(topology_features),
            'voltage': filter_valid_features(voltage_features),
            'cluster': filter_valid_features(cluster_features),
            'operations': filter_valid_features(operations_features),
            'pmu': filter_valid_features(pmu_features),
        }
        
        self.feature_combinations['all_static'] = (
            self.feature_combinations['spatial'] +
            self.feature_combinations['topology'] +
            self.feature_combinations['voltage'] +
            self.feature_combinations['cluster'] +
            self.feature_combinations['operations'] +
            self.feature_combinations['pmu']
        )
        
        return self.feature_combinations


def create_daily_sequences(daily_df: pd.DataFrame, seq_features: List[str], 
                          static_features: List[str], seq_len: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences from daily data
    
    Args:
        daily_df: Daily dataframe
        seq_features: Features for sequences
        static_features: Static features
        seq_len: Sequence length
        
    Returns:
        Tuple of (sequences, static_data, targets)
    """
    daily_df = daily_df.sort_values('date')
    seq_data = daily_df[seq_features].values
    
    static_features_present = [f for f in daily_df.columns if f in static_features]
    if not static_features_present:
        static_data = np.array([])
    else:
        static_data = daily_df[static_features_present].iloc[0].values
    
    target_data = daily_df['target_has_failure'].values
    sequences, statics, targets = [], [], []
    
    for i in range(seq_len, len(daily_df)):
        hist_seq = seq_data[i-seq_len:i]
        current_target = target_data[i]
        sequences.append(hist_seq)
        statics.append(static_data)
        targets.append(current_target)
        
    return (np.array(sequences, dtype=np.float32), 
            np.array(statics, dtype=np.float32), 
            np.array(targets, dtype=np.float32))


def create_multi_section_sequences(df: pd.DataFrame, top_sections: List[float], 
                                 feature_combinations: Dict[str, List[str]], 
                                 seq_len: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                             np.ndarray, np.ndarray, Dict[float, int]]:
    """
    Create sequences for multiple sections
    
    Args:
        df: Full dataframe
        top_sections: List of section IDs
        feature_combinations: Feature groups
        seq_len: Sequence length
        
    Returns:
        Tuple of (X_seq, X_static, y, sections, node_indices, section_to_node)
    """
    static_features = [f for f in feature_combinations['all_static'] 
                      if f not in feature_combinations['temporal_sequence']]
    all_sequences, all_static, all_targets, all_sections, all_node_indices = [], [], [], [], []
    
    section_to_node = {section_id: i for i, section_id in enumerate(top_sections)}
    
    processor = TopSectionsDataProcessor()
    for section_id in top_sections:
        section_data = df[df['SectionID'] == section_id]
        daily_df = processor.create_daily_series(section_data, section_id, feature_combinations)
        seq_features = feature_combinations['temporal_sequence']
        sequences, statics, targets = create_daily_sequences(daily_df, seq_features, static_features, seq_len)
        
        if len(sequences) > 0:
            all_sequences.extend(sequences)
            all_static.extend(statics)
            all_targets.extend(targets)
            all_sections.extend([section_id] * len(sequences))
            node_idx = section_to_node[section_id]
            all_node_indices.extend([node_idx] * len(sequences))
    
    X_seq = np.array(all_sequences)
    X_static = np.array(all_static)
    y = np.array(all_targets)
    sections = np.array(all_sections)
    node_indices = np.array(all_node_indices)
    
    print(f"Total sequences: {len(X_seq)}")
    print(f"Positive class ratio: {y.mean():.4f}")
    print(f"Number of unique sections/nodes: {len(section_to_node)}")
    
    return X_seq, X_static, y, sections, node_indices, section_to_node


def create_adjacency_matrix(df: pd.DataFrame, top_sections: List[float], 
                          section_to_node: Dict[float, int],
                          threshold_km: float = 50.0) -> np.ndarray:
    """
    Create adjacency matrix based on spatial proximity
    
    Args:
        df: Full dataframe
        top_sections: List of section IDs
        section_to_node: Mapping from section to node index
        threshold_km: Distance threshold in kilometers
        
    Returns:
        Adjacency matrix [num_nodes, num_nodes]
    """
    num_nodes = len(top_sections)
    adj_matrix = np.eye(num_nodes, dtype=bool)
    
    section_coords = {}
    for section_id in top_sections:
        section_data = df[df['SectionID'] == section_id]
        if len(section_data) > 0 and 'Latitude' in section_data.columns and 'Longitude' in section_data.columns:
            lat = section_data['Latitude'].iloc[0]
            lon = section_data['Longitude'].iloc[0]
            if not (pd.isna(lat) or pd.isna(lon)):
                section_coords[section_id] = (lat, lon)
    
    if len(section_coords) > 1:
        try:
            from sklearn.metrics.pairwise import haversine_distances
            import math
            
            coords_list = []
            section_list = []
            for section_id, (lat, lon) in section_coords.items():
                coords_list.append([math.radians(lat), math.radians(lon)])
                section_list.append(section_id)
            
            if len(coords_list) > 1:
                coords_array = np.array(coords_list)
                distances = haversine_distances(coords_array) * 6371
                
                for i, section_i in enumerate(section_list):
                    for j, section_j in enumerate(section_list):
                        if i != j and distances[i, j] < threshold_km:
                            node_i = section_to_node[section_i]
                            node_j = section_to_node[section_j]
                            adj_matrix[node_i, node_j] = True
                            adj_matrix[node_j, node_i] = True
        except ImportError:
            print("Warning: sklearn not available for haversine distance. Using identity matrix.")
    
    return adj_matrix