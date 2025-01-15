import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, interpolate, signal
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks
import os
from sklearn.decomposition import PCA

class KinsaAnalyzer:
    def __init__(self, filepath):
        """Initialize with data loading and basic preprocessing"""
        self.df = pd.read_csv(filepath)
        print("Available columns:", list(self.df.columns))
        self.preprocess_data()
        self.setup_plots_directory()
        
    # [Previous methods remain the same until plot_spline_fit]
    
    def plot_spline_fit(self, signal_data):
        """Create and plot spline fit of the signal"""
        from scipy.interpolate import UnivariateSpline
        
        # Prepare data
        signal_mean = signal_data.groupby('created_date')['fever_signal'].mean()
        x = np.arange(len(signal_mean))
        y = signal_mean.values
        
        # Fit spline with automatic smoothing
        spl = UnivariateSpline(x, y, s=len(x)/200)  # Adjust smoothing parameter
        x_smooth = np.linspace(0, len(x)-1, 300)
        y_smooth = spl(x_smooth)
        
        # Plot
        plt.figure(figsize=(15, 8))
        plt.plot(signal_mean.index, y, 'b.', alpha=0.3, label='Original Data')
        plt.plot(signal_mean.index[np.linspace(0, len(x)-1, 300).astype(int)], 
                y_smooth, 'r-', label='Spline Fit', linewidth=2)
        
        plt.title('Signal with Spline Fit')
        plt.xlabel('Date')
        plt.ylabel('Mean Fever Signal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        self.save_plot(plt, 'spline_fit.png')
        return spl
        
    # [Rest of the class methods remain the same]
        
    # [Previous methods remain the same]

    def plot_signal_analysis(self, signal_data):
        """Plot signal analysis focusing on inflection points"""
        signal_mean = signal_data.groupby('created_date')['fever_signal'].mean()
        signal_values = signal_mean.values
        
        # Create single plot without FFT
        plt.figure(figsize=(15, 8))
        
        # Time domain plot
        plt.plot(signal_mean.index, signal_values, 'b-', label='Original Signal', alpha=0.7)
        
        # Find and plot inflection points
        peaks = self.find_inflection_points(signal_values)
        if len(peaks) > 0:
            plt.scatter(signal_mean.index[peaks], signal_values[peaks],
                       color='red', s=100, marker='o', label='Inflection Points')
        
        plt.title('Signal with Inflection Points')
        plt.xlabel('Date')
        plt.ylabel('Mean Fever Signal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        self.save_plot(plt, 'signal_analysis.png')
        signal_mean = signal_data.groupby('created_date')['fever_signal'].mean()
        signal_values = signal_mean.values
        
        # Perform FFT
        fft_result = np.fft.fft(signal_values)
        frequencies = np.fft.fftfreq(len(signal_values), d=1.0)  # 1 day sampling
        
        # Create composite plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Time domain plot
        ax1.plot(signal_mean.index, signal_values, 'b-', label='Original Signal', alpha=0.7)
        
        # Find and plot inflection points
        peaks = self.find_inflection_points(signal_values)
        if len(peaks) > 0:
            ax1.scatter(signal_mean.index[peaks], signal_values[peaks],
                       color='red', s=100, marker='o', label='Inflection Points')
        
        ax1.set_title('Signal with Inflection Points')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Mean Fever Signal')
        ax1.grid(True)
        ax1.legend()
        
        # Frequency domain plot
        # Only plot positive frequencies up to Nyquist frequency
        pos_freq_mask = (frequencies > 0) & (frequencies <= 0.5)  # Up to Nyquist frequency
        magnitude_spectrum = np.abs(fft_result)/len(signal_values)  # Normalize
        
        ax2.plot(1/frequencies[pos_freq_mask], magnitude_spectrum[pos_freq_mask])  # Plot period instead of frequency
        ax2.set_xscale('log')  # Log scale for better visualization
        ax2.set_title('Frequency Components (FFT Magnitude)')
        ax2.set_xlabel('Period (days)')
        ax2.set_ylabel('Magnitude')
        ax2.grid(True)
        
        plt.tight_layout()
        self.save_plot(plt, 'signal_analysis.png')
        print(f"Found {len(peaks)} significant inflection points using FFT analysis")

    def setup_plots_directory(self):
        """Create plots directory if it doesn't exist"""
        self.plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def save_plot(self, plt, filename):
        """Save plot to plots directory"""
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def preprocess_data(self):
        """Basic preprocessing of the dataset"""
        # Convert date to datetime
        self.df['created_date'] = pd.to_datetime(self.df['created_date'])
        
        # Create age groups
        self.df['age_group'] = np.where(self.df['age_yr'] <= 12, 'Pediatric', 'Adult')
        
        # Create time-based features
        self.df['year'] = self.df['created_date'].dt.year
        self.df['month'] = self.df['created_date'].dt.month
        self.df['season'] = self.df.apply(self._assign_season, axis=1)

    def _assign_season(self, row):
        """Helper function to assign illness seasons (July-June)"""
        if row['month'] >= 7:
            return f"{row['year']}-{row['year']+1}"
        return f"{row['year']-1}-{row['year']}"

    def analyze_demographics(self):
        """Analyze demographic trends over time"""
        # Age distribution over time
        age_trends = self.df.groupby(['year', 'age_group'])['reading_id'].count().unstack()
        
        # Gender distribution over time
        gender_trends = self.df.groupby(['year', 'gender'])['reading_id'].count().unstack()
        
        # Regional distribution over time
        region_trends = self.df.groupby(['year', 'state'])['reading_id'].count().unstack()
        
        # Create and save demographic plots
        self.plot_demographic_trends(age_trends, gender_trends, region_trends)
        
        return {
            'age_trends': age_trends,
            'gender_trends': gender_trends,
            'region_trends': region_trends
        }

    def plot_demographic_trends(self, age_trends, gender_trends, region_trends):
        """Create and save demographic trend plots"""
        # Age trends plot
        plt.figure(figsize=(15, 8))
        age_trends.plot(kind='bar')
        plt.title('Age Distribution Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Readings')
        self.save_plot(plt, 'age_trends.png')
        
        # Gender trends plot
        plt.figure(figsize=(15, 8))
        gender_trends.plot(kind='bar')
        plt.title('Gender Distribution Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Readings')
        self.save_plot(plt, 'gender_trends.png')
        
        # Top 10 states plot
        plt.figure(figsize=(15, 8))
        region_trends.sum().sort_values(ascending=False)[:10].plot(kind='bar')
        plt.title('Top 10 States by Number of Readings')
        plt.xlabel('State')
        plt.ylabel('Number of Readings')
        self.save_plot(plt, 'top_states.png')

    def create_national_signal(self):
        """Create national fever signal with denominator adjustments"""
        # Group by date and calculate fever percentage, maintaining state and county info
        national = self.df.groupby(['created_date', 'state', 'county'])['fever'].agg(['count', 'sum']).reset_index()
        
        # Add season column
        national['year'] = national['created_date'].dt.year
        national['month'] = national['created_date'].dt.month
        national['season'] = national.apply(self._assign_season, axis=1)
        
        # Calculate rolling denominator (7-day average of total readings)
        national['rolling_denom'] = national.groupby(['state', 'county'])['count'].transform(lambda x: x.rolling(7).mean())
        
        # Calculate fever percentage using rolling denominator
        national['fever_signal'] = (national['sum'] / national['rolling_denom']) * 100
        
        # Apply EWMA smoothing
        national['fever_signal'] = national.groupby(['state', 'county'])['fever_signal'].transform(
            lambda x: x.ewm(span=7).mean())
        
        return national

    def create_age_group_signals(self):
        """Create signals split by age groups"""
        age_signals = self.df.groupby(['created_date', 'age_group'])['fever'].agg(['count', 'sum']).reset_index()
        
        signals = {}
        for age_group in ['Adult', 'Pediatric']:
            group_data = age_signals[age_signals['age_group'] == age_group].copy()
            
            # Add season information
            group_data.loc[:, 'year'] = group_data['created_date'].dt.year
            group_data.loc[:, 'month'] = group_data['created_date'].dt.month
            group_data.loc[:, 'season'] = group_data.apply(self._assign_season, axis=1)
            
            # Calculate signals
            group_data.loc[:, 'rolling_denom'] = group_data['count'].rolling(7).mean()
            group_data.loc[:, 'fever_signal'] = (group_data['sum'] / group_data['rolling_denom']) * 100
            group_data.loc[:, 'fever_signal'] = group_data['fever_signal'].ewm(span=7).mean()
            
            signals[age_group] = group_data
        
        self.plot_age_group_signals(signals)
        return signals

    def plot_age_group_signals(self, signals):
        """Plot age group signals comparison"""
        plt.figure(figsize=(15, 8))
        for age_group, data in signals.items():
            plt.plot(data['created_date'], data['fever_signal'], label=age_group)
        plt.title('Fever Signals by Age Group')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal (%)')
        plt.legend()
        self.save_plot(plt, 'age_group_signals.png')

    def create_county_signals(self, states=['California', 'Texas']):
        """Create county-level signals for specified states"""
        county_signals = {}
        
        for state in states:
            state_data = self.df[self.df['state'] == state].copy()
            
            county_data = state_data.groupby(['created_date', 'county'])['fever'].agg(['count', 'sum']).reset_index()
            
            for county in county_data['county'].unique():
                county_subset = county_data[county_data['county'] == county].copy()
                
                # Add season information and calculate signals
                county_subset.loc[:, 'year'] = county_subset['created_date'].dt.year
                county_subset.loc[:, 'month'] = county_subset['created_date'].dt.month
                county_subset.loc[:, 'season'] = county_subset.apply(self._assign_season, axis=1)
                county_subset.loc[:, 'rolling_denom'] = county_subset['count'].rolling(7).mean()
                county_subset.loc[:, 'fever_signal'] = (county_subset['sum'] / county_subset['rolling_denom']) * 100
                county_subset.loc[:, 'fever_signal'] = county_subset['fever_signal'].ewm(span=7).mean()
                
                county_signals[f"{state}_{county}"] = county_subset
        
        self.plot_county_signals(county_signals)
        return county_signals

    def plot_county_signals(self, county_signals):
        """Plot county signals for each state"""
        states = list(set([k.split('_')[0] for k in county_signals.keys()]))
        
        for state in states:
            plt.figure(figsize=(15, 10))  # Increased height for better layout
            state_counties = {k: v for k, v in county_signals.items() if k.startswith(state)}
            
            for county, data in state_counties.items():
                plt.plot(data['created_date'], data['fever_signal'], 
                        label=county.split('_')[1], alpha=0.7)
                
            plt.title(f'Fever Signals by County - {state}')
            plt.xlabel('Date')
            plt.ylabel('Fever Signal (%)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.subplots_adjust(left=0.1, right=0.85, bottom=0.15, top=0.95)  # Explicit margins instead of tight_layout
            self.save_plot(plt, f'county_signals_{state}.png')

    def plot_year_over_year(self, signal_data, title='Year over Year Comparison'):
        """Create year over year comparison plot"""
        plt.figure(figsize=(15, 8))
        
        seasons = signal_data.groupby(['season', 'created_date'])['fever_signal'].mean().unstack(level=0)
        
        for season in seasons.columns:
            plt.plot(range(len(seasons[season])), seasons[season].values, label=season)
        
        plt.title(title)
        plt.xlabel('Days (July to June)')
        plt.ylabel('Fever Signal (%)')
        plt.legend()
        plt.grid(True)
        
        self.save_plot(plt, 'year_over_year.png')
        return plt

    def analyze_regional_patterns(self):
        """Analyze regional patterns and clustering"""
        # Group states into regions
        regions = {
            'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
            'Midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
            'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
            'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI']
        }
        
        # Create region mapping
        state_to_region = {state: region for region, states in regions.items() for state in states}
        
        # Add region to dataframe
        self.df['region'] = self.df['state'].map(state_to_region)
        
        # Calculate regional statistics with handling for missing regions
        regional_stats = self.df[self.df['region'].notna()].groupby(
            ['region', pd.Grouper(key='created_date', freq='W')])['fever'].agg([
                'mean',
                'std',
                'count'
            ]).reset_index()
            
        # Plot regional trends only if we have data
        plt.figure(figsize=(15, 8))
        has_data = False
        
        for region in regions.keys():
            region_data = regional_stats[regional_stats['region'] == region]
            if len(region_data) > 0:
                plt.plot(region_data['created_date'], 
                        region_data['mean'].rolling(window=4).mean(),  # Add smoothing
                        label=region, alpha=0.7)
                has_data = True
        
        if has_data:
            plt.title('Regional Fever Trends Over Time')
            plt.xlabel('Date')
            plt.ylabel('Average Fever Rate')
            plt.legend(loc='upper right')
            plt.grid(True)
            self.save_plot(plt, 'regional_trends.png')
            print(f"Successfully created regional visualization with {regional_stats['region'].nunique()} regions")
        else:
            print("Warning: No regional data available for plotting")
            plt.close()
        
        return regional_stats

    def analyze_county_patterns(self, states=['CA', 'TX']):
        """Analyze county-level patterns for specified states"""
        results = {}
        state_names = {'CA': 'California', 'TX': 'Texas'}
        
        for state in states:
            # Get state data
            state_data = self.df[self.df['state'] == state].copy()
            
            if state_data.empty:
                print(f"No data found for state code: {state}")
                continue
                
            print(f"Analyzing patterns for {state_names.get(state, state)}")
            
            # Calculate county statistics
            county_stats = state_data.groupby('county').agg({
                'fever': ['mean', 'std', 'count']
            }).reset_index()
            county_stats.columns = ['county', 'mean', 'std', 'count']
            
            # Sort counties by volume
            county_stats = county_stats.sort_values('count', ascending=False)
            
            # Get top 10 counties by volume
            top_counties = county_stats.nlargest(10, 'count')
            
            if len(top_counties) > 0:
                plt.figure(figsize=(15, 10))  # Increased height
                has_data = False
                
                for _, county_row in top_counties.iterrows():
                    county = county_row['county']
                    county_data = state_data[state_data['county'] == county]
                    
                    if not county_data.empty:
                        # Group by date and calculate 7-day rolling mean of fever
                        daily_avg = county_data.groupby('created_date')['fever'].mean()
                        rolling_avg = daily_avg.rolling(window=7, min_periods=1).mean()
                        
                        if len(rolling_avg) > 0:
                            plt.plot(rolling_avg.index, rolling_avg.values, 
                                   label=f'County {county}', alpha=0.7)
                            has_data = True
                
                if has_data:
                    plt.title(f'Fever Trends in Top 10 Counties - {state_names.get(state, state)}')
                    plt.xlabel('Date')
                    plt.ylabel('Fever Rate (7-day rolling average)')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True)
                    plt.subplots_adjust(left=0.1, right=0.85, bottom=0.15, top=0.95)
                    self.save_plot(plt, f'county_patterns_{state}.png')
                    print(f"Created visualization for {state_names.get(state, state)} with {len(top_counties)} counties")
                else:
                    print(f"No plotting data available for {state}")
                    plt.close()
            else:
                print(f"No counties found for {state}")
            
            results[state] = county_stats
            
        return results  # Changed from full names to codes
        """Analyze county-level patterns for specified states"""
        results = {}
        
        state_names = {'CA': 'California', 'TX': 'Texas'}  # For display purposes
        
        for state in states:
            # Get state data
            state_data = self.df[self.df['state'] == state].copy()
            
            if state_data.empty:
                print(f"No data found for state code: {state}")
                continue
                
            print(f"Analyzing patterns for {state_names.get(state, state)}")
            
            # Calculate county statistics
            county_stats = state_data.groupby('county').agg({
                'fever': ['mean', 'std', 'count']
            }).reset_index()
            county_stats.columns = ['county', 'mean', 'std', 'count']
            
            # Sort counties by volume
            county_stats = county_stats.sort_values('count', ascending=False)
            
            # Get top 10 counties by volume
            top_counties = county_stats.nlargest(10, 'count')
            
            if len(top_counties) > 0:
                plt.figure(figsize=(15, 8))
                has_data = False
                
                for _, county_row in top_counties.iterrows():
                    county = county_row['county']
                    county_data = state_data[state_data['county'] == county]
                    
                    if not county_data.empty:
                        # Group by date and calculate 7-day rolling mean of fever
                        daily_avg = county_data.groupby('created_date')['fever'].mean()
                        rolling_avg = daily_avg.rolling(window=7, min_periods=1).mean()
                        
                        if len(rolling_avg) > 0:
                            plt.plot(rolling_avg.index, rolling_avg.values, 
                                   label=f'County {county}', alpha=0.7)
                            has_data = True
                
                if has_data:
                    plt.title(f'Fever Trends in Top 10 Counties - {state_names.get(state, state)}')
                    plt.xlabel('Date')
                    plt.ylabel('Fever Rate (7-day rolling average)')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True)
                    plt.tight_layout()
                    self.save_plot(plt, f'county_patterns_{state}.png')
                    print(f"Created visualization for {state_names.get(state, state)} with {len(top_counties)} counties")
                else:
                    print(f"No plotting data available for {state}")
                    plt.close()
            else:
                print(f"No counties found for {state}")
            
            results[state] = county_stats
            
        return results
        results = {}
        
        for state in states:
            # Get state data
            state_data = self.df[self.df['state'] == state].copy()
            
            if state_data.empty:
                print(f"No data found for state: {state}")
                continue
                
            # Calculate county statistics
            county_stats = state_data.groupby('county').agg({
                'fever': ['mean', 'std', 'count']
            }).reset_index()
            county_stats.columns = ['county', 'mean', 'std', 'count']
            
            # Sort counties by volume
            county_stats = county_stats.sort_values('count', ascending=False)
            
            # Get top 10 counties by volume
            top_counties = county_stats.nlargest(10, 'count')
            
            plt.figure(figsize=(15, 8))
            
            # Plot each top county's trend
            has_data = False
            for _, county_row in top_counties.iterrows():
                county = county_row['county']
                county_data = state_data[state_data['county'] == county]
                
                if not county_data.empty:
                    # Group by date and calculate mean fever
                    daily_avg = county_data.groupby('created_date')['fever'].mean()
                    if len(daily_avg) > 0:
                        plt.plot(daily_avg.index, daily_avg.values, 
                               label=f'County {county}', alpha=0.7)
                        has_data = True
            
            if has_data:
                plt.title(f'Fever Trends in Top 10 Counties - {state}')
                plt.xlabel('Date')
                plt.ylabel('Fever Rate')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True)
                plt.tight_layout()
                self.save_plot(plt, f'county_patterns_{state}.png')
            else:
                print(f"No plotting data available for {state}")
                plt.close()
            
            results[state] = county_stats
            
        return results

    def detect_anomalies(self, signal_data):
        """Detect anomalies using advanced signal processing and dimensionality reduction"""
        from scipy import signal, interpolate
        from sklearn.decomposition import PCA
        from scipy.stats import zscore
        
        # Calculate time span of data
        date_range = signal_data['created_date'].agg(['min', 'max'])
        total_days = (date_range['max'] - date_range['min']).days
        
        # Prepare data for PCA
        pivot_data = signal_data.pivot_table(
            index='created_date', 
            columns='state', 
            values='fever_signal',
            aggfunc='mean'
        ).ffill().bfill()
        
        # Apply bandpass filter to each state's signal
        fs = 1  # 1 sample per day
        lowcut = 1/14  # 14-day seasonal variations
        highcut = 1/3   # 3-day variations
        b, a = signal.butter(3, [lowcut, highcut], btype='band', fs=fs)
        
        filtered_data = pd.DataFrame(
            index=pivot_data.index,
            columns=pivot_data.columns,
            data=np.apply_along_axis(
                lambda x: signal.filtfilt(b, a, x),
                0,
                pivot_data.values
            )
        )
        
        # Apply PCA to filtered data
        pca = PCA(n_components=0.90)  # Keep components explaining 90% of variance
        pca_result = pca.fit_transform(filtered_data)
        
        # Plot PCA explained variance right after PCA calculation
        plt.figure(figsize=(10, 6))
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% Threshold')
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.grid(True)
        plt.legend()
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.95)
        self.save_plot(plt, 'pca_variance.png')
        
        # Reconstruct signal using dominant components
        reconstructed = pd.DataFrame(
            pca.inverse_transform(pca_result),
            index=filtered_data.index,
            columns=filtered_data.columns
        )
        
        # Calculate residuals and their z-scores
        residuals = filtered_data - reconstructed
        zscore_residuals = pd.DataFrame(
            zscore(residuals, axis=0),
            index=residuals.index,
            columns=residuals.columns
        )
        
        # Calculate spatial correlations with better layout handling
        plt.figure(figsize=(15, 10))
        corr_matrix = pivot_data.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title('Spatial Correlations Between States')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.subplots_adjust(bottom=0.15, right=0.95)
        self.save_plot(plt, 'spatial_correlations.png')
        
        # More stringent anomaly detection
        magnitude_anomalies = (abs(zscore_residuals) > 3.5).any(axis=1)
        spatial_spread = (abs(zscore_residuals) > 2).sum(axis=1) / zscore_residuals.shape[1]
        anomaly_mask = magnitude_anomalies & (spatial_spread < 0.25)
        
        anomaly_dates = zscore_residuals.index[anomaly_mask]
        
        # Create final anomalies dataset
        anomalies = signal_data[signal_data['created_date'].isin(anomaly_dates)].copy()
        anomalies['date'] = pd.to_datetime(anomalies['created_date']).dt.date
        
        # Additional filtering: only keep states with significant deviations
        state_dates = []
        for date in anomaly_dates:
            significant_states = zscore_residuals.loc[date].abs() > 3
            states_affected = significant_states[significant_states].index
            for state in states_affected:
                state_dates.append((date.date(), state))
        
        state_date_pairs = pd.DataFrame(state_dates, columns=['date', 'state'])
        anomalies = anomalies.merge(state_date_pairs, on=['date', 'state'])
        
        # Calculate daily statistics
        daily_stats = anomalies.groupby('date').agg({
            'state': 'nunique',
            'fever_signal': 'count'
        }).reset_index()
        daily_stats.columns = ['date', 'states_affected', 'anomaly_count']
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"PCA components used: {pca_result.shape[1]} (explaining 90% variance)")
        print(f"Days with anomalies: {len(anomaly_dates):,} ({(len(anomaly_dates)/total_days*100):.2f}% of time span)")
        print(f"Total anomalous observations: {len(anomalies):,} ({(len(anomalies)/len(signal_data)*100):.2f}%)")
        print(f"States showing anomalies: {anomalies['state'].nunique():,}")
        print(f"Average states per anomaly day: {daily_stats['states_affected'].mean():.1f}")
        
        return anomalies
        """Detect anomalies using advanced signal processing and dimensionality reduction"""
        from scipy import signal, interpolate
        from sklearn.decomposition import PCA
        from scipy.stats import zscore
        
        # Calculate time span of data
        date_range = signal_data['created_date'].agg(['min', 'max'])
        total_days = (date_range['max'] - date_range['min']).days
        
        # First, fit splines to smooth the data
        signal_data = signal_data.sort_values('created_date')
        days_elapsed = (signal_data['created_date'] - signal_data['created_date'].min()).dt.days.values
        
        # Fit spline for visualization
        spl = interpolate.splrep(days_elapsed, signal_data['fever_signal'], k=3)
        smooth_days = np.linspace(0, total_days, 300)
        smooth_signal = interpolate.splev(smooth_days, spl)
        
        # Create date range for smooth signal
        smooth_dates = pd.date_range(start=signal_data['created_date'].min(), 
                                   periods=len(smooth_signal), 
                                   freq='D')
        
        # Plot spline fit
        plt.figure(figsize=(15, 8))
        plt.plot(signal_data['created_date'], signal_data['fever_signal'], 'o', 
                alpha=0.2, label='Original Data')
        plt.plot(smooth_dates, smooth_signal, 'r-', label='Spline Fit')
        plt.title('Fever Signal with Spline Fit')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal')
        plt.legend()
        plt.subplots_adjust(right=0.95, bottom=0.15)
        self.save_plot(plt, 'spline_fit.png')
        
        # Prepare data for PCA
        pivot_data = signal_data.pivot_table(
            index='created_date', 
            columns='state', 
            values='fever_signal',
            aggfunc='mean'
        ).ffill().bfill()
        
        # Apply bandpass filter to each state's signal
        fs = 1  # 1 sample per day
        lowcut = 1/14  # 14-day seasonal variations
        highcut = 1/3   # 3-day variations
        b, a = signal.butter(3, [lowcut, highcut], btype='band', fs=fs)
        
        filtered_data = pd.DataFrame(
            index=pivot_data.index,
            columns=pivot_data.columns,
            data=np.apply_along_axis(
                lambda x: signal.filtfilt(b, a, x),
                0,
                pivot_data.values
            )
        )
        
        # Apply PCA to filtered data
        pca = PCA(n_components=0.90)  # Keep components explaining 90% of variance
        pca_result = pca.fit_transform(filtered_data)
        
        # Reconstruct signal using dominant components
        reconstructed = pd.DataFrame(
            pca.inverse_transform(pca_result),
            index=filtered_data.index,
            columns=filtered_data.columns
        )
        
        # Calculate residuals and their z-scores
        residuals = filtered_data - reconstructed
        zscore_residuals = pd.DataFrame(
            zscore(residuals, axis=0),
            index=residuals.index,
            columns=residuals.columns
        )
        
        # Calculate spatial correlations with better layout handling
        plt.figure(figsize=(15, 10))  # Increased height
        corr_matrix = pivot_data.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title('Spatial Correlations Between States')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.subplots_adjust(bottom=0.15, right=0.95)  # Explicit margin adjustment
        self.save_plot(plt, 'spatial_correlations.png')
        
        # More stringent anomaly detection
        magnitude_anomalies = (abs(zscore_residuals) > 3.5).any(axis=1)  # Increased threshold
        spatial_spread = (abs(zscore_residuals) > 2).sum(axis=1) / zscore_residuals.shape[1]
        anomaly_mask = magnitude_anomalies & (spatial_spread < 0.25)  # Limit spatial spread
        
        anomaly_dates = zscore_residuals.index[anomaly_mask]
        
        # Create final anomalies dataset
        anomalies = signal_data[signal_data['created_date'].isin(anomaly_dates)].copy()
        # Convert to datetime.date() objects consistently
        anomalies['date'] = pd.to_datetime(anomalies['created_date']).dt.date
        
        # Additional filtering: only keep states with significant deviations
        state_dates = []
        for date in anomaly_dates:
            significant_states = zscore_residuals.loc[date].abs() > 3
            states_affected = significant_states[significant_states].index
            for state in states_affected:
                # Convert date to datetime.date() object
                state_dates.append((date.date(), state))
        
        state_date_pairs = pd.DataFrame(state_dates, columns=['date', 'state'])
        # Now both dataframes should have date as datetime.date objects
        anomalies = anomalies.merge(state_date_pairs, on=['date', 'state'])
        
        # Calculate daily statistics
        daily_stats = anomalies.groupby('date').agg({
            'state': 'nunique',
            'fever_signal': 'count'
        }).reset_index()
        daily_stats.columns = ['date', 'states_affected', 'anomaly_count']
        
        # Visualize results
        plt.figure(figsize=(15, 8))
        
        # Plot original and reconstructed signals for a sample state
        sample_state = pivot_data.columns[0]
        plt.plot(pivot_data.index, pivot_data[sample_state], 
                label='Original Signal', alpha=0.5)
        plt.plot(reconstructed.index, reconstructed[sample_state], 
                label='Reconstructed Signal', alpha=0.5)
        
        # Plot anomalies
        if len(anomalies) > 0:
            state_anomalies = anomalies[anomalies['state'] == sample_state]
            if not state_anomalies.empty:
                plt.scatter(state_anomalies['created_date'], 
                           state_anomalies['fever_signal'],
                           color='red', marker='o', label='Anomalies')
        
        plt.title(f'Signal Analysis and Anomalies ({sample_state})')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.85)
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"PCA components used: {pca_result.shape[1]} (explaining 90% variance)")
        print(f"Days with anomalies: {len(anomaly_dates):,} ({(len(anomaly_dates)/total_days*100):.2f}% of time span)")
        print(f"Total anomalous observations: {len(anomalies):,} ({(len(anomalies)/len(signal_data)*100):.2f}%)")
        print(f"States showing anomalies: {anomalies['state'].nunique():,}")
        print(f"Average states per anomaly day: {daily_stats['states_affected'].mean():.1f}")
        
        return anomalies
        """Detect anomalies using advanced signal processing and dimensionality reduction"""
        from scipy import signal, interpolate
        from sklearn.decomposition import PCA
        from scipy.stats import zscore
        
        # Calculate time span of data
        date_range = signal_data['created_date'].agg(['min', 'max'])
        total_days = (date_range['max'] - date_range['min']).days
        
        # First, fit splines to smooth the data
        signal_data = signal_data.sort_values('created_date')
        days_elapsed = (signal_data['created_date'] - signal_data['created_date'].min()).dt.days.values
        
        # Fit spline for visualization
        spl = interpolate.splrep(days_elapsed, signal_data['fever_signal'], k=3)
        smooth_days = np.linspace(0, total_days, 300)
        smooth_signal = interpolate.splev(smooth_days, spl)
        
        # Create date range for smooth signal
        smooth_dates = pd.date_range(start=signal_data['created_date'].min(), 
                                   periods=len(smooth_signal), 
                                   freq='D')
        
        # Plot spline fit
        plt.figure(figsize=(15, 8))
        plt.plot(signal_data['created_date'], signal_data['fever_signal'], 'o', 
                alpha=0.2, label='Original Data')
        plt.plot(smooth_dates, smooth_signal, 'r-', label='Spline Fit')
        plt.title('Fever Signal with Spline Fit')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal')
        plt.legend()
        plt.subplots_adjust(right=0.95, bottom=0.15)
        self.save_plot(plt, 'spline_fit.png')
        
        # Prepare data for PCA
        pivot_data = signal_data.pivot_table(
            index='created_date', 
            columns='state', 
            values='fever_signal',
            aggfunc='mean'
        ).ffill().bfill()
        
        # Apply bandpass filter to each state's signal
        fs = 1  # 1 sample per day
        lowcut = 1/14  # 14-day seasonal variations
        highcut = 1/3   # 3-day variations
        b, a = signal.butter(3, [lowcut, highcut], btype='band', fs=fs)
        
        filtered_data = pd.DataFrame(
            index=pivot_data.index,
            columns=pivot_data.columns,
            data=np.apply_along_axis(
                lambda x: signal.filtfilt(b, a, x),
                0,
                pivot_data.values
            )
        )
        
        # Apply PCA to filtered data
        pca = PCA(n_components=0.90)  # Keep components explaining 90% of variance
        pca_result = pca.fit_transform(filtered_data)
        
        # Reconstruct signal using dominant components
        reconstructed = pd.DataFrame(
            pca.inverse_transform(pca_result),
            index=filtered_data.index,
            columns=filtered_data.columns
        )
        
        # Calculate residuals and their z-scores
        residuals = filtered_data - reconstructed
        zscore_residuals = pd.DataFrame(
            zscore(residuals, axis=0),
            index=residuals.index,
            columns=residuals.columns
        )
        
        # Calculate spatial correlations
        plt.figure(figsize=(15, 8))
        corr_matrix = pivot_data.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title('Spatial Correlations Between States')
        plt.tight_layout()  # This one should be okay as it's a heatmap
        self.save_plot(plt, 'spatial_correlations.png')
        
        # More stringent anomaly detection
        # Require both magnitude and spatial criteria
        magnitude_anomalies = (abs(zscore_residuals) > 3.5).any(axis=1)  # Increased threshold
        spatial_spread = (abs(zscore_residuals) > 2).sum(axis=1) / zscore_residuals.shape[1]
        anomaly_mask = magnitude_anomalies & (spatial_spread < 0.25)  # Limit spatial spread
        
        anomaly_dates = zscore_residuals.index[anomaly_mask]
        
        # Create final anomalies dataset
        anomalies = signal_data[signal_data['created_date'].isin(anomaly_dates)].copy()
        anomalies['date'] = anomalies['created_date'].dt.date
        
        # Additional filtering: only keep states with significant deviations
        state_dates = []
        for date in anomaly_dates:
            significant_states = zscore_residuals.loc[date].abs() > 3
            states_affected = significant_states[significant_states].index
            for state in states_affected:
                state_dates.append((date, state))
        
        state_date_pairs = pd.DataFrame(state_dates, columns=['date', 'state'])
        anomalies = anomalies.merge(state_date_pairs, on=['date', 'state'])
        
        # Calculate daily statistics
        daily_stats = anomalies.groupby('date').agg({
            'state': 'nunique',
            'fever_signal': 'count'
        }).reset_index()
        daily_stats.columns = ['date', 'states_affected', 'anomaly_count']
        
        # Visualize results
        plt.figure(figsize=(15, 8))
        
        # Plot original and reconstructed signals for a sample state
        sample_state = pivot_data.columns[0]
        plt.plot(pivot_data.index, pivot_data[sample_state], 
                label='Original Signal', alpha=0.5)
        plt.plot(reconstructed.index, reconstructed[sample_state], 
                label='Reconstructed Signal', alpha=0.5)
        
        # Plot anomalies
        if len(anomalies) > 0:
            state_anomalies = anomalies[anomalies['state'] == sample_state]
            if not state_anomalies.empty:
                plt.scatter(state_anomalies['created_date'], 
                           state_anomalies['fever_signal'],
                           color='red', marker='o', label='Anomalies')
        
        plt.title(f'Signal Analysis and Anomalies ({sample_state})')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.85)
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"PCA components used: {pca_result.shape[1]} (explaining 90% variance)")
        print(f"Days with anomalies: {len(anomaly_dates):,} ({(len(anomaly_dates)/total_days*100):.2f}% of time span)")
        print(f"Total anomalous observations: {len(anomalies):,} ({(len(anomalies)/len(signal_data)*100):.2f}%)")
        print(f"States showing anomalies: {anomalies['state'].nunique():,}")
        print(f"Average states per anomaly day: {daily_stats['states_affected'].mean():.1f}")
        
        return anomalies
        
        # Calculate time span of data
        date_range = signal_data['created_date'].agg(['min', 'max'])
        total_days = (date_range['max'] - date_range['min']).days
        
        # Prepare data for PCA
        pivot_data = signal_data.pivot_table(
            index='created_date', 
            columns='state', 
            values='fever_signal',
            aggfunc='mean'
        )
        # Handle missing values more explicitly
        pivot_data = pivot_data.ffill().bfill()
        
        # Apply bandpass filter to each state's signal
        fs = 1  # 1 sample per day
        lowcut = 1/14  # 14-day seasonal variations
        highcut = 1/3   # 3-day variations
        b, a = signal.butter(3, [lowcut, highcut], btype='band', fs=fs)
        
        filtered_data = pd.DataFrame(
            index=pivot_data.index,
            columns=pivot_data.columns,
            data=np.apply_along_axis(
                lambda x: signal.filtfilt(b, a, x),
                0,
                pivot_data.values
            )
        )
        
        # Apply PCA to filtered data
        pca = PCA(n_components=0.90)  # Reduce to 90% variance explained
        pca_result = pca.fit_transform(filtered_data)
        
        # Reconstruct signal using dominant components
        reconstructed = pd.DataFrame(
            pca.inverse_transform(pca_result),
            index=filtered_data.index,
            columns=filtered_data.columns
        )
        
        # Calculate residuals and their z-scores
        residuals = filtered_data - reconstructed
        zscore_residuals = pd.DataFrame(
            zscore(residuals, axis=0),
            index=residuals.index,
            columns=residuals.columns
        )
        
        # More stringent anomaly detection
        # Require both magnitude and spatial criteria
        magnitude_anomalies = (abs(zscore_residuals) > 3.5).any(axis=1)  # Increased threshold
        spatial_spread = (abs(zscore_residuals) > 2).sum(axis=1) / zscore_residuals.shape[1]
        anomaly_mask = magnitude_anomalies & (spatial_spread < 0.25)  # Limit spatial spread
        
        anomaly_dates = zscore_residuals.index[anomaly_mask]
        
        # Create final anomalies dataset
        anomalies = signal_data[signal_data['created_date'].isin(anomaly_dates)].copy()
        # Convert to datetime before converting to date
        anomalies['date'] = pd.to_datetime(anomalies['created_date']).dt.date
        
        # Additional filtering: only keep states with significant deviations
        state_dates = []
        for date in anomaly_dates:
            significant_states = zscore_residuals.loc[date].abs() > 3
            states_affected = significant_states[significant_states].index
            for state in states_affected:
                state_dates.append((pd.to_datetime(date).date(), state))
        
        state_date_pairs = pd.DataFrame(state_dates, columns=['date', 'state'])
        # Ensure both sides of the merge have the same date type
        anomalies = anomalies.merge(state_date_pairs, on=['date', 'state'])
        
        # Calculate daily statistics
        daily_stats = anomalies.groupby('date').agg({
            'state': 'nunique',
            'fever_signal': 'count'
        }).reset_index()
        daily_stats.columns = ['date', 'states_affected', 'anomaly_count']
        
        # Visualize results
        plt.figure(figsize=(15, 8))
        
        # Plot original and reconstructed signals for a sample state
        sample_state = pivot_data.columns[0]
        plt.plot(pivot_data.index, pivot_data[sample_state], 
                label='Original Signal', alpha=0.5)
        plt.plot(reconstructed.index, reconstructed[sample_state], 
                label='Reconstructed Signal', alpha=0.5)
        
        # Plot anomalies
        state_anomalies = anomalies[anomalies['state'] == sample_state]
        if not state_anomalies.empty:
            plt.scatter(state_anomalies['created_date'], state_anomalies['fever_signal'],
                       color='red', marker='o', label='Anomalies')
        
        plt.title(f'Signal Analysis and Anomalies ({sample_state})')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal')
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Plot explained variance
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        self.save_plot(plt, 'pca_variance.png')
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"PCA components used: {pca_result.shape[1]} (explaining 90% variance)")
        print(f"Days with anomalies: {len(anomaly_dates):,} ({(len(anomaly_dates)/total_days*100):.2f}% of time span)")
        print(f"Total anomalous observations: {len(anomalies):,} ({(len(anomalies)/len(signal_data)*100):.2f}%)")
        print(f"States showing anomalies: {anomalies['state'].nunique():,}")
        print(f"Average states per anomaly day: {daily_stats['states_affected'].mean():.1f}")
        
        return anomalies
        from scipy import signal
        from sklearn.decomposition import PCA
        from scipy.stats import zscore
        
        # Calculate time span of data
        date_range = signal_data['created_date'].agg(['min', 'max'])
        total_days = (date_range['max'] - date_range['min']).days
        
        # Prepare data for PCA
        pivot_data = signal_data.pivot_table(
            index='created_date', 
            columns='state', 
            values='fever_signal',
            aggfunc='mean'
        ).fillna(method='ffill').fillna(method='bfill')
        
        # Apply bandpass filter to each state's signal
        fs = 1  # 1 sample per day
        lowcut = 1/14  # 14-day seasonal variations
        highcut = 1/3   # 3-day variations
        b, a = signal.butter(3, [lowcut, highcut], btype='band', fs=fs)
        
        filtered_data = pd.DataFrame(
            index=pivot_data.index,
            columns=pivot_data.columns,
            data=np.apply_along_axis(
                lambda x: signal.filtfilt(b, a, x),
                0,
                pivot_data.values
            )
        )
        
        # Apply PCA to filtered data
        pca = PCA(n_components=0.95)  # Keep components explaining 95% of variance
        pca_result = pca.fit_transform(filtered_data)
        
        # Reconstruct signal using dominant components
        reconstructed = pd.DataFrame(
            pca.inverse_transform(pca_result),
            index=filtered_data.index,
            columns=filtered_data.columns
        )
        
        # Calculate residuals and their z-scores
        residuals = filtered_data - reconstructed
        zscore_residuals = pd.DataFrame(
            zscore(residuals, axis=0),
            index=residuals.index,
            columns=residuals.columns
        )
        
        # Identify anomalies where residuals are significant
        anomaly_mask = (abs(zscore_residuals) > 3).any(axis=1)
        anomaly_dates = zscore_residuals.index[anomaly_mask]
        
        # Create final anomalies dataset
        anomalies = signal_data[signal_data['created_date'].isin(anomaly_dates)].copy()
        anomalies['date'] = anomalies['created_date'].dt.date
        
        # Calculate daily statistics
        daily_stats = anomalies.groupby('date').agg({
            'state': 'nunique',
            'fever_signal': 'count'
        }).reset_index()
        daily_stats.columns = ['date', 'states_affected', 'anomaly_count']
        
        # Visualize results
        plt.figure(figsize=(15, 8))
        
        # Plot original and reconstructed signals for a sample state
        sample_state = pivot_data.columns[0]
        plt.plot(pivot_data.index, pivot_data[sample_state], 
                label='Original Signal', alpha=0.5)
        plt.plot(reconstructed.index, reconstructed[sample_state], 
                label='Reconstructed Signal', alpha=0.5)
        
        # Plot anomalies
        anomaly_points = pivot_data.loc[anomaly_dates, sample_state]
        plt.scatter(anomaly_points.index, anomaly_points.values,
                   color='red', marker='o', label='Anomalies')
        
        plt.title(f'Signal Analysis and Anomalies ({sample_state})')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal')
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Plot explained variance
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        self.save_plot(plt, 'pca_variance.png')
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"PCA components used: {pca_result.shape[1]} (explaining 95% variance)")
        print(f"Days with anomalies: {len(anomaly_dates):,} ({(len(anomaly_dates)/total_days*100):.2f}% of time span)")
        print(f"Total anomalous observations: {len(anomalies):,} ({(len(anomalies)/len(signal_data)*100):.2f}%)")
        print(f"States showing anomalies: {anomalies['state'].nunique():,}")
        print(f"Average states per anomaly day: {daily_stats['states_affected'].mean():.1f}")
        
        return anomalies
        # Calculate time span of data
        date_range = signal_data['created_date'].agg(['min', 'max'])
        total_days = (date_range['max'] - date_range['min']).days
        
        # Apply initial smoothing to reduce noise
        signal_data = signal_data.copy()
        signal_data['fever_signal_smooth'] = signal_data.groupby('state')['fever_signal'].transform(
            lambda x: x.ewm(span=7, min_periods=3).mean())
        
        # Calculate rolling statistics on smoothed signal
        rolling_mean = signal_data.groupby('state')['fever_signal_smooth'].transform(
            lambda x: x.rolling(window=14, min_periods=7).mean())
        rolling_std = signal_data.groupby('state')['fever_signal_smooth'].transform(
            lambda x: x.rolling(window=14, min_periods=7).std())
        
        # Define thresholds (2.5 standard deviations)
        upper_bound = rolling_mean + 2.5 * rolling_std
        lower_bound = rolling_mean - 2.5 * rolling_std
        
        # Identify initial anomalies
        anomalies = signal_data[
            (signal_data['fever_signal_smooth'] > upper_bound) |
            (signal_data['fever_signal_smooth'] < lower_bound)
        ].copy()
        
        # Add date column for grouping
        anomalies['date'] = anomalies['created_date'].dt.date
        
        # Calculate daily state counts
        daily_state_counts = anomalies.groupby('date')['state'].nunique().reset_index(name='states_affected')
        
        # Find significant dates (at least 2 states affected)
        significant_dates = daily_state_counts[daily_state_counts['states_affected'] >= 2]['date']
        
        # Get final set of anomalies
        persistent_anomalies = anomalies[anomalies['date'].isin(significant_dates)]
        
        # Visualize anomalies
        plt.figure(figsize=(15, 8))
        plt.plot(signal_data['created_date'], signal_data['fever_signal_smooth'], 
                label='Smoothed Fever Signal', alpha=0.7)
        plt.plot(signal_data['created_date'], upper_bound, 'r--', 
                label='Upper/Lower Bounds (2.5σ)', alpha=0.5)
        plt.plot(signal_data['created_date'], lower_bound, 'r--', alpha=0.5)
        
        if len(persistent_anomalies) > 0:
            plt.scatter(persistent_anomalies['created_date'], 
                       persistent_anomalies['fever_signal_smooth'],
                       color='red', marker='o', label='Significant Anomalies')
        
        plt.title('Fever Signal with Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal (%)')
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Calculate period statistics
        sig_daily_stats = daily_state_counts[daily_state_counts['date'].isin(significant_dates)]
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"Initial anomalies detected: {len(anomalies):,} ({(len(anomalies)/len(signal_data)*100):.2f}%)")
        print(f"Significant anomalies: {len(persistent_anomalies):,} ({(len(persistent_anomalies)/len(signal_data)*100):.2f}%)")
        print(f"Days with significant anomalies: {len(significant_dates):,} ({(len(significant_dates)/total_days*100):.2f}% of time span)")
        
        if len(sig_daily_stats) > 0:
            print(f"Number of states affected: {persistent_anomalies['state'].nunique():,}")
            print(f"Average states per anomaly day: {sig_daily_stats['states_affected'].mean():.1f}")
        
        return persistent_anomalies
        # Calculate time span of data
        date_range = signal_data['created_date'].agg(['min', 'max'])
        total_days = (date_range['max'] - date_range['min']).days
        
        # Apply initial smoothing to reduce noise
        signal_data = signal_data.copy()
        signal_data['fever_signal_smooth'] = signal_data.groupby('state')['fever_signal'].transform(
            lambda x: x.ewm(span=7, min_periods=3).mean())
        
        # Calculate rolling statistics on smoothed signal
        rolling_mean = signal_data.groupby('state')['fever_signal_smooth'].transform(
            lambda x: x.rolling(window=14, min_periods=7).mean())
        rolling_std = signal_data.groupby('state')['fever_signal_smooth'].transform(
            lambda x: x.rolling(window=14, min_periods=7).std())
        
        # Define thresholds (3 standard deviations)
        upper_bound = rolling_mean + 3 * rolling_std
        lower_bound = rolling_mean - 3 * rolling_std
        
        # Identify initial anomalies
        anomalies = signal_data[
            (signal_data['fever_signal_smooth'] > upper_bound) |
            (signal_data['fever_signal_smooth'] < lower_bound)
        ].copy()
        
        # Add date column for grouping
        anomalies['date'] = anomalies['created_date'].dt.date
        
        # Calculate daily anomaly statistics by state
        daily_state_stats = anomalies.groupby(['date', 'state']).size().reset_index(name='anomaly_count')
        daily_summary = daily_state_stats.groupby('date').agg({
            'state': 'nunique',  # Number of states affected
            'anomaly_count': 'sum'  # Total anomalies that day
        }).reset_index()
        daily_summary.columns = ['date', 'states_affected', 'total_anomalies']
        
        # Filter for significant days (using multiple criteria)
        significant_dates = daily_summary[
            # At least 3 states affected and meaningful number of anomalies
            (daily_summary['states_affected'] >= 3) &
            (daily_summary['total_anomalies'] >= daily_summary['states_affected'] * 2)
        ]['date']
        
        # Get final set of anomalies
        persistent_anomalies = anomalies[anomalies['date'].isin(significant_dates)]
        
        # Visualize anomalies
        plt.figure(figsize=(15, 8))
        plt.plot(signal_data['created_date'], signal_data['fever_signal_smooth'], 
                label='Smoothed Fever Signal', alpha=0.7)
        plt.plot(signal_data['created_date'], upper_bound, 'r--', 
                label='Upper Bound (3σ)', alpha=0.5)
        plt.plot(signal_data['created_date'], lower_bound, 'r--', 
                label='Lower Bound (3σ)', alpha=0.5)
        
        if len(persistent_anomalies) > 0:
            plt.scatter(persistent_anomalies['created_date'], 
                       persistent_anomalies['fever_signal_smooth'],
                       color='red', marker='o', label='Significant Anomalies')
        
        plt.title('Fever Signal with Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal (%)')
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Calculate period statistics
        sig_daily_stats = daily_summary[daily_summary['date'].isin(significant_dates)]
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"Significant anomalies detected: {len(persistent_anomalies):,} ({(len(persistent_anomalies)/len(signal_data)*100):.2f}%)")
        print(f"Days with anomalies: {len(significant_dates):,} ({(len(significant_dates)/total_days*100):.2f}% of time span)")
        print(f"States affected: {persistent_anomalies['state'].nunique():,}")
        if len(sig_daily_stats) > 0:
            print(f"Average states per anomaly day: {sig_daily_stats['states_affected'].mean():.1f}")
            print(f"Average anomalies per state per day: {(sig_daily_stats['total_anomalies']/sig_daily_stats['states_affected']).mean():.1f}")
        
        return persistent_anomalies
        # Calculate time span of data
        date_range = signal_data['created_date'].agg(['min', 'max'])
        total_days = (date_range['max'] - date_range['min']).days
        
        # Calculate rolling statistics with a longer window
        rolling_mean = signal_data.groupby(['state'])['fever_signal'].transform(
            lambda x: x.rolling(window=14, min_periods=7).mean())
        rolling_std = signal_data.groupby(['state'])['fever_signal'].transform(
            lambda x: x.rolling(window=14, min_periods=7).std())
        
        # Define primary and secondary thresholds
        primary_upper = rolling_mean + 3 * rolling_std
        primary_lower = rolling_mean - 3 * rolling_std
        secondary_upper = rolling_mean + 2.5 * rolling_std
        secondary_lower = rolling_mean - 2.5 * rolling_std
        
        # Identify primary (strong) and secondary (moderate) anomalies
        primary_anomalies = signal_data[
            (signal_data['fever_signal'] > primary_upper) |
            (signal_data['fever_signal'] < primary_lower)
        ].copy()
        
        secondary_anomalies = signal_data[
            ((signal_data['fever_signal'] > secondary_upper) & 
             (signal_data['fever_signal'] <= primary_upper)) |
            ((signal_data['fever_signal'] < secondary_lower) & 
             (signal_data['fever_signal'] >= primary_lower))
        ].copy()
        
        # Combine anomalies with type indicator
        primary_anomalies['anomaly_type'] = 'primary'
        secondary_anomalies['anomaly_type'] = 'secondary'
        all_anomalies = pd.concat([primary_anomalies, secondary_anomalies])
        all_anomalies['date'] = all_anomalies['created_date'].dt.date
        
        # Analyze daily patterns
        daily_stats = all_anomalies.groupby('date').agg({
            'state': 'nunique',
            'anomaly_type': lambda x: (x == 'primary').sum()
        }).reset_index()
        daily_stats.columns = ['date', 'states_affected', 'primary_anomalies']
        
        # Define significant days as those with either:
        # 1. At least 3 states with any anomalies, including at least 1 primary
        # 2. At least 2 states with primary anomalies
        significant_dates = daily_stats[
            ((daily_stats['states_affected'] >= 3) & (daily_stats['primary_anomalies'] >= 1)) |
            (daily_stats['primary_anomalies'] >= 2)
        ]['date']
        
        # Filter anomalies to significant dates
        persistent_anomalies = all_anomalies[all_anomalies['date'].isin(significant_dates)]
        
        # Visualize anomalies
        plt.figure(figsize=(15, 8))
        plt.plot(signal_data['created_date'], signal_data['fever_signal'], 
                label='Fever Signal', alpha=0.7)
        plt.plot(signal_data['created_date'], primary_upper, 'r--', 
                label='Primary Threshold (3σ)', alpha=0.5)
        plt.plot(signal_data['created_date'], primary_lower, 'r--', alpha=0.5)
        
        if len(persistent_anomalies) > 0:
            primary_mask = persistent_anomalies['anomaly_type'] == 'primary'
            plt.scatter(persistent_anomalies[primary_mask]['created_date'], 
                       persistent_anomalies[primary_mask]['fever_signal'],
                       color='red', marker='o', label='Primary Anomalies')
            plt.scatter(persistent_anomalies[~primary_mask]['created_date'], 
                       persistent_anomalies[~primary_mask]['fever_signal'],
                       color='orange', marker='o', label='Secondary Anomalies')
        
        plt.title('Fever Signal with Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal (%)')
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"Total anomalies detected: {len(persistent_anomalies):,} ({(len(persistent_anomalies)/len(signal_data)*100):.2f}%)")
        print(f"  - Primary anomalies: {sum(persistent_anomalies['anomaly_type'] == 'primary'):,}")
        print(f"  - Secondary anomalies: {sum(persistent_anomalies['anomaly_type'] == 'secondary'):,}")
        print(f"Days with significant anomalies: {len(significant_dates):,} ({(len(significant_dates)/total_days*100):.2f}% of time span)")
        print(f"States affected: {persistent_anomalies['state'].nunique():,}")
        print(f"Average states per anomaly day: {daily_stats[daily_stats['date'].isin(significant_dates)]['states_affected'].mean():.1f}")
        
        return persistent_anomalies
        
        # Visualize anomalies
        plt.figure(figsize=(15, 8))
        plt.plot(signal_data['created_date'], signal_data['fever_signal'], 
                label='Fever Signal', alpha=0.7)
        plt.plot(signal_data['created_date'], upper_bound, 'r--', 
                label='Upper Bound (3σ)', alpha=0.5)
        plt.plot(signal_data['created_date'], lower_bound, 'r--', 
                label='Lower Bound (3σ)', alpha=0.5)
        
        if len(persistent_anomalies) > 0:
            plt.scatter(persistent_anomalies['created_date'], persistent_anomalies['fever_signal'],
                       color='red', marker='o', label='Significant Anomalies')
        
        plt.title('Fever Signal with Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal (%)')
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Get unique anomaly dates
        unique_anomaly_dates = len(significant_dates)
        
        # Print summary statistics with time context
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"Initial anomalies detected: {len(anomalies):,} ({(len(anomalies)/len(signal_data)*100):.2f}%)")
        print(f"Persistent anomalies: {len(persistent_anomalies):,} ({(len(persistent_anomalies)/len(signal_data)*100):.2f}%)")
        print(f"Days with multi-state anomalies: {unique_anomaly_dates:,} ({(unique_anomaly_dates/total_days*100):.2f}% of time span)")
        if len(persistent_anomalies) > 0:
            print(f"Number of states with anomalies: {persistent_anomalies['state'].nunique():,}")
            print(f"Average states affected per anomaly day: {daily_state_counts['states_affected'].mean():.1f}")
        
        return persistent_anomalies
        
        # Filter to keep only anomalies from significant dates
        persistent_anomalies = anomalies[anomalies['date'].isin(significant_dates)]
        
        # Visualize anomalies
        plt.figure(figsize=(15, 8))
        plt.plot(signal_data['created_date'], signal_data['fever_signal'], 
                label='Fever Signal', alpha=0.7)
        plt.plot(signal_data['created_date'], upper_bound, 'r--', 
                label='Upper Bound (3.5σ)', alpha=0.5)
        plt.plot(signal_data['created_date'], lower_bound, 'r--', 
                label='Lower Bound (3.5σ)', alpha=0.5)
        
        if len(persistent_anomalies) > 0:
            plt.scatter(persistent_anomalies['created_date'], persistent_anomalies['fever_signal'],
                       color='red', marker='o', label='Significant Anomalies')
        
        plt.title('Fever Signal with Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal (%)')
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, 'anomalies.png')
        
        # Get unique anomaly dates
        unique_anomaly_dates = len(significant_dates)
        
        # Print summary statistics with time context
        print("\nAnomaly Detection Summary:")
        print(f"Data time span: {date_range['min'].date()} to {date_range['max'].date()} ({total_days:,} days)")
        print(f"Total observations: {len(signal_data):,}")
        print(f"Initial anomalies detected: {len(anomalies):,} ({(len(anomalies)/len(signal_data)*100):.2f}%)")
        print(f"Persistent anomalies: {len(persistent_anomalies):,} ({(len(persistent_anomalies)/len(signal_data)*100):.2f}%)")
        print(f"Days with multi-state anomalies: {unique_anomaly_dates:,} ({(unique_anomaly_dates/total_days*100):.2f}% of time span)")
        if len(persistent_anomalies) > 0:
            print(f"Number of states with anomalies: {persistent_anomalies['state'].nunique():,}")
            print(f"Average states affected per anomaly day: {daily_state_counts['states_affected'].mean():.1f}")
        
        return persistent_anomalies

    def visualize_inflection_points(self, signal_data):
        """Visualize significant inflection points in the data"""
        plt.figure(figsize=(15, 8))
        
        # Plot the main signal
        plt.plot(signal_data['created_date'], signal_data['fever_signal'], label='Fever Signal')
        
        # Find and plot inflection points
        signal = signal_data['fever_signal'].values
        peaks = self.find_inflection_points(signal)
        
        # Plot inflection points
        plt.scatter(signal_data['created_date'].iloc[peaks], 
                   signal_data['fever_signal'].iloc[peaks],
                   color='red', marker='o', label='Inflection Points')
        
        plt.title('Fever Signal with Significant Inflection Points')
        plt.xlabel('Date')
        plt.ylabel('Fever Signal (%)')
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, 'inflection_points.png')

    def find_inflection_points(self, signal, min_distance=30, min_prominence=1.0):
        """Identify significant inflection points using peak and valley detection"""
        from scipy.signal import find_peaks
        
        # Find peaks with increased prominence and distance requirements
        peak_indices, _ = find_peaks(signal, distance=min_distance, prominence=min_prominence)
        
        # Find valleys (invert signal for valley detection)
        valley_indices, _ = find_peaks(-signal, distance=min_distance, prominence=min_prominence)
        
        # Combine peaks and valleys
        all_points = np.sort(np.concatenate([peak_indices, valley_indices]))
        
        return all_points

    def perform_spatio_temporal_analysis(self):
        """Perform spatio-temporal analysis using county and state information"""
        # Group data by location and time
        spatial_temporal = self.df.groupby(['state', 'county', 'created_date'])['fever'].mean().reset_index()
        
        # Create temporal bins (e.g., weekly)
        spatial_temporal['week'] = pd.to_datetime(spatial_temporal['created_date']).dt.isocalendar().week
        
        # Calculate average fever rates by location and week
        weekly_rates = spatial_temporal.groupby(['state', 'county', 'week'])['fever'].mean().reset_index()
        
        # Calculate temporal correlation between counties
        correlation_matrix = {}
        for state in weekly_rates['state'].unique():
            state_data = weekly_rates[weekly_rates['state'] == state]
            pivot_table = state_data.pivot(index='week', columns='county', values='fever')
            if not pivot_table.empty and pivot_table.shape[1] > 1:  # Check if we have enough data
                correlation_matrix[state] = pivot_table.corr()
        
        # Plot temporal patterns by state
        plt.figure(figsize=(15, 8))
        states_to_plot = weekly_rates['state'].value_counts().nlargest(10).index  # Plot top 10 states
        for state in states_to_plot:
            state_data = weekly_rates[weekly_rates['state'] == state]
            state_mean = state_data.groupby('week')['fever'].mean()
            plt.plot(state_mean.index, state_mean.values, label=state, alpha=0.7)
            
        plt.title('Weekly Fever Rates by State (Top 10 States)')
        plt.xlabel('Week of Year')
        plt.ylabel('Average Fever Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        self.save_plot(plt, 'temporal_patterns.png')
        
        # Analyze spatial diffusion
        self.analyze_spatial_diffusion(spatial_temporal)
        
        return {
            'weekly_rates': weekly_rates,
            'correlation_matrix': correlation_matrix
        }

    def analyze_spatial_diffusion(self, spatial_temporal):
        """Analyze how fever patterns spread across space and time"""
        # Create time windows
        spatial_temporal['month'] = pd.to_datetime(spatial_temporal['created_date']).dt.month
        
        # Calculate monthly peaks for each location
        peak_timing = spatial_temporal.groupby(['state', 'county', 'month'])['fever'].mean().reset_index()
        peak_timing = peak_timing.sort_values('fever', ascending=False).groupby(['state', 'county']).first().reset_index()
        
        # Plot peak timing distribution
        plt.figure(figsize=(15, 8))
        for state in peak_timing['state'].unique():
            state_data = peak_timing[peak_timing['state'] == state]
            plt.hist(state_data['month'], alpha=0.5, label=state, bins=12)
            
        plt.title('Distribution of Peak Fever Timing by State')
        plt.xlabel('Month')
        plt.ylabel('Number of Counties')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        self.save_plot(plt, 'spatial_diffusion.png')

    def comprehensive_analysis(self):
        """Perform comprehensive analysis addressing all requirements"""
        # Part 1: Demographic Analysis
        print("\nPart 1: Demographic Analysis")
        demographic_trends = self.analyze_demographics()
        
        # Part 2: Signal Creation
        print("\nPart 2: Creating Illness Signals")
        # National signal
        national_signal = self.create_national_signal()
        print("\nDenominator Methodology:")
        print("- Using 7-day rolling average of total readings as denominator")
        print("- This accounts for temporal variations in device usage")
        print("- Smoothed using EWMA to reduce noise while preserving trends")
        
        # Age group signals
        age_signals = self.create_age_group_signals()
        
        # County signals for CA and TX
        county_signals = self.create_county_signals(['CA', 'TX'])
        
        # Create visualizations and analysis
        self.plot_signal_analysis(national_signal)  # FFT and inflection points
        
        # Spline fit
        self.plot_spline_fit(national_signal)  # Make sure this gets called
        
        # Analyze anomalies
        anomalies = self.detect_anomalies(national_signal)
        
        # Create year over year comparison
        self.plot_year_over_year(national_signal, 
                               "National Fever Signal - Year over Year (July-June)")
        
        # Geospatial Analysis
        print("\nPart 3: Geospatial Analysis")
        regional_patterns = self.analyze_regional_patterns()
        county_patterns = self.analyze_county_patterns(['CA', 'TX'])
        
        # Spatio-temporal Analysis
        print("\nPart 4: Spatio-temporal Analysis")
        st_analysis = self.perform_spatio_temporal_analysis()
        
        return {
            'demographic_trends': demographic_trends,
            'national_signal': national_signal,
            'age_signals': age_signals,
            'county_signals': county_signals,
            'anomalies': anomalies,
            'regional_patterns': regional_patterns,
            'county_patterns': county_patterns,
            'spatio_temporal_analysis': st_analysis
        }
        """Perform comprehensive analysis addressing all requirements"""
        # Part 1: Demographic Analysis
        print("\nPart 1: Demographic Analysis")
        demographic_trends = self.analyze_demographics()
        
        # Part 2: Signal Creation
        print("\nPart 2: Creating Illness Signals")
        # National signal
        national_signal = self.create_national_signal()
        print("\nDenominator Methodology:")
        print("- Using 7-day rolling average of total readings as denominator")
        print("- This accounts for temporal variations in device usage")
        print("- Smoothed using EWMA to reduce noise while preserving trends")
        
        # Age group signals
        age_signals = self.create_age_group_signals()
        
        # County signals for CA and TX
        county_signals = self.create_county_signals(['CA', 'TX'])
        
        # In comprehensive_analysis method, replace the inflection point plotting with:
        self.plot_signal_analysis(national_signal)
        
        # Analyze inflection points and detect anomalies
        anomalies = self.detect_anomalies(national_signal)
        
        # Create year over year comparison
        self.plot_year_over_year(national_signal, 
                               "National Fever Signal - Year over Year (July-June)")
        
        # Geospatial Analysis
        print("\nPart 3: Geospatial Analysis")
        regional_patterns = self.analyze_regional_patterns()
        county_patterns = self.analyze_county_patterns(['CA', 'TX'])
        
        # Spatio-temporal Analysis
        print("\nPart 4: Spatio-temporal Analysis")
        st_analysis = self.perform_spatio_temporal_analysis()
        
        return {
            'demographic_trends': demographic_trends,
            'national_signal': national_signal,
            'age_signals': age_signals,
            'county_signals': county_signals,
            'anomalies': anomalies,
            'regional_patterns': regional_patterns,
            'county_patterns': county_patterns,
            'spatio_temporal_analysis': st_analysis
        }
        # Part 1: Demographic Analysis
        print("\nPart 1: Demographic Analysis")
        demographic_trends = self.analyze_demographics()
        
        # Part 2: Signal Creation and Analysis
        print("\nPart 2: Creating Illness Signals")
        # National signal
        national_signal = self.create_national_signal()
        print("\nDenominator Methodology:")
        print("- Using 7-day rolling average of total readings as denominator")
        print("- This accounts for temporal variations in device usage")
        print("- Smoothed using EWMA to reduce noise while preserving trends")
        
        # Age group signals
        age_signals = self.create_age_group_signals()
        
        # County signals for CA and TX
        county_signals = self.create_county_signals(['CA', 'TX'])
        
        # Special analyses: anomalies, inflection points, and PCA
        anomalies = self.detect_anomalies(national_signal)
        
        # Plot inflection points separately
        days_elapsed = (national_signal['created_date'] - national_signal['created_date'].min()).dt.days.values
        signal_mean = national_signal.groupby('created_date')['fever_signal'].mean()
        peaks = self.find_inflection_points(signal_mean.values)
        
        plt.figure(figsize=(15, 8))
        plt.plot(signal_mean.index, signal_mean.values, label='Mean Fever Signal')
        plt.scatter(signal_mean.index[peaks], signal_mean.values[peaks], 
                   color='red', marker='o', label='Inflection Points')
        plt.title('Signal Inflection Points')
        plt.xlabel('Date')
        plt.ylabel('Mean Fever Signal')
        plt.legend()
        plt.grid(True)
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.95)
        self.save_plot(plt, 'inflection_points.png')
        
        # Create year over year comparison
        self.plot_year_over_year(national_signal, 
                               "National Fever Signal - Year over Year (July-June)")
        
        # Geospatial Analysis
        print("\nPart 3: Geospatial Analysis")
        regional_patterns = self.analyze_regional_patterns()
        county_patterns = self.analyze_county_patterns(['CA', 'TX'])
        
        # Spatio-temporal Analysis
        print("\nPart 4: Spatio-temporal Analysis")
        st_analysis = self.perform_spatio_temporal_analysis()
        
        return {
            'demographic_trends': demographic_trends,
            'national_signal': national_signal,
            'age_signals': age_signals,
            'county_signals': county_signals,
            'anomalies': anomalies,
            'regional_patterns': regional_patterns,
            'county_patterns': county_patterns,
            'spatio_temporal_analysis': st_analysis
        }
        # Part 1: Demographic Analysis
        print("\nPart 1: Demographic Analysis")
        demographic_trends = self.analyze_demographics()
        
        # Part 2: Signal Creation
        print("\nPart 2: Creating Illness Signals")
        # National signal
        national_signal = self.create_national_signal()
        print("\nDenominator Methodology:")
        print("- Using 7-day rolling average of total readings as denominator")
        print("- This accounts for temporal variations in device usage")
        print("- Smoothed using EWMA to reduce noise while preserving trends")
        
        # Age group signals
        age_signals = self.create_age_group_signals()
        
        # County signals for CA and TX
        county_signals = self.create_county_signals(['California', 'Texas'])
        
        # Analyze inflection points
        self.visualize_inflection_points(national_signal)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(national_signal)
        print(f"\nAnomalies detected: {len(anomalies)}")
        
        # Create year over year comparison
        self.plot_year_over_year(national_signal, 
                               "National Fever Signal - Year over Year (July-June)")
        
        # Geospatial Analysis
        print("\nPart 3: Geospatial Analysis")
        regional_patterns = self.analyze_regional_patterns()
        county_patterns = self.analyze_county_patterns(['California', 'Texas'])
        
        # Spatio-temporal Analysis
        print("\nPart 4: Spatio-temporal Analysis")
        st_analysis = self.perform_spatio_temporal_analysis()
        
        # Return comprehensive results
        return {
            'demographic_trends': demographic_trends,
            'national_signal': national_signal,
            'age_signals': age_signals,
            'county_signals': county_signals,
            'anomalies': anomalies,
            'regional_patterns': regional_patterns,
            'county_patterns': county_patterns,
            'spatio_temporal_analysis': st_analysis
        }

# Main execution block
if __name__ == "__main__":
    # Initialize analyzer with correct file path
    file_path = '/Users/joelskaria/Desktop/Python_WF/Kinsa_CaseStudy/Masked and Summarized Kinsa Fever.csv'
    analyzer = KinsaAnalyzer(file_path)
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis()
    
    print("\nAnalysis complete. Check the 'plots' directory for visualizations.")

    # # Navigate to project directory
    # cd /Users/joelskaria/Desktop/Python_WF/Kinsa_CaseStudy/
    # # Run script
    # python KinsaAnalyzer_JoelSkaria.py