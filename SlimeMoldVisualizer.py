import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SlimeMoldVisualizer:
    def __init__(self):
        self.solution_history = []
        self.metrics_history = defaultdict(list)
        self.timestamps = []
        
    def add_solution(self, path: List[str], score: float, edge_weights: Dict[str, Dict[str, float]]):
        """Record a solution and its metrics"""
        timestamp = datetime.now()
        self.solution_history.append({
            'timestamp': timestamp,
            'path': path,
            'score': score,
            'edge_weights': edge_weights.copy()
        })
        self.timestamps.append(timestamp)
        
        # Calculate and store metrics
        metrics = self._calculate_metrics(edge_weights, path, score)
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
    def _calculate_metrics(self, edge_weights: Dict[str, Dict[str, float]], 
                         path: List[str], score: float) -> Dict[str, float]:
        """Calculate network metrics"""
        G = nx.DiGraph(edge_weights)
        
        avg_path_length = nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else float('inf')
        density = nx.density(G)
        
        # Calculate edge weight stats
        weights = [w for d in edge_weights.values() for w in d.values()]
        avg_weight = np.mean(weights)
        weight_std = np.std(weights)
        
        return {
            'average_path_length': avg_path_length,
            'network_density': density,
            'average_edge_weight': avg_weight,
            'edge_weight_std': weight_std,
            'solution_score': score
        }

    def plot_response_graph(self, edge_weights: Dict[str, Dict[str, float]], 
                          node_usage: Dict[str, int], figsize=(12, 8)):
        """Plot current state of response graph"""
        plt.figure(figsize=figsize)
        G = nx.DiGraph(edge_weights)
        
        # Node sizes based on usage
        sizes = [1000 * (node_usage.get(node, 1)) for node in G.nodes()]
        
        # Create layout
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=sizes, 
                             node_color='lightblue', alpha=0.7)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray',
                             alpha=0.6, arrows=True)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Response Graph Visualization")
        plt.axis('off')
        return plt

    def plot_solution_history(self, figsize=(15, 10)):
        """Plot temporal evolution of solutions"""
        if not self.solution_history:
            return None
            
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Solution Scores Over Time',
                                         'Path Length Distribution',
                                         'Edge Weight Evolution',
                                         'Network Metrics'))
        
        # Solution scores
        scores = [sol['score'] for sol in self.solution_history]
        fig.add_trace(
            go.Scatter(x=self.timestamps, y=scores, name="Solution Score"),
            row=1, col=1
        )
        
        # Path length distribution
        path_lengths = [len(sol['path']) for sol in self.solution_history]
        fig.add_trace(
            go.Histogram(x=path_lengths, name="Path Lengths"),
            row=1, col=2
        )
        
        # Edge weight evolution
        avg_weights = self.metrics_history['average_edge_weight']
        std_weights = self.metrics_history['edge_weight_std']
        fig.add_trace(
            go.Scatter(x=self.timestamps, y=avg_weights, 
                      name="Avg Edge Weight",
                      line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.timestamps, y=std_weights,
                      name="Edge Weight Std",
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # Network metrics
        fig.add_trace(
            go.Scatter(x=self.timestamps, 
                      y=self.metrics_history['network_density'],
                      name="Network Density"),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.timestamps,
                      y=self.metrics_history['average_path_length'],
                      name="Avg Path Length"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        return fig

    def plot_performance_metrics(self, window_size=10):
        """Plot key performance metrics"""
        if not self.metrics_history:
            return None
            
        # Calculate moving averages
        def moving_average(data, window):
            return pd.Series(data).rolling(window=window, min_periods=1).mean()
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Network Density Plot
        ax1 = fig.add_subplot(gs[0, 0])
        density_ma = moving_average(self.metrics_history['network_density'], window_size)
        ax1.plot(density_ma, label='Moving Average')
        ax1.plot(self.metrics_history['network_density'], alpha=0.3, label='Raw')
        ax1.set_title('Network Density')
        ax1.legend()
        
        # Average Path Length Plot
        ax2 = fig.add_subplot(gs[0, 1])
        path_length_ma = moving_average(self.metrics_history['average_path_length'], window_size)
        ax2.plot(path_length_ma, label='Moving Average')
        ax2.plot(self.metrics_history['average_path_length'], alpha=0.3, label='Raw')
        ax2.set_title('Average Path Length')
        ax2.legend()
        
        # Solution Success Rate Plot
        ax3 = fig.add_subplot(gs[1, :])
        success_rate = moving_average([s['score'] for s in self.solution_history], window_size)
        ax3.plot(success_rate)
        ax3.set_title('Solution Success Rate (Moving Average)')
        
        # Edge Weight Distribution Plot
        ax4 = fig.add_subplot(gs[2, :])
        latest_weights = [w for d in self.solution_history[-1]['edge_weights'].values() 
                         for w in d.values()]
        sns.histplot(data=latest_weights, ax=ax4)
        ax4.set_title('Current Edge Weight Distribution')
        
        plt.tight_layout()
        return plt

    def generate_metrics_report(self) -> pd.DataFrame:
        """Generate a dataframe with key metrics over time"""
        if not self.metrics_history:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'network_density': self.metrics_history['network_density'],
            'avg_path_length': self.metrics_history['average_path_length'],
            'avg_edge_weight': self.metrics_history['average_edge_weight'],
            'edge_weight_std': self.metrics_history['edge_weight_std'],
            'solution_score': self.metrics_history['solution_score']
        })
        
        return df

    def plot_interactive_network(self, edge_weights: Dict[str, Dict[str, float]], 
                               node_usage: Dict[str, int]):
        """Create interactive network visualization using plotly"""
        G = nx.DiGraph(edge_weights)
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_text.append(f'{edge[0]} → {edge[1]}: {weight:.2f}')
        
        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            usage = node_usage.get(node, 0)
            node_text.append(f'{node}<br>Usage: {usage}')
            node_sizes.append(usage * 20)
        
        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                size=node_sizes,
                colorscale='YlGnBu',
                line_width=2))
        
        fig = go.Figure(data=[edges_trace, nodes_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           title='Interactive Network Visualization',
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
