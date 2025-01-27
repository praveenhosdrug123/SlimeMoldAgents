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
from utils.debug_utils import DebugUtils

class SlimeMoldVisualizer:
    def __init__(self):
        self.debug = DebugUtils()
        self.logger = self.debug.get_logger(__name__)
        self.logger.info("Initializing SlimeMoldVisualizer")
        
        self.solution_history = []
        self.metrics_history = defaultdict(list)
        self.timestamps = []
        self.logger.debug("Initialized empty history containers")

    @DebugUtils.trace_function
    def add_solution(self, path: List[str], score: float, edge_weights: Dict[str, Dict[str, float]]):
        """Record a solution and its metrics"""
        try:
            self.logger.debug(f"Adding solution - Path length: {len(path)}, Score: {score}")
            timestamp = datetime.now()
            
            self.solution_history.append({
                'timestamp': timestamp,
                'path': path,
                'score': score,
                'edge_weights': edge_weights.copy()
            })
            self.timestamps.append(timestamp)
            
            metrics = self._calculate_metrics(edge_weights, path, score)
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
                
            self.logger.debug(f"Solution added successfully. Current history size: {len(self.solution_history)}")
            
        except Exception as e:
            self.logger.error(f"Error adding solution: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _calculate_metrics(self, edge_weights: Dict[str, Dict[str, float]], 
                         path: List[str], score: float) -> Dict[str, float]:
        """Calculate network metrics"""
        try:
            self.logger.debug("Calculating network metrics")
            G = nx.DiGraph(edge_weights)
            
            metrics = {}
            
            # Calculate network metrics with error handling
            try:
                metrics['average_path_length'] = nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else float('inf')
                self.logger.debug(f"Average path length: {metrics['average_path_length']}")
            except Exception as e:
                self.logger.warning(f"Error calculating average path length: {str(e)}")
                metrics['average_path_length'] = float('inf')
            
            metrics['network_density'] = nx.density(G)
            
            weights = [w for d in edge_weights.values() for w in d.values()]
            metrics['average_edge_weight'] = np.mean(weights)
            metrics['edge_weight_std'] = np.std(weights)
            metrics['solution_score'] = score
            
            self.logger.debug(f"Metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in metric calculation: {str(e)}")
            raise

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=1000)
    def plot_response_graph(self, edge_weights: Dict[str, Dict[str, float]], 
                          node_usage: Dict[str, int], figsize=(12, 8)):
        """Plot current state of response graph"""
        try:
            self.logger.debug("Plotting response graph")
            plt.figure(figsize=figsize)
            G = nx.DiGraph(edge_weights)
            
            self.logger.debug(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            sizes = [1000 * (node_usage.get(node, 1)) for node in G.nodes()]
            pos = nx.spring_layout(G)
            
            self._draw_graph_elements(G, pos, sizes)
            
            plt.title("Response Graph Visualization")
            plt.axis('off')
            self.logger.info("Response graph plotted successfully")
            return plt
            
        except Exception as e:
            self.logger.error(f"Error plotting response graph: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _draw_graph_elements(self, G, pos, sizes):
        """Helper method to draw graph elements"""
        try:
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=sizes, 
                                 node_color='lightblue', alpha=0.7)
            
            # Draw edges
            edges = G.edges()
            weights = [G[u][v]['weight'] * 3 for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray',
                                 alpha=0.6, arrows=True)
            
            # Add labels
            nx.draw_networkx_labels(G, pos)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            self.logger.debug("Graph elements drawn successfully")
            
        except Exception as e:
            self.logger.error(f"Error drawing graph elements: {str(e)}")
            raise

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=1500)
    def plot_solution_history(self, figsize=(15, 10)):
        """Plot temporal evolution of solutions"""
        try:
            if not self.solution_history:
                self.logger.warning("No solution history available for plotting")
                return None
                
            self.logger.debug(f"Plotting solution history with {len(self.solution_history)} entries")
            
            fig = make_subplots(rows=2, cols=2,
                               subplot_titles=('Solution Scores Over Time',
                                             'Path Length Distribution',
                                             'Edge Weight Evolution',
                                             'Network Metrics'))
            
            self._add_solution_plots(fig)
            
            fig.update_layout(height=800, showlegend=True)
            self.logger.info("Solution history plotted successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting solution history: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_solution_plots(self, fig):
        """Helper method to add individual plots to the figure"""
        try:
            # Add each plot with separate error handling
            self._add_scores_plot(fig)
            self._add_path_length_plot(fig)
            self._add_edge_weight_plot(fig)
            self._add_network_metrics_plot(fig)
            
        except Exception as e:
            self.logger.error(f"Error adding solution plots: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_scores_plot(self, fig):
        """Add solution scores plot"""
        try:
            scores = [sol['score'] for sol in self.solution_history]
            self.logger.debug(f"Adding scores plot with {len(scores)} points")
            
            fig.add_trace(
                go.Scatter(x=self.timestamps, y=scores, name="Solution Score"),
                row=1, col=1
            )
            
        except Exception as e:
            self.logger.error(f"Error adding scores plot: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_path_length_plot(self, fig):
        """Add path length distribution plot"""
        try:
            path_lengths = [len(sol['path']) for sol in self.solution_history]
            self.logger.debug(f"Adding path length plot. Length range: [{min(path_lengths)}, {max(path_lengths)}]")
            
            fig.add_trace(
                go.Histogram(x=path_lengths, name="Path Lengths"),
                row=1, col=2
            )
            
        except Exception as e:
            self.logger.error(f"Error adding path length plot: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_edge_weight_plot(self, fig):
        """Add edge weight evolution plot"""
        try:
            self.logger.debug("Adding edge weight evolution plot")
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
            
        except Exception as e:
            self.logger.error(f"Error adding edge weight plot: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_network_metrics_plot(self, fig):
        """Add network metrics plot"""
        try:
            self.logger.debug("Adding network metrics plot")
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
            
        except Exception as e:
            self.logger.error(f"Error adding network metrics plot: {str(e)}")
            raise

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=1000)
    def plot_performance_metrics(self, window_size=10):
        """Plot key performance metrics"""
        try:
            if not self.metrics_history:
                self.logger.warning("No metrics history available for plotting")
                return None
                
            self.logger.debug(f"Plotting performance metrics with window size {window_size}")
            
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(3, 2)
            
            self._add_performance_subplot(gs, window_size)
            
            plt.tight_layout()
            self.logger.info("Performance metrics plotted successfully")
            return plt
            
        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_performance_subplot(self, gs, window_size):
        """Add subplots for performance metrics"""
        try:
            self.logger.debug("Adding performance subplots")
            
            # Network Density Plot
            self._add_density_plot(gs, window_size)
            
            # Average Path Length Plot
            self._add_path_length_subplot(gs, window_size)
            
            # Solution Success Rate Plot
            self._add_success_rate_subplot(gs, window_size)
            
            # Edge Weight Distribution Plot
            self._add_weight_distribution_subplot(gs)
            
        except Exception as e:
            self.logger.error(f"Error adding performance subplots: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _calculate_moving_average(self, data, window):
        """Calculate moving average with logging"""
        try:
            self.logger.debug(f"Calculating moving average with window size {window}")
            return pd.Series(data).rolling(window=window, min_periods=1).mean()
        except Exception as e:
            self.logger.error(f"Error calculating moving average: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_density_plot(self, gs, window_size):
        """Add network density subplot"""
        try:
            ax1 = plt.subplot(gs[0, 0])
            density_ma = self._calculate_moving_average(self.metrics_history['network_density'], window_size)
            ax1.plot(density_ma, label='Moving Average')
            ax1.plot(self.metrics_history['network_density'], alpha=0.3, label='Raw')
            ax1.set_title('Network Density')
            ax1.legend()
            self.logger.debug("Network density subplot added")
        except Exception as e:
            self.logger.error(f"Error adding density plot: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_path_length_subplot(self, gs, window_size):
        """Add path length subplot"""
        try:
            ax2 = plt.subplot(gs[0, 1])
            path_length_ma = self._calculate_moving_average(
                self.metrics_history['average_path_length'], window_size)
            ax2.plot(path_length_ma, label='Moving Average')
            ax2.plot(self.metrics_history['average_path_length'], alpha=0.3, label='Raw')
            ax2.set_title('Average Path Length')
            ax2.legend()
            self.logger.debug("Path length subplot added")
        except Exception as e:
            self.logger.error(f"Error adding path length subplot: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_success_rate_subplot(self, gs, window_size):
        """Add success rate subplot"""
        try:
            ax3 = plt.subplot(gs[1, :])
            success_rate = self._calculate_moving_average(
                [s['score'] for s in self.solution_history], window_size)
            ax3.plot(success_rate)
            ax3.set_title('Solution Success Rate (Moving Average)')
            self.logger.debug("Success rate subplot added")
        except Exception as e:
            self.logger.error(f"Error adding success rate subplot: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_weight_distribution_subplot(self, gs):
        """Add weight distribution subplot"""
        try:
            ax4 = plt.subplot(gs[2, :])
            latest_weights = [w for d in self.solution_history[-1]['edge_weights'].values() 
                            for w in d.values()]
            self.logger.debug(f"Plotting weight distribution with {len(latest_weights)} weights")
            sns.histplot(data=latest_weights, ax=ax4)
            ax4.set_title('Current Edge Weight Distribution')
        except Exception as e:
            self.logger.error(f"Error adding weight distribution subplot: {str(e)}")
            raise

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=500)
    def generate_metrics_report(self) -> pd.DataFrame:
        """Generate a dataframe with key metrics over time"""
        try:
            if not self.metrics_history:
                self.logger.warning("No metrics history available for report generation")
                return pd.DataFrame()
                
            self.logger.debug("Generating metrics report")
            
            df = pd.DataFrame({
                'timestamp': self.timestamps,
                'network_density': self.metrics_history['network_density'],
                'avg_path_length': self.metrics_history['average_path_length'],
                'avg_edge_weight': self.metrics_history['average_edge_weight'],
                'edge_weight_std': self.metrics_history['edge_weight_std'],
                'solution_score': self.metrics_history['solution_score']
            })
            
            self.logger.info(f"Metrics report generated with {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating metrics report: {str(e)}")
            raise

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=1000)
    def plot_interactive_network(self, edge_weights: Dict[str, Dict[str, float]], 
                               node_usage: Dict[str, int]):
        """Create interactive network visualization using plotly"""
        try:
            self.logger.debug("Creating interactive network visualization")
            G = nx.DiGraph(edge_weights)
            pos = nx.spring_layout(G)
            
            self.logger.debug(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            edge_traces = self._create_edge_traces(G, pos)
            node_traces = self._create_node_traces(G, pos, node_usage)
            
            fig = self._create_network_figure(edge_traces, node_traces)
            
            self.logger.info("Interactive network visualization created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating interactive network visualization: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _create_edge_traces(self, G, pos):
        """Create edge traces for interactive visualization"""
        try:
            self.logger.debug("Creating edge traces")
            edge_x, edge_y = [], []
            edge_text = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                weight = G[edge[0]][edge[1]]['weight']
                edge_text.append(f'{edge[0]} → {edge[1]}: {weight:.2f}')
            
            return go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='text',
                text=edge_text,
                mode='lines')
                
        except Exception as e:
            self.logger.error(f"Error creating edge traces: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _create_node_traces(self, G, pos, node_usage):
        """Create node traces for interactive visualization"""
        try:
            self.logger.debug("Creating node traces")
            node_x, node_y = [], []
            node_text = []
            node_sizes = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                usage = node_usage.get(node, 0)
                node_text.append(f'{node}<br>Usage: {usage}')
                node_sizes.append(usage * 20)
            
            return go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    size=node_sizes,
                    colorscale='YlGnBu',
                    line_width=2))
                
        except Exception as e:
            self.logger.error(f"Error creating node traces: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _create_network_figure(self, edge_traces, node_traces):
        """Create the final network figure"""
        try:
            self.logger.debug("Creating network figure")
            return go.Figure(
                data=[edge_traces, node_traces],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    title='Interactive Network Visualization',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )
            
        except Exception as e:
            self.logger.error(f"Error creating network figure: {str(e)}")
            raise
