import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
import chess.svg
import io
import time

class MCTSVisualizer:
    def __init__(self):
        self.node_positions = {}
        
    def create_mcts_graph(self, root_node, max_depth=3):
        """Convert MCTS tree to NetworkX graph for visualization"""
        G = nx.DiGraph()
        
        def add_nodes_recursive(node, depth=0, parent_id=None, child_num=0):
            if depth > max_depth:
                return
                
            # Create unique node ID
            node_id = f"{chess.square_name(node.board.king(True))}-{depth}-{child_num}"
            
            # Node attributes
            visits = node.visit_count if node.visit_count > 0 else 'Root'
            value = f"{node.value():.3f}" if node.visit_count > 0 else 'N/A'
            
            # Add node to graph
            G.add_node(node_id, 
                      visits=visits,
                      value=value,
                      fen=node.board.fen(),
                      depth=depth)
            
            # Add edge from parent
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            
            # Add children
            for i, (move, child) in enumerate(node.children.items()):
                add_nodes_recursive(child, depth + 1, node_id, i)
                
        add_nodes_recursive(root_node)
        return G

    def draw_mcts_tree(self, G, current_board=None):
        """Draw MCTS tree with chess board positions"""
        plt.figure(figsize=(20, 12))
        
        # Create main tree layout
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node in G.nodes:
            visits = G.nodes[node]['visits']
            value = G.nodes[node]['value']
            
            # Node size based on visits
            size = 3000 if visits == 'Root' else min(visits * 100 + 1000, 5000)
            node_sizes.append(size)
            
            # Node color based on value
            if value == 'N/A':
                color = 'lightgray'
            else:
                # Convert string value to float
                v = float(value)
                color = plt.cm.RdYlGn((v + 1) / 2)  # Map [-1,1] to [0,1]
            node_colors.append(color)
            
            # Node label
            labels[node] = f'Visits: {visits}\nValue: {value}'
        
        # Draw the graph
        nx.draw(G, pos,
                node_color=node_colors,
                node_size=node_sizes,
                labels=labels,
                with_labels=True,
                arrows=True,
                edge_color='gray',
                font_size=8)
        
        # Add chess boards for selected nodes
        if current_board:
            fig = plt.gcf()
            ax_board = fig.add_axes([0.8, 0.6, 0.2, 0.2])
            ax_board.axis('off')
            board_svg = chess.svg.board(current_board)
            ax_board.imshow(self.svg_to_png(board_svg))
        
        plt.title("Monte Carlo Tree Search Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def svg_to_png(svg_string):
        """Convert SVG to PNG for matplotlib"""
        from cairosvg import svg2png
        import numpy as np
        from PIL import Image
        
        png_data = svg2png(bytestring=svg_string.encode())
        img = Image.open(io.BytesIO(png_data))
        return np.array(img)

# Example usage in a Jupyter notebook:
if __name__ == '__main__':
    # Initialize visualizer
    visualizer = MCTSVisualizer()

    # During MCTS search:
    def mcts_search_with_viz(self, board):
        root = Node(board)
        
        for i in range(self.num_simulations):
            # Regular MCTS logic...
            
            # Visualize every N iterations
            if i % 50 == 0:
                clear_output(wait=True)
                G = visualizer.create_mcts_graph(root)
                visualizer.draw_mcts_tree(G, board)
                display(HTML(f"<p>Simulation {i}/{self.num_simulations}</p>"))
                time.sleep(0.5)
        
        return root

