import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Optional
import time

from indexing import move_to_index

class CurriculumManager:
    def __init__(self):
        self.games_played = 0
        self.resign_threshold = -0.9  # Resign if value prediction below this
        self.resign_counter_threshold = 10  # Number of moves value must stay below threshold
        self.temperature_drop_move = 30  # When to switch from exploration to exploitation
        self.max_moves = 300 # Due to compute constraints only calculate 300 moves
        
    def update_curriculum(self):
        """Just track games played"""
        self.games_played += 1

    def get_position_value(self, board: chess.Board) -> float:
        """
        Get position value from current player's perspective.
        Only terminal positions have definitive values:
        1 for win, -1 for loss, 0 for draw
        """
        if board.is_game_over():
            if board.is_checkmate():
                return -1  # Current player lost (opponent checkmated them)
            return 0  # Draw (stalemate, repetition, insufficient material, etc.)
            
        return 0  # Non-terminal positions don't have a defined value

    def should_resign(self, value_history):
        """Check if position is hopeless based on recent neural network evaluations"""
        if len(value_history) < self.resign_counter_threshold:
            return False
        
        recent_values = value_history[-self.resign_counter_threshold:]
        return all(v < self.resign_threshold for v in recent_values)

    def get_temperature(self, move_number):
        """Temperature schedule for move selection"""
        if move_number < self.temperature_drop_move:
            return 1.0  # High temperature for exploration
        return 0.1  # Low temperature for exploitation

class ReplayBuffer:
    '''
    Stores previously played games.
    '''
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def push(self, state, policy, value):
        # Convert policy to fixed size vector
        full_policy = torch.zeros(1968)  # Fixed size for all possible moves
        if len(policy) > 0:  # Only try to set values if we have any
            policy_tensor = torch.FloatTensor(policy)
            full_policy[:len(policy)] = policy_tensor  # Fill only the valid moves
        self.buffer.append((state, full_policy, value))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*samples)
        return (torch.stack(states), 
                torch.stack(policies),
                torch.tensor(values).float().unsqueeze(1))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Compute query, key, value
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        attention = F.softmax(torch.bmm(proj_query, proj_key), dim=-1)
        proj_value = self.value(x).view(batch_size, -1, height * width)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

class ChessNet(nn.Module):
    def __init__(self, num_residual_blocks=8):
        super(ChessNet, self).__init__()
        
        # Initial convolution block
        self.conv_initial = nn.Sequential(
            nn.Conv2d(19, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Residual tower
        self.residual_tower = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_residual_blocks)
        ])
        
        # Attention layer
        self.attention = SelfAttention(256)
        
        # Policy head with deeper architecture
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 73, 1),  # Reduced channels before flattening
            nn.BatchNorm2d(73),
            nn.ReLU()
        )
        self.policy_fc = nn.Sequential(
            nn.Linear(73 * 64, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1968)  # Original output size
        )
        
        # Value head with deeper architecture
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Auxiliary heads for piece prediction
        self.piece_location_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 12, 1)  # 12 piece types
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.conv_initial(x)
        
        # Residual blocks
        for block in self.residual_tower:
            x = block(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Policy head
        policy = self.policy_head(x)
        policy = policy.view(-1, 73 * 64)
        policy = self.policy_fc(policy)
        
        # Value head
        value = self.value_head(x)
        value = value.view(-1, 64 * 64)
        value = self.value_fc(value)
        #value = torch.tanh(value)
        
        # Auxiliary piece location predictions
        #piece_locations = self.piece_location_head(x)
        
        
        return policy, value#, piece_locations
        

class Node:
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board # The board state at this node
        self.parent = parent
        self.move = move
        self.prior = float(prior) # The probability of picking this state from the parent [0, 1]
        
        self.children = {} # A lookup of legal child postions
        self.visit_count = 0 # Number of times this state was visited during MTCS. "Good" states are visted more often.
        self.value_sum = 0 # Total value of this node accumulated from all visits
        self.is_expanded = False

    def value(self):
        # Average value for a node
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.0):
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            visit_count = float(self.visit_count)
            child_visit_count = float(child.visit_count)
            child_value = float(child.value())
            child_prior = float(child.prior)
            
            score = child_value + c_puct * child_prior * np.sqrt(visit_count) / (1 + child_visit_count)
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

class AlphaZero:
    def __init__(self, model):
        self.model = model # Chess net
        self.optimizer = optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.001)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[1000, 5000, 10000],
            gamma=0.1
        )
        self.replay_buffer = ReplayBuffer() # Stores previous games
        self.curriculum = CurriculumManager() # Different training logic for early stages
        self.temperature = 1
        self.num_simulations = 600 # paper is 800
        self.c_puct = 1
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def self_play_game(self) -> List[Tuple]:
        """Play a full game, returning training data."""
        board = chess.Board()
        game_history = []
        moves_made = 0
        value_history = []  # Track recent value predictions for resignation
        
        while not board.is_game_over() and moves_made < self.curriculum.max_moves:
            state_tensor = board_to_tensor(board)
            current_player = board.turn
            
            # Get move probabilities from MCTS using temperature schedule
            temperature = self.curriculum.get_temperature(moves_made)
            actions, full_policy, original_probs = self.get_action_probs(board, temperature)
            if actions is None:
                break
            
            # Get neural network's evaluation
            with torch.no_grad():
                _, value_pred = self.model(state_tensor.unsqueeze(0).to(self.device))
                value_pred = value_pred.item()
            
            value_history.append(value_pred)
            
            # Check for resignation
            if self.curriculum.should_resign(value_history):
                # Resign gives a loss
                final_value = -1
                break
                
            game_history.append({
                'state': state_tensor,
                'policy': full_policy,
                'player': current_player,
                'board': board.copy()  # Store board state for debugging
            })
            
            # Select and make move
            action_idx = np.random.choice(len(actions), p=original_probs)
            move = actions[action_idx]
            board.push(move)
            moves_made += 1
        
        # Get final position value
        if board.is_game_over():
            final_value = self.curriculum.get_position_value(board)
        else:
            final_value = -.01  # Non-terminal game end (max moves reached)
        
        # Convert game history to training data
        game_data = []
        current_value = final_value
        
        for entry in reversed(game_history):
            # Store state and policy with value from that player's perspective
            game_data.append([
                entry['state'],
                entry['policy'],
                current_value
            ])
            current_value = -current_value
        
        game_data.reverse()
        
        # Update games played
        self.curriculum.update_curriculum()
                
        return game_data, moves_made


    def mcts_search(self, board):
        root = Node(board)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Select
            while node.is_expanded and not node.board.is_game_over():
                node = node.select_child(self.c_puct)
                if node is None:
                    break
                search_path.append(node)
            
            # Expand and evaluate
            if node is not None and not node.board.is_game_over():
                state_tensor = board_to_tensor(node.board).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy, value = self.model(state_tensor) # Probabilities of picking X move and whether win/loss
                policy = F.softmax(policy.squeeze(), dim=0).cpu().numpy()
                value = float(value.item())
                
                # Create children for legal moves
                for move in node.board.legal_moves:
                    new_board = node.board.copy()
                    new_board.push(move)
                    
                    # Use the neural network's policy output as prior
                    move_idx = move_to_index(move, node.board)
                    prior = float(policy[move_idx])
                    
                    node.children[move] = Node(
                        new_board,
                        parent=node,
                        move=move,
                        prior=prior
                    )
                node.is_expanded = True
            else:
                # Get terminal value from the perspective of the player who just moved
                value = self.curriculum.get_position_value(node.board)
                #if not node.board.turn:  # If it's black's turn, value is already from their perspective
                #    value = -value
            
            # Backup
            # Values are propagated up the tree, flipping at each level
            # because parent nodes alternate between players
            current_value = value
            for node in reversed(search_path):
                node.value_sum += current_value
                node.visit_count += 1
                current_value = -current_value  # Flip for parent's perspective
        
        return root

    def get_action_probs(self, board, temperature=1):
        """Get move probabilities from MCTS visit counts."""
        root = self.mcts_search(board)
        visits = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        
        if len(actions) == 0:
            return None, None, None
            
        if temperature == 0:
            probs = np.zeros_like(visits, dtype=float)
            probs[visits.argmax()] = 1
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
            
        # Create fixed-size policy vector
        full_policy = np.zeros(1968)
        # Map each move to its proper index
        for action, prob in zip(actions, probs):
            idx = move_to_index(action, board)
            full_policy[idx] = prob
            
        return actions, full_policy, probs


    def train_step(self):
        """Perform a single training step on a batch of data."""
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0
            
        self.model.train()
        states, target_policies, target_values = self.replay_buffer.sample(self.batch_size)
        states, target_policies = states.to(self.device), target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        # Forward pass
        policy_pred, value_pred = self.model(states)
        
        # Calculate losses
        policy_loss = -torch.sum(target_policies * F.log_softmax(policy_pred, dim=1)) / target_policies.size(0)
        value_loss = F.mse_loss(value_pred, target_values)
        
        total_loss = policy_loss + value_loss * 4
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()

        # Add gradient clipping to prevent value predictions from exploding
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()

    def train(self, num_iterations=1000):
        """Main training loop with enhanced monitoring."""
        for iteration in range(num_iterations):
            # Self-play phase
            self.model.eval()
            game_data, moves_made = self.self_play_game()
            
            # Store game data
            for state, policy, value in game_data:
                self.replay_buffer.push(state, policy, value)
            
            # Training phase
            policy_loss, value_loss = self.train_step()
            
            if iteration % 5 == 0:
                # Sample a batch for monitoring
                states, target_policies, values = self.replay_buffer.sample(min(10, len(self.replay_buffer)))
                
                with torch.no_grad():
                    pred_policies, pred_values = self.model(states.to(self.device))
                    # Convert to probabilities
                    pred_policies = F.softmax(pred_policies, dim=1)


                print(f"Iteration {iteration}")
                print(f"Moves Made: {moves_made}")
                print(f"Policy Loss: {policy_loss:.4f}")
                print(f"Value Loss: {value_loss:.4f}")
                print(f"Buffer size: {len(self.replay_buffer)}")
                print(f"Sample predictions:")

                # Print sequential data
                for i in range(min(5, len(game_data))):
                    print(f"Position {i}: Player {'White' if i % 2 == 0 else 'Black'}, "
                          f"Value: {game_data[i][2]:.4f}")
                
                print("Values - Predicted vs Actual:")
                for i in range(5):  # Show first 5 samples (random order)
                    print(f"  First Position {i+1}: {pred_values[i].item():.4f} vs {values[i].item():.4f}")
                
                print("\nTop 3 Policy Predictions vs Targets:")
                for i in range(3):  # Show first 5 samples
                    # Get indices of top 3 moves for both predicted and target
                    top_pred = torch.topk(pred_policies[i], 3)
                    top_target = torch.topk(target_policies[i], 3)
                    
                    print(f"\nPosition {i+1}:")
                    print("  Predicted moves (index, probability):")
                    for idx, prob in zip(top_pred.indices, top_pred.values):
                        try:
                            # Try to convert index to chess move for clarity
                            move = index_to_move(int(idx), chess.Board(states[i]))
                            print(f"    {move}: {prob:.4f}")
                        except:
                            print(f"    Index {idx}: {prob:.4f}")
                    
                    print("  Target moves (index, probability):")
                    for idx, prob in zip(top_target.indices, top_target.values):
                        try:
                            move = index_to_move(int(idx), chess.Board(states[i]))
                            print(f"    {move}: {prob:.4f}")
                        except:
                            print(f"    Index {idx}: {prob:.4f}")
                            
                print(f"\nPolicy entropy: {(-F.softmax(pred_policies, dim=1) * F.log_softmax(pred_policies, dim=1)).sum(1).mean():.4f}")
                print("-------------------")


def board_to_tensor(board):
    # Convert chess board to neural network input
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    
    # Piece positions (12 planes)
    piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            planes[piece_idx[piece.symbol()]][rank][file] = 1
    
    # Additional features (castling rights, en passant, etc.)
    # This is simplified - you'd want more features in practice
    planes[12] = float(board.turn)  # Current player
    planes[13] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square:
        rank, file = divmod(board.ep_square, 8)
        planes[17][rank][file] = 1
    planes[18] = float(board.halfmove_clock / 100.0)  # Normalized halfmove clock
    
    return torch.FloatTensor(planes)
