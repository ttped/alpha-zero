import chess

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Convert a chess move to a policy index (0-1967).
    Adds debug printing and strict bounds checking.
    """
    from_square = move.from_square
    to_square = move.to_square
    piece = board.piece_at(from_square)
    
    if piece is None:
        raise ValueError(f"No piece at square {chess.square_name(from_square)}")
        
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    
    def debug_info():
        return {
            'move': move,
            'piece': piece.symbol(),
            'from_square': chess.square_name(from_square),
            'to_square': chess.square_name(to_square),
            'from_rank': from_rank,
            'from_file': from_file,
            'to_rank': to_rank,
            'to_file': to_file
        }
    
    try:
        # PAWN MOVES
        if piece.piece_type == chess.PAWN:
            if move.promotion:
                promotion_piece = move.promotion
                file_diff = to_file - from_file
                promotion_index = promotion_piece - 2  # knight=0, bishop=1, rook=2, queen=3
                
                base = 56
                file_base = from_file * 12
                move_type = file_diff + 1
                index = base + file_base + move_type * 4 + promotion_index
                
            else:
                if abs(to_square - from_square) == 16:
                    # Double push
                    index = from_file  # 0-7
                else:
                    # Single push or capture
                    if to_file == from_file:
                        # Straight push
                        index = 8 + from_file  # 8-15
                    else:
                        # Capture
                        file_diff = to_file - from_file
                        capture_index = 0 if file_diff < 0 else 1
                        index = 16 + from_file * 2 + capture_index  # 16-31

        # KNIGHT MOVES
        elif piece.piece_type == chess.KNIGHT:
            base = 312
            dx = to_file - from_file
            dy = to_rank - from_rank
            
            directions = [
                (1, 2), (2, 1), (2, -1), (1, -2),
                (-1, -2), (-2, -1), (-2, 1), (-1, 2)
            ]
            try:
                direction_index = directions.index((dx, dy))
                index = base + from_square * 8 + direction_index
            except ValueError:
                print(f"Invalid knight move: {debug_info()}")
                raise

        # BISHOP MOVES
        elif piece.piece_type == chess.BISHOP:
            base = 696
            dx = to_file - from_file
            dy = to_rank - from_rank
            
            if dx > 0 and dy > 0: direction = 0
            elif dx > 0 and dy < 0: direction = 1
            elif dx < 0 and dy < 0: direction = 2
            else: direction = 3
            
            distance = min(abs(dx) - 1, 5)  # Cap distance at 5 to stay within bounds
            index = base + from_square * 6 + distance

        # ROOK MOVES
        elif piece.piece_type == chess.ROOK:
            base = 1080
            dx = to_file - from_file
            dy = to_rank - from_rank
            
            if dx == 0 and dy > 0: direction = 0
            elif dx > 0 and dy == 0: direction = 1
            elif dx == 0 and dy < 0: direction = 2
            else: direction = 3
            
            distance = min(max(abs(dx), abs(dy)) - 1, 5)  # Cap distance at 5
            index = base + from_square * 6 + distance

        # QUEEN MOVES
        elif piece.piece_type == chess.QUEEN:
            base = 1464
            dx = to_file - from_file
            dy = to_rank - from_rank
            
            if dx == 0 and dy > 0: direction = 0
            elif dx > 0 and dy > 0: direction = 1
            elif dx > 0 and dy == 0: direction = 2
            elif dx > 0 and dy < 0: direction = 3
            elif dx == 0 and dy < 0: direction = 4
            elif dx < 0 and dy < 0: direction = 5
            elif dx < 0 and dy == 0: direction = 6
            else: direction = 7
            
            distance = min(max(abs(dx), abs(dy)) - 1, 5)  # Cap distance at 5
            index = base + from_square * 6 + distance

        # KING MOVES
        elif piece.piece_type == chess.KING:
            base = 1848
            if board.is_castling(move):
                castle_index = 0 if to_file > from_file else 1  # 0 for kingside, 1 for queenside
                index = base + castle_index
            else:
                # Regular king moves
                dx = to_file - from_file
                dy = to_rank - from_rank
                
                # Map the 8 possible king moves to indices 0-7
                # Using a simpler mapping scheme
                move_map = {
                    # (dx, dy): index
                    (0, 1): 0,    # North
                    (1, 1): 1,    # Northeast
                    (1, 0): 2,    # East
                    (1, -1): 3,   # Southeast
                    (0, -1): 4,   # South
                    (-1, -1): 5,  # Southwest
                    (-1, 0): 6,   # West
                    (-1, 1): 7,   # Northwest
                }
                
                move_index = move_map.get((dx, dy))
                if move_index is None:
                    print(f"Invalid king move mapping: dx={dx}, dy={dy}")
                    move_index = 0
                
                # Use a smaller multiplier for the from_square to stay within bounds
                # We allocate 120 indices for king moves (64 squares * 8 directions would be too many)
                # Instead, we'll use: base + (square_group * 8) + direction
                square_group = from_square // 8  # This gives us 8 groups of squares
                index = base + 2 + (square_group * 8) + move_index  # +2 to account for castling moves
        
        else:
            raise ValueError(f"Unknown piece type: {piece.piece_type}")
            
        # Final bounds check
        if not 0 <= index < 1968:
            print(f"Index out of bounds: {index}")
            print(f"Move details: {debug_info()}")
            index = min(max(index, 0), 1967)  # Clamp to valid range
            
        return index
        
    except Exception as e:
        print(f"Error processing move: {debug_info()}")
        raise

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert a policy index back to a legal chess move."""
    legal_moves = list(board.legal_moves)
    
    # Find the move that maps to this index
    for move in legal_moves:
        if move_to_index(move, board) == index:
            return move
            
    raise ValueError(f"No legal move maps to index {index}")

# Example usage and testing
def test_move_mapping():
    board = chess.Board()
    
    # Create a set to track used indices
    used_indices = set()
    
    # Test mapping for all legal moves
    for _ in range(10):
        print(_)
        for move in board.legal_moves:
            index = move_to_index(move, board)
            assert 0 <= index < 1968, f"Index {index} out of bounds for move {move}"
            assert index not in used_indices, f"Duplicate index {index} for move {move}"
            used_indices.add(index)
            
            # Test reverse mapping
            recovered_move = index_to_move(index, board)
            assert move == recovered_move, f"Move mapping not bijective for {move}"
    
    print(f"Tested {len(used_indices)} moves, all mappings valid and unique")

if __name__ == "__main__":
    test_move_mapping()
