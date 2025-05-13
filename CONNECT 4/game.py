import tkinter as tk
from tkinter import ttk 
from tkinter import messagebox
import random
import math
from PIL import Image, ImageTk, ImageDraw
import os

class Board:
    EMPTY = 0
    HUMAN = 1
    AI = 2

    def __init__(self, rows=6, cols=7):
        
        self.rows = rows
        self.cols = cols
        self.board = [[self.EMPTY] * cols for _ in range(rows)]
        self.current_player = self.HUMAN
        self.winner = None
        self.game_over = False
        self.last_move = None
        self.winning_positions = []

    def reset(self):
        self.board = [[self.EMPTY] * self.cols for _ in range(self.rows)]
        self.current_player = self.HUMAN
        self.winner = None
        self.game_over = False
        self.last_move = None
        self.winning_positions = []

    def is_valid_move(self, col):
        return 0 <= col < self.cols and self.board[0][col] == self.EMPTY

    def get_valid_moves(self):
        return [col for col in range(self.cols) if self.is_valid_move(col)]

    def drop_piece(self, col):
        if not self.is_valid_move(col) or self.game_over:
            return False
        row = -1
        for r in range(self.rows-1, -1, -1):
            if self.board[r][col] == self.EMPTY:
                self.board[r][col] = self.current_player
                row = r
                self.last_move = (row, col)
                break
        if self.check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        elif self.is_full():
            self.game_over = True
        else:
            self.current_player = self.AI if self.current_player == self.HUMAN else self.HUMAN
        return True

    def is_full(self):
        return all(cell != self.EMPTY for cell in self.board[0])

    def check_win(self, row, col):
        if row < 0 or col < 0:
            return False
        player = self.board[row][col]
        if player == self.EMPTY:
            return False
        directions = [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ]
        for dir_pair in directions:
            count = 1
            positions = [(row, col)]
            for dr, dc in dir_pair:
                r, c = row, col
                for _ in range(3):
                    r += dr
                    c += dc
                    if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == player:
                        count += 1
                        positions.append((r, c))
                    else:
                        break
            if count >= 4:
                self.winning_positions = positions
                return True
        return False

    def get_winner(self):
        return self.winner if self.game_over else None

    def get_current_player(self):
        return self.current_player

    def get_winning_positions(self):
        return self.winning_positions

    def clone(self):
        new_board = Board(self.rows, self.cols)
        new_board.board = [row[:] for row in self.board]
        new_board.current_player = self.current_player
        new_board.winner = self.winner
        new_board.game_over = self.game_over
        new_board.last_move = self.last_move
        return new_board

class AI:
    def __init__(self, depth=4):
        self.depth = depth
        self.board = None

    def set_board(self, board):
        self.board = board

    def get_move(self):
        valid_moves = self.board.get_valid_moves()
        if not valid_moves:
            return None

        # Difficulty: easy = more random, medium = some random, hard = always best
        difficulty = self.get_difficulty_level()
        if difficulty == "easy" and random.random() < 0.7:
            return random.choice(valid_moves)
        elif difficulty == "medium" and random.random() < 0.3:
            return random.choice(valid_moves)

        # Check for immediate win
        for col in valid_moves:
            temp_board = self.board.clone()
            temp_board.drop_piece(col)
            if temp_board.winner == Board.AI:
                return col

        # Block human's immediate win
        for col in valid_moves:
            temp_board = self.board.clone()
            temp_board.current_player = Board.HUMAN
            temp_board.drop_piece(col)
            if temp_board.winner == Board.HUMAN:
                return col

        # Use A*-inspired minimax for deeper analysis
        best_score = -math.inf
        best_move = random.choice(valid_moves)
        for col in valid_moves:
            temp_board = self.board.clone()
            temp_board.drop_piece(col)
            score = self.alpha_beta_minmax(temp_board, self.depth-1, False, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_move = col
        return best_move

    def alpha_beta_minmax(self, board, depth, maximizing, alpha, beta): # Min max with alpha beta pruning 
        # Terminal state 
        if board.game_over:
            if board.winner == Board.AI:
                return 10000
            elif board.winner == Board.HUMAN:
                return -10000
            else:
                return 0
        if depth == 0:
            return self.astar_evaluate_board(board)

        valid_moves = board.get_valid_moves()
        if maximizing:
            max_eval = -math.inf
            for col in valid_moves:
                temp_board = board.clone()
                temp_board.current_player = Board.AI
                temp_board.drop_piece(col)
                eval = self.alpha_beta_minmax(temp_board, depth-1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for col in valid_moves:
                temp_board = board.clone()
                temp_board.current_player = Board.HUMAN
                temp_board.drop_piece(col)
                eval = self.alpha_beta_minmax(temp_board, depth-1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def astar_evaluate_board(self, board):
        # A*-inspired heuristic: count possible lines, center control, block/threats
        score = 0
        center_col = board.cols // 2
        # Center column preference
        for r in range(board.rows):
            if board.board[r][center_col] == Board.AI:
                score += 3
            elif board.board[r][center_col] == Board.HUMAN:
                score -= 3

        # Check all 4-in-a-row windows
        for row in range(board.rows):
            for col in range(board.cols):
                if board.board[row][col] != Board.EMPTY:
                    score += self.evaluate_window(board, row, col, Board.AI)
                    score -= self.evaluate_window(board, row, col, Board.HUMAN)
        return score

    def evaluate_window(self, board, row, col, player):
        # Check all directions for possible 4-in-a-row from (row, col)
        score = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 0
            empty = 0
            for i in range(4):
                r = row + dr*i
                c = col + dc*i
                if 0 <= r < board.rows and 0 <= c < board.cols:
                    if board.board[r][c] == player:
                        count += 1
                    elif board.board[r][c] == Board.EMPTY:
                        empty += 1
            # Heuristic: 4 in a row = big score, 3+empty = medium, 2+empty = small
            if count == 4:
                score += 100
            elif count == 3 and empty == 1:
                score += 10
            elif count == 2 and empty == 2:
                score += 2
        return score

    def get_difficulty_level(self):
        if self.depth <= 2:
            return "easy"
        elif self.depth <= 4:
            return "medium"
        else:
            return "hard"

    def __init__(self, depth=4):
        self.depth = depth
        self.board = None
        self.difficulty_weights = {
            "easy": {'win': 1, 'block': 0.5, 'center': 0.3},
            "medium": {'win': 3, 'block': 2, 'center': 1},
            "hard": {'win': 5, 'block': 4, 'center': 2}
        }

    def set_board(self, board):
        self.board = board

    def a_star_heuristic(self, board, difficulty):
        """A*-inspired heuristic evaluation with difficulty-based weights"""
        weights = self.difficulty_weights[difficulty]
        score = 0
        
        # Check all possible lines
        for row in range(board.rows):
            for col in range(board.cols):
                if board.board[row][col] == Board.AI:
                    score += self.evaluate_position(board, row, col, Board.AI, weights)
                elif board.board[row][col] == Board.HUMAN:
                    score -= self.evaluate_position(board, row, col, Board.HUMAN, weights)
        
        # Center column preference
        center_col = board.cols // 2
        for r in range(board.rows):
            if board.board[r][center_col] == Board.AI:
                score += weights['center']
            elif board.board[r][center_col] == Board.HUMAN:
                score -= weights['center']
        
        return score

    def evaluate_position(self, board, row, col, player, weights):
        """A* style path evaluation for potential lines"""
        value = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            path_length = 0
            empty_spaces = 0
            for i in range(4):
                r = row + dr * i
                c = col + dc * i
                if 0 <= r < board.rows and 0 <= c < board.cols:
                    if board.board[r][c] == player:
                        path_length += 1
                    elif board.board[r][c] == Board.EMPTY:
                        empty_spaces += 1
                    else:
                        break
            if path_length + empty_spaces >= 4:
                value += weights['win'] * path_length**2
                
        return value

    def get_move(self):
        valid_moves = self.board.get_valid_moves()
        difficulty = self.get_difficulty_level()
        
        if not valid_moves:
            return None

        # Difficulty-based strategy
        if difficulty == "easy":
            if random.random() < 0.5:
                return random.choice(valid_moves)
            
        best_score = -math.inf
        best_move = random.choice(valid_moves)
        
        for col in valid_moves:
            temp_board = self.board.clone()
            temp_board.drop_piece(col)
            score = self.a_star_search(temp_board, self.depth-1, False)
            if score > best_score:
                best_score = score
                best_move = col
                
        return best_move

    def a_star_search(self, board, depth, maximizing):
        """A*-inspired search with alpha-beta pruning"""
        if board.game_over or depth == 0:
            return self.a_star_heuristic(board, self.get_difficulty_level())
        
        valid_moves = board.get_valid_moves()
        if maximizing:
            max_eval = -math.inf
            for col in valid_moves:
                temp_board = board.clone()
                temp_board.drop_piece(col)
                eval = self.a_star_search(temp_board, depth-1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = math.inf
            for col in valid_moves:
                temp_board = board.clone()
                temp_board.drop_piece(col)
                eval = self.a_star_search(temp_board, depth-1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def get_difficulty_level(self):
        return {
            2: "easy",
            4: "medium",
            6: "hard"
        }.get(self.depth, "medium")

class Game:
    def __init__(self, ai_depth=4):
        self.board = Board()
        self.ai = AI(depth=ai_depth)
        self.ai.set_board(self.board)
        self.scores = {"human": 0, "ai": 0, "draws": 0}

    def make_move(self, col):
        if self.board.drop_piece(col):
            return True
        return False

    def make_ai_move(self):
        col = self.ai.get_move()
        if col is not None:
            self.board.drop_piece(col)
            return True
        return False

    def is_game_over(self):
        return self.board.game_over

    def get_winner(self):
        return self.board.get_winner()

    def get_current_player(self):
        return self.board.get_current_player()

    def get_winning_positions(self):
        return self.board.get_winning_positions()

    def reset(self):
        self.board.reset()

    def update_score(self):
        winner = self.get_winner()
        if winner == Board.HUMAN:
            self.scores["human"] += 1
        elif winner == Board.AI:
            self.scores["ai"] += 1
        else:
            self.scores["draws"] += 1

    def get_scores(self):
        return self.scores

    def set_ai_depth(self, depth):
        self.ai = AI(depth=depth)
        self.ai.set_board(self.board)

class GameGUI(tk.Tk):
    def __init__(self, ai_depth=4):
        super().__init__()
        self.title("Connect 4 - Space Edition")
        self.geometry("800x850")
        self.resizable(False, False)
        self.configure(bg="#0B0B2B")  # Deep space blue

        # ttk style configuration for a space theme
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TButton',
                             font=('Helvetica', 14, 'bold'),
                             padding=12,
                             background='#4A90E2',   # Bright blue
                             foreground='#FFFFFF',   # White text
                             borderwidth=2,
                             relief='raised',
                             bordercolor='#F5F5F5')  # Off-white border
        self.style.map('TButton',
                       background=[('active', '#357ABD'), ('!disabled', '#4A90E2')],
                       foreground=[('active', '#fff')],
                       bordercolor=[('active', '#F5F5F5'), ('!disabled', '#F5F5F5')])
        self.style.configure('TLabel',
                             font=('Helvetica', 12),
                             background='#0B0B2B',   # Deep space blue
                             foreground='#FFFFFF')   # White text
        self.style.configure('Title.TLabel',
                             font=('Helvetica', 20, 'bold'),
                             foreground='#4A90E2',   # Bright blue
                             background='#0B0B2B')
        self.style.configure('TFrame', background='#0B0B2B')

        self.game = Game(ai_depth=ai_depth)
        self.cell_size = 90
        self.colors = {
            0: "#0B0B2B",      # Empty cell: deep space blue
            1: "#4A90E2",      # Human: bright blue
            2: "#FF6B6B",      # AI: cosmic red
            "highlight": "#FFD700"  # Gold for winning pieces
        }
        self.create_background()
        self.create_widgets()
        self.draw_board()

    def create_background(self):
        width = self.game.board.cols * self.cell_size
        height = self.game.board.rows * self.cell_size
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(current_dir, "space_background.png")
            image = Image.open(image_path)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Create a new image with border
            border_size = 10
            bordered_image = Image.new('RGB', (width + 2*border_size, height + 2*border_size), '#2A2A4A')
            bordered_image.paste(image, (border_size, border_size))
            
            # Draw border
            draw = ImageDraw.Draw(bordered_image)
            # Outer border
            draw.rectangle([(0, 0), (width + 2*border_size - 1, height + 2*border_size - 1)], 
                         outline='#4A90E2', width=2)
            # Inner border
            draw.rectangle([(border_size-2, border_size-2), 
                          (width + border_size + 1, height + border_size + 1)], 
                         outline='#FF6B6B', width=2)
            
            self.bg_image = ImageTk.PhotoImage(bordered_image)
        except Exception:
            # Create a bordered image even if background image fails to load
            border_size = 10
            image = Image.new('RGB', (width + 2*border_size, height + 2*border_size), '#0B0B2B')
            draw = ImageDraw.Draw(image)
            # Outer border
            draw.rectangle([(0, 0), (width + 2*border_size - 1, height + 2*border_size - 1)], 
                         outline='#4A90E2', width=2)
            # Inner border
            draw.rectangle([(border_size-2, border_size-2), 
                          (width + border_size + 1, height + border_size + 1)], 
                         outline='#FF6B6B', width=2)
            self.bg_image = ImageTk.PhotoImage(image)

    def create_widgets(self):
        main_frame = ttk.Frame(self, style='TFrame')
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH)

        ttk.Label(main_frame, text="CONNECT 4", style='Title.TLabel').pack(pady=10)

        # Update canvas size to account for border
        border_size = 10
        self.canvas = tk.Canvas(main_frame,
                               width=self.game.board.cols * self.cell_size + 2*border_size,
                               height=self.game.board.rows * self.cell_size + 2*border_size,
                               bg="#0B0B2B", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg_image)
        self.canvas.bind("<Button-1>", self.on_click)

        score_frame = ttk.Frame(main_frame, style='TFrame')
        score_frame.pack(pady=10)
        
        self.human_score = ttk.Label(score_frame, text="HUMAN: 0", style='TLabel')
        self.human_score.pack(side=tk.LEFT, padx=30)
        
        self.ai_score = ttk.Label(score_frame, text="AI: 0", style='TLabel')
        self.ai_score.pack(side=tk.LEFT, padx=30)
        
        self.draws_score = ttk.Label(score_frame, text="DRAWS: 0", style='TLabel')
        self.draws_score.pack(side=tk.LEFT, padx=30)

        control_frame = ttk.Frame(main_frame, style='TFrame')
        control_frame.pack(pady=10)

        ttk.Label(control_frame, text="Difficulty:", style='TLabel').pack(side=tk.LEFT)
        self.difficulty = tk.StringVar(value="MEDIUM")
        difficulty_menu = ttk.Combobox(control_frame, textvariable=self.difficulty,
                                      values=["EASY", "MEDIUM", "HARD"], state="readonly",
                                      font=('Helvetica', 12))
        difficulty_menu.pack(side=tk.LEFT, padx=10)
        difficulty_menu.bind("<<ComboboxSelected>>", self.change_difficulty)

        reset_btn = ttk.Button(control_frame, text="New Game", command=self.reset_game, style='TButton')
        reset_btn.pack(side=tk.LEFT, padx=15)

        self.status = ttk.Label(main_frame, text="Human's turn", style='TLabel')
        self.status.pack(pady=10)

    def draw_board(self):
        self.canvas.delete("all")
        if hasattr(self, 'bg_image'):
            self.canvas.create_image(0, 0, anchor="nw", image=self.bg_image)
        winning_positions = set(self.game.get_winning_positions()) if self.game.is_game_over() else set()
        
        # Calculate cell size to fit within the background
        border_size = 10
        available_width = self.game.board.cols * self.cell_size
        available_height = self.game.board.rows * self.cell_size
        piece_radius = min(available_width // (2 * self.game.board.cols), 
                          available_height // (2 * self.game.board.rows)) - 4
        
        for row in range(self.game.board.rows):
            for col in range(self.game.board.cols):
                x = col * self.cell_size + self.cell_size//2 + border_size
                y = row * self.cell_size + self.cell_size//2 + border_size
                
                # Outer ring for depth
                self.canvas.create_oval(x-piece_radius-2, y-piece_radius-2,
                                      x+piece_radius+2, y+piece_radius+2,
                                      fill="#1A1A3A", outline="#F5F5F5")
                
                value = self.game.board.board[row][col]
                if value != Board.EMPTY:
                    color = self.colors[value]
                    if (row, col) in winning_positions:
                        self.canvas.create_oval(x-piece_radius-4, y-piece_radius-4,
                                              x+piece_radius+4, y+piece_radius+4,
                                              fill=self.colors["highlight"],
                                              outline="")
                    self.canvas.create_oval(x-piece_radius, y-piece_radius,
                                          x+piece_radius, y+piece_radius,
                                          fill=color, outline="#F5F5F5", width=2)
                else:
                    # Empty cell
                    self.canvas.create_oval(x-piece_radius, y-piece_radius,
                                           x+piece_radius, y+piece_radius,
                                           fill=self.colors[0], outline="#F5F5F5")

    def on_click(self, event):
        if self.game.is_game_over() or self.game.get_current_player() != Board.HUMAN:
            return
        col = event.x // self.cell_size
        if self.game.make_move(col):
            self.draw_board()
            if self.game.is_game_over():
                self.game_over()
            else:
                self.status.config(text="AI is thinking...")
                self.after(600, self.ai_move)

    def ai_move(self):
        self.game.make_ai_move()
        self.draw_board()
        if self.game.is_game_over():
            self.game_over()
        else:
            current_player = self.game.get_current_player()
            player_name = "Human" if current_player == Board.HUMAN else "AI"
            self.status.config(text=f"{player_name}'s turn")

    def game_over(self):
        self.game.update_score()
        self.update_scores()
        winner = self.game.get_winner()
        if winner:
            message = f"{'Human' if winner == Board.HUMAN else 'AI'} wins!"
        else:
            message = "It's a draw!"
        self.status.config(text=message)
        messagebox.showinfo("Game Over", message)
        self.after(1200, self.new_round)

    def new_round(self):
        self.game.reset()
        self.draw_board()
        self.status.config(text="Human's turn")
        if self.game.get_current_player() == Board.AI:
            self.status.config(text="AI is thinking...")
            self.after(600, self.ai_move)

    def update_scores(self):
        scores = self.game.get_scores()
        self.human_score.config(text=f"Human: {scores['human']}")
        self.ai_score.config(text=f"AI: {scores['ai']}")
        self.draws_score.config(text=f"Draws: {scores['draws']}")

    def reset_game(self):
        self.game.reset()
        self.game.scores = {"human": 0, "ai": 0, "draws": 0}
        self.update_scores()
        self.draw_board()
        self.status.config(text="Human's turn")

    def change_difficulty(self, event):
        depth_map = {"easy": 2, "medium": 4, "hard": 6}
        self.game.set_ai_depth(depth_map[self.difficulty.get()])
        self.game.reset()
        self.draw_board()
        self.status.config(text="Human's turn")

if __name__ == "__main__": 
    app = GameGUI(ai_depth=4)
    app.mainloop()