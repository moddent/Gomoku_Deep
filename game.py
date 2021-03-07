from __future__ import print_function
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from mcts_alphaZero import MCTSPlayer
import numpy as np
from PIL import Image, ImageTk
import copy
from policy_value_net_pytorch import PolicyValueNet

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player




class Game(tk.Tk):
    """game server"""

    def __init__(self, board, **kwargs):
        super().__init__()
        self.player = None
        self.board = board
        self.copy_board = copy.copy(board)
        self.board.init_board(start_player=0)

        # window size
        self.WINDOW_SIZE_width = 1000
        self.WINDOW_SIZE_height = 600

        # show who black, who white
        self.player_info_black = tk.Label(self,
                     text='Black: None',
                     font=('Arial', 12, "bold"),
                     anchor='w',
                     width=15, height=2
                     )
        self.player_info_black.place(x=610, y=100)

        # show who black, who white
        self.player_info_white = tk.Label(self,
                     text='White: None',
                     font=('Arial', 12, "bold"),
                     anchor='w',
                     width=15, height=2
                     )
        self.player_info_white.place(x=610, y=130)

        # show information
        self.text_info = tk.Label(self,
                     text='Information',
                     font=('Arial', 12, "bold"),
                     width=15, height=2
                     )
        self.text_info.place(x=730, y=365)
        self.text_box = tk.Text(self, height=14)
        self.text_box.config(font=('Courier', 12))
        self.text_box.place(x=602, y=400)

        # restart button
        img = tk.PhotoImage(file="restart_button_img.png")
        self.restart_button = tk.Button(self, image=img)
        self.restart_button.image = img
        self.restart_button.place(x=900, y=100)
        self.restart_button.bind("<ButtonRelease-1>", self.restart_button_handler)

        # only the board size
        self.BOARD_SIZE = 600
        self.title("Gomoku")
        self.geometry('600x600')
        self.resizable(False, False)
        self.flag = True
        self.minsize(self.WINDOW_SIZE_width, self.WINDOW_SIZE_height)

        self.canvas = tk.Canvas(self,
                    width=self.BOARD_SIZE,
                    height=self.BOARD_SIZE,
                    bg='white')

        self.background = ImageTk.PhotoImage(Image.open("background.jpg"))
        self.choice_who_first_view_background = ImageTk.PhotoImage(Image.open("background2.jpg"))
        self.start_view()

        # check for the mouse press to release -> Action
        self.check_click_coordinate_press = (np.inf, np.inf)
        self.check_click_coordinate_Release = (np.inf, np.inf)
        self.fist_time_release = True

        # get the level and who first
        self.LEVEL = None
        self.WHO_FIRST = None

        # step counter
        self.step_counter = 1

        # # Record_board
        # self.record_board = Record_board()

    def start_view(self):
        self.canvas.create_image(300, 450, image=self.background)  # image position
        self.canvas.pack(side="left")

        buttonBG = self.canvas.create_rectangle(230, 400, 370, 460, fill="#FFFFFF", outline="black")
        buttonTXT = self.canvas.create_text(300, 430, font=("courier", 25, "bold"), text="Start")
        self.canvas.create_rectangle(233, 403, 367, 457, outline="black")

        self.canvas.tag_bind(buttonBG, "<ButtonRelease-1>", self.reset_window)
        self.canvas.tag_bind(buttonTXT, "<ButtonRelease-1>", self.reset_window)

    # reset all element in start window
    def reset_window(self, *args):
        print("------------Start------------")
        self.handle_show_info("Start")
        self.canvas.delete('all')
        self.choice_who_first_view()

    def choice_who_first_view(self):
        self.canvas.create_image(200, 300, image=self.choice_who_first_view_background)

        self.canvas.create_text(300, 120,  # text position
                                text='Who first?',  # text
                                font=("courier", 35, "bold"),
                                fill='#FFFFFF')  # text color
        player_first_button = self.canvas.create_rectangle(230, 230, 370, 280, fill="#FFFFFF", outline="black")
        player_first_buttonTXT = self.canvas.create_text(300, 257, font=("courier", 18, "bold"), text="You")
        self.canvas.create_rectangle(233, 233, 367, 277, outline="black")

        computer_first_button = self.canvas.create_rectangle(230, 330, 370, 380, fill="#FFFFFF", outline="black")
        computer_first_buttonTXT = self.canvas.create_text(300, 357, font=("courier", 18, "bold"), text="Computer")
        self.canvas.create_rectangle(233, 333, 367, 377, outline="black")

        # Event handle
        self.canvas.tag_bind(player_first_button, "<ButtonRelease-1>", lambda x: self.handle_who_first(1))
        self.canvas.tag_bind(player_first_buttonTXT, "<ButtonRelease-1>", lambda x: self.handle_who_first(1))

        self.canvas.tag_bind(computer_first_button, "<ButtonRelease-1>", lambda x: self.handle_who_first(2))
        self.canvas.tag_bind(computer_first_buttonTXT, "<ButtonRelease-1>", lambda x: self.handle_who_first(2))

        self.canvas.pack()

    def handle_who_first(self, button_id):
        switcher = {
            1: "You",
            2: "Computer"
        }
        WHO_FIRST = switcher.get(button_id)
        print("----------Press " + WHO_FIRST + "----------")
        self.handle_show_info("First is < " + WHO_FIRST + " >")
        self.WHO_FIRST = WHO_FIRST
        self.canvas.delete('all')

        if WHO_FIRST == "You":
            self.player_info_black.config(text='Black: You')
            self.player_info_white.config(text='White: Computer')
        else:
            self.player_info_black.config(text='Black: Computer')
            self.player_info_white.config(text='White: You')
        self.choice_level()

    def choice_level(self):
        self.canvas.create_image(200, 300, image=self.choice_who_first_view_background)

        self.canvas.create_text(300, 120,  # text position
                                text='Degree of difficulty',  # text
                                font=("courier", 30, "bold"),
                                fill='#FFFFFF')  # text color
        hard_button = self.canvas.create_rectangle(230, 230, 370, 280, fill="#FFFFFF", outline="black")
        hard_buttonTXT = self.canvas.create_text(300, 257, font=("courier", 18, "bold"), text="Hard")
        self.canvas.create_rectangle(233, 233, 367, 277, outline="black")

        normal_button = self.canvas.create_rectangle(230, 330, 370, 380, fill="#FFFFFF", outline="black")
        normal_buttonTXT = self.canvas.create_text(300, 357, font=("courier", 18, "bold"), text="Normal")
        self.canvas.create_rectangle(233, 333, 367, 377, outline="black")

        easy_button = self.canvas.create_rectangle(230, 430, 370, 480, fill="#FFFFFF", outline="black")
        easy_buttonTXT = self.canvas.create_text(300, 457, font=("courier", 18, "bold"), text="Easy")
        self.canvas.create_rectangle(233, 433, 367, 477, outline="black")

        # return to previous view
        return_button = Button(self, text="BACK", font=("courier", 9, "bold"), command=self.reset_window, anchor=W)
        return_button.configure(width=5, activebackground="#33B5E5", relief=GROOVE, bg="#FFCCCC")
        self.canvas.create_window(25, 550, anchor=NW, window=return_button)

        # Event handle
        self.canvas.tag_bind(hard_button, "<ButtonRelease-1>", lambda x: self.handle_level(1))
        self.canvas.tag_bind(hard_buttonTXT, "<ButtonRelease-1>", lambda x: self.handle_level(1))

        self.canvas.tag_bind(normal_button, "<ButtonRelease-1>", lambda x: self.handle_level(2))
        self.canvas.tag_bind(normal_buttonTXT, "<ButtonRelease-1>", lambda x: self.handle_level(2))

        self.canvas.tag_bind(easy_button, "<ButtonRelease-1>", lambda x: self.handle_level(3))
        self.canvas.tag_bind(easy_buttonTXT, "<ButtonRelease-1>", lambda x: self.handle_level(3))

        self.canvas.pack()

    def handle_level(self, button_id):
        switcher = {
            1: "Hard",
            2: "Normal",
            3: "Easy"
        }
        LEVEL = switcher.get(button_id)
        print("-------------" + LEVEL + "------------")
        self.handle_show_info("Level:  < " + LEVEL + " >")
        self.LEVEL = LEVEL
        self.canvas.delete('all')
        self.starting_gomoku()

    def handle_show_info(self, info):
        info = info + "\n"
        self.text_box.insert('end', info)
        self.text_box.see(END)

    # (row, col) to (9, D)...
    def handle_coordinate_info(self, coor):
        switcher = {
            0: "9",
            1: "8",
            2: "7",
            3: "6",
            4: "5",
            5: "4",
            6: "3",
            7: "2",
            8: "1",
        }
        row = switcher.get(coor[0])

        switcher = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H",
            8: "J",
        }
        col = switcher.get(coor[1])
        return row, col

    def starting_gomoku(self):
        print("-------starting gomoku-------")
        self.handle_show_info("-------starting gomoku-------")
        self.canvas.create_rectangle(0, 0, self.BOARD_SIZE, self.BOARD_SIZE, fill="#FFBB66")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", lambda event: self.MousePress(event, 1))
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.MousePress(event, 2))
        self.draw_Line()

        # run setting here
        self.game_run_setting()

    def MousePress(self, event, check):
        if check == 1:
            # print("Press")
            self.check_click_coordinate_press = (event.x, event.y)
        else:
            # print("Release")
            if self.fist_time_release:
                self.fist_time_release = False
            else:
                self.check_click_coordinate_Release = (event.x, event.y)
                if abs(sum(self.check_click_coordinate_press) - sum(self.check_click_coordinate_Release)) < 10:
                    current_player = self.board.get_current_player()
                    location = self.Coordinate(event.x, event.y)
                    move = self.get_action(location)
                    if move != -1 and self.flag:
                        x = move // self.board.width
                        y = move % self.board.width
                        # print("player{} action (row,column):(" .format(current_player) + str(x) + "," + str(y) + ")")
                        info_x, info_y = self.handle_coordinate_info((x, y))
                        self.handle_show_info("player{} action :(" .format(current_player) + info_x + "," + info_y + ")")
                        self.drawXY(x, y, current_player)
                        self.flag = False
                        self.board.do_move(move)
                        if self.check_game_state() is not True:
                            self.player_action()
                            self.flag = True
                        else:
                            self.board = copy.copy(self.copy_board)
                            self.canvas.destroy()
                            self.canvas = tk.Canvas(self,
                                                    width=self.BOARD_SIZE,
                                                    height=self.BOARD_SIZE,
                                                    bg='white')
                            self.canvas.pack(side="left")
                            print("End")
                            self.handle_show_info("Cancel action")
                            self.restart()

                        # # previous board here
                        # self.record_board.push(copy.deepcopy(self.board))
                    else:
                        return 0
                else:
                    print("Cancel action")
                    self.handle_show_info("Cancel action")

    def game_run_setting(self):
        width, height = 9, 9
        model_file = 'best_policy.model'
        best_policy = PolicyValueNet(width, height, model_file=model_file)

        # LEVEL = (hard, normal, easy) n_playout
        switcher = {
            "Hard": 120,
            "Normal": 8,
            "Easy": 3
        }
        LEVEL = switcher.get(self.LEVEL)

        AlphaZero_mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=LEVEL)

        print("Level is :", self.LEVEL, "n_playout:", LEVEL)

        # ===Who first here===
        if self.WHO_FIRST == "You":
            self.set_player(AlphaZero_mcts_player)
        else:
            self.set_player(AlphaZero_mcts_player)
            self.player_action()

    def draw_Line(self):
        x0, y0, x1, y1 = 60, 60, 540, 540  # start window X and Y
        line = [0] * 15
        j = 0
        for i in range(0, 9):
            line[i] = x0 + i * 60
            self.canvas.create_line(line[i], y0, line[i], y1)
            self.canvas.create_line(x0, line[i], x1, line[i])
            self.canvas.create_text(30, 60 + i * 60, text=9 - i)
            self.canvas.create_text(570, 60 + i * 60, text=9 - i)
            if j == 8:
                j = j + 1
            self.canvas.create_text(60 + i * 60, 30, text=chr(65 + j))
            self.canvas.create_text(60 + i * 60, 570, text=chr(65 + j))
            j = j + 1

    def get_action(self, location):
        try:
            move = self.board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in self.board.availables:
            print("invalid move")
            self.handle_show_info("invalid move")
            move = -1
        return move

    def set_player(self, player):
        self.player = player

    def player_action(self):
        current_player = self.board.get_current_player()
        # print("\nWait for player{} action...".format(current_player))
        self.handle_show_info("\nWait for player{} action...".format(current_player))
        move = self.player.get_action(self.board)
        x = move // self.board.width
        y = move % self.board.width
        self.drawXY(x, y, current_player)
        self.board.do_move(move)
        if self.check_game_state() is not True:
            # print("player{} action (row,column):(" .format(current_player) + str(x) + "," + str(y) + ")")
            info_x, info_y = self.handle_coordinate_info((x, y))
            self.handle_show_info("player{} action :(".format(current_player) + info_x + "," + info_y + ")")
        else:
            self.board = copy.copy(self.copy_board)
            self.canvas.destroy()
            self.canvas = tk.Canvas(self,
                                    width=self.BOARD_SIZE,
                                    height=self.BOARD_SIZE,
                                    bg='white')
            self.canvas.pack(side="left")
            self.restart()

    def check_game_state(self):
        end, winner = self.board.game_end()
        if end:
            if winner != -1:
                print("Game end. Winner is Player", winner)
                self.handle_show_info("\n\n---Game end. Winner is Player" + str(winner) + "---")
                if winner == 1:
                    again = tk.messagebox.askokcancel(title="Game Over", message="Black Win\nPlay again?")
                else:
                    again = tk.messagebox.askokcancel(title="Game Over", message="White Win\nPlay again?")

            else:
                again = tk.messagebox.askokcancel(title="Game Over", message="Game end. Tie\nPlay again?")
                print("Game end. Tie")
                self.handle_show_info("\n\n---Game end. Tie---")

            if again:
                return end
            else:
                self.destroy()
                exit()

    # Play again
    def restart(self):
        self.check_click_coordinate_press = (np.inf, np.inf)
        self.check_click_coordinate_Release = (np.inf, np.inf)
        self.fist_time_release = True
        self.flag = True
        self.step_counter = 1
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.text_box.destroy()
        self.text_box = tk.Text(self, height=14)
        self.text_box.place(x=602, y=400)
        self.board.init_board(start_player=0)
        self.text_box.config(font=('Courier', 12))
        self.choice_who_first_view()

    def Coordinate(self, x, y):
        X = 0
        Y = 0
        if x < 30 or x > 570 or y < 30 or y > 570:
            return
        else:
            for i in range(30, 930, 60):
                if x > i and x <= i + 60:
                    break
                else:
                    Y = Y + 1

            for i in range(30, 930, 60):
                if y > i and y <= i + 60:
                    break
                else:
                    X = X + 1
            data = [X, Y]
            return data

    def drawXY(self, x, y, current_player):
        drawX = 0
        drawY = 0
        # 判斷劃出棋子的位置並畫出來
        for i in range(0, 9, 1):
            if i == y:
                drawX = 60 + 60 * i
                break
        for i in range(0, 9, 1):
            if i == x:
                drawY = 60 + 60 * i
                break
        if current_player == 1:
            self.canvas.create_oval(drawX - 25, drawY - 25, drawX + 25, drawY + 25, fill="black")
            self.canvas.create_text(drawX, drawY, fill="white", text=str(self.step_counter), font=("courier", 14, "bold"))
        else:
            self.canvas.create_oval(drawX - 25, drawY - 25, drawX + 25, drawY + 25, fill="white")
            self.canvas.create_text(drawX, drawY, fill="black", text=str(self.step_counter), font=("courier", 14, "bold"))
        self.step_counter += 1
        self.update()

    def restart_button_handler(self, args):
        again = tk.messagebox.askokcancel(title="Restart?", message="You sure?")
        if again:
            self.restart()
            self.handle_show_info("Restart")
        else:
            self.handle_show_info("Cancel Restart")

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=0):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()    # 0 or 1
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)