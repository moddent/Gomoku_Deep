# -*- coding: utf-8 -*-
"""
human VS AI models

"""

from __future__ import print_function
from game import Board, Game

if __name__ == '__main__':
    n = 5
    width, height = 9, 9
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)
    game.mainloop()
