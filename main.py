"""Run the game."""

import board


def main(image, rows, columns):
    """Run the game."""
    game_board = board.Board(image, rows, columns)
    game_board.shuffle()
    game_board.display()


if __name__ == "__main__":
    image = r"C:\Users\Mason Grosko\OneDrive\Pictures\roy_wilkins.PNG"
    rows = 4
    columns = 3

    main(image, rows, columns)
