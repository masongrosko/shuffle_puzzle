"""The game board."""

import random
import time

import cv2
import numpy as np

import tile

TILE_COORDINATES = tuple[int, int]


class Board:
    """The game board."""

    def __init__(
        self, image: str, rows: int, columns: int, window_name: str = "Shuffle Puzzle"
    ) -> None:
        """Init the game board."""
        self.image: np.ndarray = cv2.imread(image)
        self.rows: int = rows
        self.columns: int = columns

        self.window_name: str = window_name

        self.tiles: dict = {}
        self.empty_tile_coords: TILE_COORDINATES = (-1, -1)
        self.available_neighbors: list[TILE_COORDINATES] = []

        self._reset_board()

    def display(self) -> None:
        """Create a window and wait for click."""
        self._update_image()
        cv2.setMouseCallback(self.window_name, self._on_click)
        while True:
            value = self._get_key_press_value()
            if self._check_esc_key(value) is True:
                break
            self._check_wasd_keys(value)
        cv2.destroyAllWindows()

    def shuffle(self) -> None:
        """Shuffle board."""
        if len(self.available_neighbors) < 1:
            self._reset_board()
        for _ in range(25):
            neighbor_coords = random.choice(self.available_neighbors)
            self._switch_tiles(neighbor_coords, self.empty_tile_coords)
            time.sleep(0.01)

    def _update_image(self):
        """Update the displayed image."""
        cv2.imshow(self.window_name, self._image_from_tiles())
        cv2.waitKey(1)

    def _valid_switch(
        self, tile_1_coords: TILE_COORDINATES, tile_2_coords: TILE_COORDINATES
    ) -> bool:
        tile_1_valid: bool = (tile_1_coords == self.empty_tile_coords) | (
            tile_1_coords in self.available_neighbors
        )
        tile_2_valid: bool = (tile_2_coords == self.empty_tile_coords) | (
            tile_2_coords in self.available_neighbors
        )
        return tile_1_valid & tile_2_valid

    def _switch_tiles(
        self, tile_1_coords: TILE_COORDINATES, tile_2_coords: TILE_COORDINATES
    ) -> None:
        """Switch tiles."""
        if self._valid_switch(tile_1_coords, tile_2_coords) is True:
            tile_1 = self.tiles[tile_1_coords[0]][tile_1_coords[1]]
            tile_2 = self.tiles[tile_2_coords[0]][tile_2_coords[1]]
            self.tiles[tile_1_coords[0]][tile_1_coords[1]] = tile_2
            self.tiles[tile_2_coords[0]][tile_2_coords[1]] = tile_1
            self._update_image()
            if tile_1_coords == self.empty_tile_coords:
                self._set_empty_tile(tile_2_coords)
            else:
                self._set_empty_tile(tile_1_coords)

    def _is_solved(self) -> bool:
        """Check to see if board is solved."""
        solved = True
        for row, cols in self.tiles.items():
            for col, game_tile in cols.items():
                solved *= game_tile.cell == (row, col)

        return solved

    def _create_tiles(self) -> None:
        """Create a tile for each cell on the game board."""
        self._validate_input()
        self._reset_tiles()
        for row in range(self.rows):
            self.tiles[row] = {}
            for col in range(self.columns):
                cell_image = self._get_cell_image(row, col)
                self.tiles[row][col] = tile.Tile(cell=(row, col), image=cell_image)

    def _remove_last_tile(self) -> None:
        """Remove tile to create slide puzzle."""
        last_row = self.rows - 1
        last_column = self.columns - 1
        self.tiles[last_row][last_column].image *= 0

        self._set_empty_tile((last_row, last_column))

    def _on_click(self, event: int, x: int, y: int, flags: int, params: None) -> None:
        """When user clicks on board."""
        if event == 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            tile_coords = self._get_tile_coords_from_mouse_location(x, y)
            if tile_coords in self.available_neighbors:
                self._switch_tiles(tile_coords, self.empty_tile_coords)
                self._set_empty_tile(tile_coords)
            print(f"{self._is_solved()=}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.shuffle()
            self._print_board_state()
            print("shuffle")

    def _reset_tiles(self) -> None:
        """Reset all the tiles."""
        self.tiles = {}

    def _reset_board(self) -> None:
        """Reset the board."""
        self._create_tiles()
        self._remove_last_tile()

    def _set_empty_tile(self, empty_tile_coords: TILE_COORDINATES) -> None:
        """Update empty tile and available neighbors."""
        self.empty_tile_coords = empty_tile_coords
        self.available_neighbors = self._get_valid_neighbors(
            empty_tile_coords[0], empty_tile_coords[1]
        )

    def _get_valid_neighbors(self, row: int, col: int) -> list[TILE_COORDINATES]:
        """Return list of valid neighbor tiles."""
        possible_neighbors = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        neighbors = [
            x
            for x in possible_neighbors
            if ((x[0] >= 0) & (x[0] < self.rows))
            and ((x[1] >= 0) & (x[1] < self.columns))
        ]

        return neighbors

    def _get_cell_image(self, row: int, col: int) -> np.ndarray:
        """Determine which part of the image should go with the (row, col) provided."""
        len_image_y, len_image_x, _ = self.image.shape

        start_x, end_x = self._get_subsect_bounds(col, self.columns, len_image_x)
        start_y, end_y = self._get_subsect_bounds(row, self.rows, len_image_y)

        return self.image[start_y:end_y, start_x:end_x]

    def _image_from_tiles(self) -> None:
        """Concatenate tiles to create an image."""
        row_images = []
        for cols in self.tiles.values():
            row_image = []
            for game_tile in cols.values():
                row_image.append(game_tile.image)

            row_images.append(np.concatenate(row_image, axis=1))

        return np.concatenate(row_images, axis=0)

    def _print_board_state(self) -> None:
        """Print board state."""
        for cols in self.tiles.values():
            row_to_print = []
            for game_tile in cols.values():
                row_to_print.append(game_tile.cell)
            print(row_to_print)

    def _validate_input(self):
        """Make sure input is valid."""
        if self.rows < 1:
            raise ValueError(
                f"value of {self.rows} given for rows, must be at least 1."
            )

        if self.columns < 1:
            raise ValueError(
                f"value of {self.columns} given for columns, must be at least 1."
            )

    def _get_tile_coords_from_mouse_location(self, x, y) -> TILE_COORDINATES:
        """Get tile from mouse_location."""
        y_max, x_max, _ = self.image.shape

        row = y // (y_max / self.rows)
        col = x // (x_max / self.columns)

        print(x, x_max, y, y_max, row, col)

        return row, col

    def _get_key_press_value(self) -> int:
        """Return decoded key press."""
        out = cv2.waitKey(100) & 0xFF
        if out != 255:
            print(out)
        return out

    def _check_esc_key(self, value: int) -> bool:
        """Check to see if the escape key is pressed."""
        return value == 27

    def _check_wasd_keys(self, value: int) -> None:
        """Check wasd keys to see if pressed."""
        wasd_map = {
            ord("w"): (1, 0),
            ord("a"): (0, 1),
            ord("s"): (-1, 0),
            ord("d"): (0, -1),
        }
        tile_delta = wasd_map.get(value)
        if tile_delta is not None:
            print(tile_delta)
            self._switch_tiles(
                self.empty_tile_coords,
                self._add_tuples(self.empty_tile_coords, tile_delta),
            )
            print(f"{self._is_solved()=}")

    @staticmethod
    def _add_tuples(tuple_1: tuple, tuple_2: tuple) -> tuple:
        """Add values of two tuples together."""
        return tuple(map(sum, zip(tuple_1, tuple_2)))

    @staticmethod
    def _get_subsect_bounds(
        subsect: int, total_subsects: int, total_length: int
    ) -> tuple[int, int]:
        """Calcualte start and end bounds of a subsection of a line."""
        subsect_length = total_length // total_subsects
        start = subsect * subsect_length
        end = (subsect + 1) * subsect_length

        return start, end
