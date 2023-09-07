import copy
import json
from pathlib import Path

import numpy as np
from aifs import diagnostics
from importlib import resources as impresources
from matplotlib.collections import LineCollection


class EquirectangularProjection:
    def __init__(self):
        self.x_offset = 0.0
        self.y_offset = 0.0

    def __call__(self, lon, lat):
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        x = [v - 2 * np.pi if v > np.pi else v for v in lon_rad]
        y = lat_rad
        return x, y

    def inverse(self, x, y):
        lon = np.degrees(x)
        lat = np.degrees(y)
        return lon, lat


class Coastlines:
    def __init__(self, projection=None):
        # Get the path to "continents.json" within your library
        self.continents_file = impresources.files(diagnostics) / "continents.json"

        # Load GeoJSON data from the file
        with self.continents_file.open("rt") as file:
            self.data = json.load(file)

        if projection is None:
            self.projection = EquirectangularProjection()

        self.process_data()

    # Function to extract LineString coordinates
    @staticmethod
    def extract_coordinates(feature):
        return feature["geometry"]["coordinates"]

    def process_data(self):
        lines = []
        for feature in self.data["features"]:
            coordinates = self.extract_coordinates(feature)
            x, y = zip(*coordinates)  # Unzip the coordinates into separate x and y lists

            lines.append(list(zip(*self.projection(x, y))))  # Convert lat/lon to Cartesian coordinates
        self.lines = LineCollection(lines, linewidth=0.5, color="black")

    def plot_continents(self, ax):
        # Add the lines to the axis as a collection
        # Note that we have to provide a copy of the lines, because of Matplotlib
        ax.add_collection(copy.copy(self.lines))
