"""
IEEE Research Paper Visualizations Generator
===========================================

This package generates all figures, diagrams, and visualizations required
for the research paper submission.

Author: Mohammed Aashik
Date: October 2, 2025
Purpose: IEEE JBHI/EMBC Paper Submission

Outputs:
- High-resolution figures (300 DPI, PDF format)
- Publication-ready plots
- System architecture diagrams
- Performance comparison charts
- Explainability visualizations
"""

__version__ = "1.0.0"
__author__ = "Mohammed Aashik"

# Color schemes for publication
IEEE_COLORS = {
    "primary": "#003f5c",
    "secondary": "#7a5195",
    "accent": "#ef5675",
    "highlight": "#ffa600",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
}

COLOR_BLIND_SAFE = [
    "#0173b2",  # Blue
    "#de8f05",  # Orange
    "#029e73",  # Green
    "#cc78bc",  # Purple
    "#ca9161",  # Brown
    "#fbafe4",  # Pink
    "#949494",  # Gray
    "#ece133",  # Yellow
]

# IEEE standard figure sizes (inches)
FIGURE_SIZES = {
    "single_column": (3.5, 2.625),  # Single column width
    "double_column": (7.16, 5.37),  # Double column width
    "square": (3.5, 3.5),  # Square single column
    "tall": (3.5, 4.5),  # Tall single column
    "wide": (7.16, 3.0),  # Wide double column
}

# Font sizes for IEEE publications
FONT_SIZES = {"title": 10, "label": 9, "tick": 8, "legend": 8, "annotation": 7}

# DPI settings
DPI_SCREEN = 100
DPI_PRINT = 300
