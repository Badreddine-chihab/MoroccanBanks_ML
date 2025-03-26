import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import rapidfuzz
import pkg_resources  # To get the version of textblob and textblob-fr

# Function to get package version safely
def get_version(pkg_name):
    try:
        return pkg_resources.get_distribution(pkg_name).version
    except pkg_resources.DistributionNotFound:
        return "Not Installed"

libs = {
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "rapidfuzz": rapidfuzz.__version__,
    "textblob": get_version("textblob"),  # Fixed
    "textblob-fr": get_version("textblob-fr"),  # Fixed
}

# Write to requirements.txt
with open("requirements.txt", "w") as f:
    for lib, version in libs.items():
        if version != "Not Installed":
            f.write(f"{lib}=={version}\n")

print("requirements.txt generated successfully!")
