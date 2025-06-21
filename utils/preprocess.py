import numpy as np

def preprocess_input(data: dict):
    """
    Convert input strings to model-ready numerical format.
    """
    try:
        # Categorical encodings
        sex_map = {"M": 1, "F": 0}
        cp_map = {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}
        ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
        angina_map = {"N": 0, "Y": 1}
        slope_map = {"Up": 0, "Flat": 1, "Down": 2}

        values = [
            int(data["Age"]),
            sex_map[data["Sex"]],
            cp_map[data["ChestPainType"]],
            int(data["RestingBP"]),
            int(data["Cholesterol"]),
            int(data["FastingBS"]),
            ecg_map[data["RestingECG"]],
            int(data["MaxHR"]),
            angina_map[data["ExerciseAngina"]],
            float(data["Oldpeak"]),
            slope_map[data["ST_Slope"]],
        ]

        return np.array(values).reshape(1, -1)

    except KeyError as e:
        raise ValueError(f"Missing or invalid value: {str(e)}")
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")
