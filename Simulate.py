import py222
from tensorflow.keras.models import load_model
import constants
from MCTS import solveSingleCubeVanillaMCTS, solveSingleCubeGreedy, solveSingleCubeFullMCTS

# Load the trained model
model = load_model(constants.kModelPath)

# Define the depth of scrambling
scrambleDepth = 10  # Adjust as needed

# Create a scrambled cube
scrambledCube = py222.createScrambledCube(scrambleDepth)

# Define maximum number of moves and maximum search depth
maxMoves = 100  # Adjust as needed
maxDepth = 10   # Adjust as needed

# Solve the scrambled cube using different algorithms
algorithms = {
    "Vanilla MCTS": solveSingleCubeVanillaMCTS,
    "Greedy": solveSingleCubeGreedy,
    "Full MCTS": solveSingleCubeFullMCTS
}

for name, algorithm in algorithms.items():
    print(f"Testing {name} algorithm...")
    result, numMoves, solvePath = algorithm(model, scrambledCube, maxMoves, maxDepth)
    if result:
        print(f"Cube solved in {numMoves} moves.")
        print("Solution path:", solvePath)
    else:
        print("Cube not solved within the maximum number of moves.")
