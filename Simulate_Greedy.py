from tensorflow.keras.models import load_model
import constants
import py222
from CubeModel import solveSingleCubeGreedy

# Load the trained model
model = load_model(constants.kModelPath)

# Set the parameters
scrambleDepth = 10  # Adjust as needed
maxMoves = 100      # Adjust as needed

# Create a scrambled cube
scrambledCube = py222.createScrambledCube(scrambleDepth)

# Solve the cube using the greedy algorithm
result, numMoves = solveSingleCubeGreedy(model, scrambledCube, maxMoves)

# Check if the cube was solved
if result:
    print("Cube solved in", numMoves, "moves.")
else:
    print("Cube not solved within the maximum number of moves.")
