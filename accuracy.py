from steps.load_artifacts import load_scores

accuracy, f1=load_scores()

with open('accuracy.txt', 'w') as file:
    file.write(str(accuracy))