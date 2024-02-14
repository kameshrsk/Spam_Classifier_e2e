from api import model_dict

accuracy=model_dict['accuracy']

with open('accuracy.txt', 'w') as file:
    file.write(str(accuracy))