# Knowledge-Distillation


Knowledge Distillation is an idea for model compression. As the complexity of the model architecture increases it becomes resource intensive and time consuming.
Using knowledge distillation we teach small networks with the help of teacher (large complex) networks.


Algorithm 
 - The teacher network is trained and used for predictions. (soft lables)
 - The student network is trained using soft lables and true lables (hard labels).
 - Find cross entropy loss for both soft lables and hard lables with predictions made by student network.
 - Take weighted average of both the losses and try to minimize the overall loss.

l1 = cross_entropy(student_prediction,teacher_prediction)
l2 = cross_entropy(student_prediction,actual_label)

total_loss = alpha * l1 + (1 - alpha) l2.


Project structure :
  - data.py (download Imagewoof dataset and save it into data/imagewoof-160 folder and read using dataloaders.)
  - utils.py (utility functions for training models)
  - Distiller.py (Distillation algorithm)
  - teacher.py (download and save teacher model)
  - student.py (5 layer cnn as student model, you can make your own architecture)
  - train.py (train the distilled network)


other files : 
  - requirements.txt ( run : python3 -m pip install -r requirements.txt to install all required libraries)
  - train_teacher.py (to train and check validation accuracy of teacher model)
  - train_student.py (to train and check validation accuracy of student model)
  
  
 I used resnet18 from torchvision.models as teacher model.
 link of dataset "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz"
 Used 5 layered CNN as student model.
 
 
 The accuracy of teacher model or resnet18 is 89%
 The accuracy of CNN or student model is 11%
 The accuracy of Distilled network is 39 ~ 40%
 
 It is evident from above results distilled network perfomed almost 4 times better than original model.
 
 
 Clone the git repo 
 install requiremnts
 run data.py -> Distiller.py -> utils.py -> teacher.py -> student.py -> train.py in same order.
