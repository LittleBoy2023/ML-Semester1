# Potato_Net

## Introduciton
Autism spectrum disorders (ASD) comprise a range of neurodevelopmental disorders characterized by limitations in social interaction, communication, and repetitive behaviors. Repetitive hand motions are a common stimulus behavior for individuals with autism, serving as a nonverbal communication cue and self-regulatory mechanism.
The project examined the signature hand movements of individuals with autism and used machine learning algorithms focused on identifying three specific repetitive behaviors in children with autism: arm flapping, hand clapping, and finger rubbing.
The aim is to support early and accurate ASD diagnosis. Through advanced video analytics and pattern recognition techniques, we strive to contribute to a deeper understanding of ASD behavioral markers.


## Requirements

- matplotlib==3.8.2
- numpy==1.24.1
- opencv-python==4.8.1.78
- Pillow==9.3.0
- six==1.16.0
- tensorboard==2.15.1
- tensorboard-data-server==0.7.2
- torch==2.1.2+cu121
- torchvision==0.16.2+cu121

## Structure

```sh
C:.
│  clip_cap.py
│  main_for_ori_model.py
│  main_for_potatonet.py
│  ori_model.py
│  potato_model.py
│  README.md
│  vid2fram.py
├─data
│  ├─label    
│  ├─out    
│  └─vid        
├─model
│  ├─ori
│  │      PoseModel_whole.pt
│  │      
│  └─PotatoNet
│          PoseModel_epoch60.pt       
└─runs
    ├─pose_model_20240112-133345
    │      events.out.tfevents.1705037625.albyq.86228.0
    │      
    └─pose_model_20240112-145836
            events.out.tfevents.1705042716.albyq.95204.0
```

## Data analysis
* Raw data
![data1](https://img.lu/upload/43ae5e14b8c6541a99a73.png)
* Segment


![data1](https://img.lu/upload/5674f8993d74d5dbebed4.png)
* Category


![data1](https://img.lu/upload/b99446e3fe4172864e799.png)

## Model

### Structure

![data1](https://img.lu/upload/62deb254ab45d56487dfb.png)

### Result

![data1](https://img.lu/upload/2ccb0edebc1d090885d96.png)

![data1](https://img.lu/upload/e6cd5f76bba1f0d6a9eb9.png)

